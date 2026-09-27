#include <iostream>
#include <thread>

#include "main/renderer.h"
#include "material/description.h"
#include "scene/interop.h"

using namespace krr;

int finalized{};

class SessionProbePass : public RenderPass {
public:
	void finalize() override { ++finalized; }
	friend void from_json(const json &, SessionProbePass &) {}
};

RenderPassRegister<SessionProbePass> sessionProbe("SessionProbePass");

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

void compare(const std::vector<float> &a, const std::vector<float> &b) {
	require(a.size() == b.size(), "Image dimensions differ");
	for (size_t i = 0; i < a.size(); ++i) {
		if (std::isfinite(a[i]) && std::abs(a[i] - b[i]) <= 1e-7f + 1e-6f * std::abs(b[i])) continue;
		double error = 0.0, reference = 0.0;
		for (size_t pixel = 0; pixel < a.size(); ++pixel) {
			error += double(a[pixel] - b[pixel]) * double(a[pixel] - b[pixel]);
			reference += double(b[pixel]) * double(b[pixel]);
		}
		std::cerr << "Image mismatch at " << i << ": " << a[i] << " versus " << b[i]
			<< "; NRMSE=" << std::sqrt(error / reference) << '\n';
		throw std::runtime_error("Progressive session differs from the fresh batch");
	}
}

void checkContext(CUcontext expected) {
	CUcontext current{};
	require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == expected,
		"Session did not restore the caller's CUDA context");
}

void reportMemory(const char *stage) {
	CUcontext previous{};
	size_t free{}, total{};
	require(cuCtxGetCurrent(&previous) == CUDA_SUCCESS, "Could not read CUDA context for memory reporting");
	require(cuCtxSetCurrent(gpContext->cudaContext) == CUDA_SUCCESS, "Could not bind CUDA context for memory reporting");
	CUresult result = cuMemGetInfo(&free, &total);
	require(cuCtxSetCurrent(previous) == CUDA_SUCCESS, "Could not restore CUDA context after memory reporting");
	require(result == CUDA_SUCCESS, "Could not query free CUDA memory");
	std::cout << stage << ": CUDA free=" << free << ", total=" << total
		<< ", tracked=" << CUDATrackedMemory::singleton.BytesAllocated() << '\n';
}

class ForeignContext {
public:
	ForeignContext() {
		require(cuCtxGetCurrent(&previous) == CUDA_SUCCESS, "Could not read the caller context");
#if CUDA_VERSION >= 13000
		require(cuCtxCreate(&context, nullptr, 0, 0) == CUDA_SUCCESS, "Could not create a caller-owned context");
#else
		require(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "Could not create a caller-owned context");
#endif
	}
	~ForeignContext() { cuCtxDestroy(context); cuCtxSetCurrent(previous); }
	CUcontext context{}, previous{};
};

int main(int argc, char **argv) {
	try {
		std::cout << std::unitbuf;
		json config = File::loadJSON(fs::path(KRR_PROJECT_DIR) / "tests/cases/cornell_wavefront/config.json");
		config["resolution"] = {32, 32};
		config["graphics_api"] = argc > 1 ? argv[1] : "vulkan";
		config["passes"].push_back({{"name", "SessionProbePass"}});
		std::vector<float> expected;
		{
			HeadlessRenderer batch(config);
			require(!gpContext && !batch.getDevice(), "Constructing a headless renderer initialized the GPU");
			expected = batch.render(4, 17);
		}
		const size_t idleBytes = CUDATrackedMemory::singleton.BytesAllocated();
		const int finalizedBatches = finalized;
		std::cout << "Fresh batch completed\n";
		reportMemory("Warmed idle");
		CUcontext previous{};
		require(cuCtxGetCurrent(&previous) == CUDA_SUCCESS, "Could not read current CUDA context");
		RenderSession::Snapshot retained;
		std::shared_ptr<const RenderSession::DepthSnapshot> retainedDepth;
		std::optional<RenderSession::SnapshotRequest> abandoned;
		{
			RenderSession session(config);
			session.initialize(17);
			checkContext(previous);
			bool rejected = false;
			std::thread foreign([&] {
				try { session.snapshot(); } catch (const std::runtime_error &) { rejected = true; }
			});
			foreign.join();
			require(rejected && !session.isClosed(), "Session accepted foreign-thread access");
			auto scene = session.getScene();
			require(session.step(1) == 1, "First step did not render one frame");
			auto first = session.snapshot();
			require(!first.depth && session.getDepthCaptureCount() == 0,
				"Color-only snapshots captured primary depth");
			auto pendingFirst = session.requestSnapshot();
			require(bool(pendingFirst), "Could not enqueue the first snapshot");
			require(session.step(3) == 3, "Continuation did not render three frames");
			auto pendingFinal = session.requestSnapshot();
			require(bool(pendingFinal) && !session.requestSnapshot(), "Readback queue did not enforce its capacity");
			auto asyncFirst = session.collectSnapshot(*pendingFirst, true);
			compare(asyncFirst.image, first.image);
			require(asyncFirst.completedFrames == 1, "Queued snapshot lost its sample index");
			session.wait();
			require(session.isSnapshotReady(*pendingFinal), "Completed copy did not signal readiness");
			auto asyncFinal = session.collectSnapshot(*pendingFinal);
			compare(asyncFinal.image, expected);
			require(asyncFinal.completedFrames == 4, "Final queued snapshot has the wrong sample index");
			rejected = false;
			try { session.collectSnapshot(*pendingFinal); } catch (const std::invalid_argument &) { rejected = true; }
			require(rejected && !session.isClosed(), "Consumed snapshot was accepted or closed the session");
			retained = session.snapshot();
			require(finalized == finalizedBatches, "Intermediate snapshot finalized passes");
			require(retained.completedFrames == 4 && first.completedFrames == 1, "Completed frame count is incorrect");
			compare(retained.image, expected);
			std::cout << "Progressive stepping completed\n";
			require(session.getScene() == scene, "Progressive steps replaced the scene");
			auto retainedCopy = retained.image;
			first.image[0] += 1.f;
			compare(retained.image, retainedCopy);
			session.reset(17);
			require(session.getCompletedFrames() == 0 && session.getGeneration() > retained.generation,
				"Reset did not start a new accumulation generation");
			session.step(4);
			compare(session.snapshot().image, expected);
			auto stale = session.requestSnapshot();
			session.reset(17);
			require(stale && !session.isSnapshotReady(*stale), "Reset accepted an old-generation snapshot");
			rejected = false;
			try { session.collectSnapshot(*stale, true); } catch (const std::invalid_argument &) { rejected = true; }
			require(rejected && !session.isClosed(), "Stale snapshot was accepted or closed the session");
			stale.reset();
			session.step(4);
			std::thread cancel([&] { session.requestCancel(); });
			cancel.join();
			require(session.step(2) == 0 && session.getCompletedFrames() == 4, "Cancellation did not stop frame stepping");
			session.reset(17);
			session.step(4);
			compare(session.snapshot().image, expected);
			std::cout << "Reset and cancellation completed\n";
			auto rtScene = scene->getSceneRT();
			auto optixScene = rtScene->getOptixScene();
			auto geometry = rtScene->getMeshData()[0].positions.data();
			const auto lightCount = rtScene->getLightData().size();
			require(lightCount > 0, "Cornell fixture has no sampleable lights");
			std::vector<std::pair<Material::SharedPtr, Material::SharedPtr>> emitters;
			for (auto &material : scene->getMaterials()) {
				if (!material->hasEmission()) continue;
				auto original = std::make_shared<Material>();
				original->updateFrom(*material);
				emitters.emplace_back(material, original);
				Material disabled;
				disabled.updateFrom(*material);
				disabled.setTexture(Material::TextureType::Emissive, nullptr);
				disabled.setDescription(nullptr);
				material->updateFrom(disabled);
			}
			session.reset(17);
			session.step(4);
			auto dark = session.snapshot();
			require(rtScene->getLightData().empty(), "Disabling emission retained lights in the sampler");
			for (float value : dark.image) require(value == 0.f, "Scene without lights is not black");
			for (auto &[material, original] : emitters) material->updateFrom(*original);
			session.reset(17);
			session.step(4);
			compare(session.snapshot().image, expected);
			require(rtScene->getLightData().size() == lightCount && scene->getSceneRT() == rtScene &&
				rtScene->getOptixScene() == optixScene && rtScene->getMeshData()[0].positions.data() == geometry,
				"Material updates replaced geometry or failed to restore emission sampling");
			std::cout << "Material emission updates completed\n";
			require(!rtScene->hasAuthoredMaterials(), "Legacy scene selected authored scattering");
			auto editedMaterial = *std::find_if(scene->getMaterials().begin(), scene->getMaterials().end(),
				[](const auto &material) { return !material->hasEmission(); });
			Material originalMaterial;
			originalMaterial.updateFrom(*editedMaterial);
			auto description = std::make_shared<MaterialDescription>();
			description->model = MaterialModel::PreviewSurface;
			editedMaterial->setDescription(description);
			session.reset(17);
			session.step(1);
			auto authored = session.snapshot();
			require(rtScene->hasAuthoredMaterials(), "Material edit did not select authored scattering");
			for (float value : authored.image) require(std::isfinite(value), "Authored scattering returned an invalid value");
			editedMaterial->updateFrom(originalMaterial);
			session.reset(17);
			session.step(4);
			compare(session.snapshot().image, expected);
			require(!rtScene->hasAuthoredMaterials() && rtScene->getOptixScene() == optixScene,
				"Restoring legacy material retained authored scattering or rebuilt the scene");
			auto instanceNode = scene->getMeshInstances()[0]->getNode();
			const Affine3f originalTransform = instanceNode->getLocalTransform();
			Affine3f movedTransform = originalTransform;
			movedTransform.translation()[0] += .5f;
			instanceNode->setLocalTransform(movedTransform);
			session.reset(17);
			session.step(4);
			require(session.snapshot().image != expected && rtScene->getMeshData()[0].positions.data() == geometry,
				"Instance transform update was ignored or replaced geometry");
			instanceNode->setLocalTransform(originalTransform);
			session.reset(17);
			session.step(4);
			compare(session.snapshot().image, expected);
			std::cout << "Instance transform updates completed\n";
			CameraState camera;
			camera.cameraToWorld = scene->getCamera()->getTransform().matrix();
			camera.projection = scene->getCamera()->getProjectionMatrix();
			camera.nearClip = 0.f;
			camera.cameraToWorld(0, 3) += .2f;
			session.updateCamera(camera);
			require(session.getScene() == scene && !scene->getCameraController(), "Camera update replaced geometry or retained controller");
			session.step(2);
			auto moved = session.snapshot(true);
			require(moved.image != expected && moved.completedFrames == 2, "Camera update did not reset and change rendering");
			require(moved.depth && moved.depth->linear.size() == 32 * 32 && moved.depth->projected.size() == 32 * 32,
				"Depth snapshot has incorrect dimensions");
			require(std::any_of(moved.depth->linear.begin(), moved.depth->linear.end(), [](float depth) {
				return std::isfinite(depth) && depth > 0.f;
			}), "Depth snapshot contains no geometry hits");
			for (float depth : moved.depth->projected)
				require(std::isfinite(depth) && depth >= 0.f && depth <= 1.f, "Projected depth is invalid");
			const auto captures = session.getDepthCaptureCount();
			auto cached = session.snapshot(true);
			require(cached.depth == moved.depth && session.getDepthCaptureCount() == captures,
				"Repeated snapshot retraced or copied unchanged depth");
			session.step(3);
			session.wait();
			require(session.snapshot(true).depth == moved.depth && session.getDepthCaptureCount() == captures,
				"Progressive samples invalidated primary depth");
			instanceNode->setLocalTransform(movedTransform);
			session.step(1);
			auto transformed = session.snapshot(true);
			require(transformed.depth != moved.depth && session.getDepthCaptureCount() == captures + 1,
				"Instance edits retained stale depth");
			instanceNode->setLocalTransform(originalTransform);
			session.reset(17);
			session.step(1);
			auto restored = session.snapshot(true);
			require(restored.depth != transformed.depth && restored.depth->linear == moved.depth->linear,
				"Reset failed to restore depth or mutated a retained snapshot");
			Material depthMaterial;
			depthMaterial.updateFrom(*editedMaterial);
			editedMaterial->updateFrom(depthMaterial);
			session.step(1);
			require(session.snapshot(true).depth != restored.depth,
				"Material edits retained the depth cache");
			std::cout << "Camera and depth updates completed\n";
			auto beforeResize = session.requestSnapshot(true);
			session.resize({16, 24});
			require(beforeResize && !session.isSnapshotReady(*beforeResize), "Resize retained a stale readback");
			beforeResize.reset();
			session.step(1);
			auto resized = session.snapshot(true);
			require(resized.size == Vector2i(16, 24) && resized.image.size() == 16 * 24 * 3,
				"Resize did not update the rendered image");
			require(resized.depth && resized.depth->linear.size() == 16 * 24 && resized.depth != restored.depth,
				"Resize retained depth at the old dimensions");
			retainedDepth = resized.depth;
			abandoned = session.requestSnapshot(true);
			require(bool(abandoned), "Could not enqueue a snapshot before cleanup");
			session.finish();
			session.finish();
			require(finalized == finalizedBatches + 1, "Explicit finish must finalize once");
			session.close();
			session.close();
			require(finalized == finalizedBatches + 1 && !scene->getSceneRT(),
				"Close finalized passes or retained external scene GPU resources");
			checkContext(previous);
		}
		require(retainedDepth && retainedDepth->linear.size() == 16 * 24,
			"Closing the session invalidated a retained depth snapshot");
		abandoned.reset();
		compare(retained.image, expected);
		require(CUDATrackedMemory::singleton.BytesAllocated() == idleBytes, "Session leaked tracked GPU memory");
		{
			auto supplied = std::make_shared<Scene>();
			require(SceneImporter().import(config["scene"], supplied), "Could not create supplied scene");
			supplied->update(0, 0.0);
			json overrideConfig = config;
			overrideConfig.erase("scene");
			overrideConfig["model"] = "missing-model-for-scene-override.obj";
			RenderSession session(overrideConfig);
			session.initialize(17, supplied);
			session.step(4);
			compare(session.snapshot().image, expected);
			auto replacement = std::make_shared<Scene>();
			require(SceneImporter().import(config["scene"], replacement), "Could not create replacement scene");
			replacement->update(0, 0.0);
			session.replaceScene(replacement);
			require(!supplied->getSceneRT() && session.getScene() == replacement,
				"Scene replacement retained old resources or ignored the replacement");
			session.step(4);
			compare(session.snapshot().image, expected);
			session.close();
			require(!replacement->getSceneRT(), "Close retained replacement GPU resources");
		}
		require(CUDATrackedMemory::singleton.BytesAllocated() == idleBytes, "Scene replacement leaked GPU memory");
		for (bool multilevel : {false, true}) {
			json hierarchyConfig = config;
			hierarchyConfig["scene"]["options"]["multilevel"] = multilevel;
			RenderSession session(hierarchyConfig);
			session.initialize(17);
			session.step(1);
			const auto initial = session.snapshot(true);
			auto scene = session.getScene();
			auto rtScene = scene->getSceneRT();
			scene->setCameraController(nullptr);
			const auto geometry = rtScene->getMeshData()[0].positions.data();
			for (auto *node : {scene->getMeshInstances()[0]->getNode(), scene->getSceneGraph()->getRoot().get()}) {
				const Affine3f original = node->getLocalTransform();
				auto *cameraNode = scene->getCamera()->getNode();
				const Affine3f cameraOriginal = cameraNode->getLocalTransform();
				Affine3f moved = original;
				moved.translation()[0] += .5f;
				node->setLocalTransform(moved);
				if (node == scene->getSceneGraph()->getRoot().get()) {
					Affine3f cameraFixed = cameraOriginal;
					cameraFixed.translation()[0] -= .5f;
					cameraNode->setLocalTransform(cameraFixed);
				}
				session.reset(17);
				session.step(1);
				require(session.snapshot().image != initial.image,
					"Hierarchy transform did not affect the first frame after a quiet frame");
				node->setLocalTransform(original);
				cameraNode->setLocalTransform(cameraOriginal);
				session.reset(17);
				session.step(1);
				compare(session.snapshot().image, initial.image);
				require(rtScene->getMeshData()[0].positions.data() == geometry,
					"Hierarchy transform rebuilt geometry");
			}
			session.close();
		}
		std::cout << "Single- and multilevel hierarchy transforms completed\n";
		for (const char *integrator : {"WavefrontPathTracer", "MegakernelPathTracer"}) {
			for (bool multilevel : {false, true}) {
				json lightConfig = config;
				lightConfig["passes"][0]["name"] = integrator;
				auto sceneConfig = config["scene"];
				sceneConfig.erase("model");
				sceneConfig["options"]["multilevel"] = multilevel;
				auto scene = std::make_shared<Scene>();
				require(SceneImporter().import(sceneConfig, scene), "Could not create light-only scene");
				scene->update(0, 0.0);
				interop::LightInput light;
				light.type = interop::LightInput::Type::Rectangle;
				light.width = light.height = 2.f;
				light.transform = scene->getCamera()->getTransform();
				light.transform.translate(Vector3f{0.f, 0.f, -1.f});
				light.transform.linear().col(0) *= -1.f;
				light.transform.linear().col(2) *= -1.f;
				interop::attachLight(scene, light);
				RenderSession session(lightConfig);
				session.initialize(17, scene);
				session.step(1);
				auto hidden = session.snapshot(true);
				for (float value : hidden.image) require(value == 0.f, "Analytic light is visible to a camera ray");
				for (float depth : hidden.depth->linear) require(std::isinf(depth), "Hidden analytic light appears in primary depth");
				for (auto &mesh : scene->getMeshes()) mesh->cameraVisible = true;
				session.initialize(17, scene);
				session.step(1);
				auto visible = session.snapshot(true);
				require(std::any_of(visible.image.begin(), visible.image.end(), [](float value) { return value > 0.f; }),
					"Camera-visible emissive geometry did not appear");
				require(std::any_of(visible.depth->linear.begin(), visible.depth->linear.end(), [](float value) { return std::isfinite(value); }),
					"Camera-visible emissive geometry is missing from depth");
				session.close();
			}
		}
		std::cout << "Analytic light camera visibility completed\n";
		for (int i = 0; i < 2; ++i) {
			std::exception_ptr error;
			std::thread worker([&] {
				try {
					RenderSession session(config);
					session.initialize(17);
					session.step(4);
					compare(session.snapshot().image, expected);
					session.close();
					checkContext(nullptr);
				} catch (...) { error = std::current_exception(); }
			});
			worker.join();
			if (error) std::rethrow_exception(error);
			require(CUDATrackedMemory::singleton.BytesAllocated() == idleBytes, "Sequential worker session leaked GPU memory");
			reportMemory("Sequential worker idle");
		}
		{
			ForeignContext caller;
			RenderSession session(config);
			session.initialize(17);
			checkContext(caller.context);
			session.step(4);
			checkContext(caller.context);
			compare(session.snapshot().image, expected);
			session.close();
			checkContext(caller.context);
			gpContext.reset();
			checkContext(caller.context);
			require(CUDATrackedMemory::singleton.BytesAllocated() == 0, "Context cleanup retained tracked GPU memory");
			require(cuCtxSynchronize() == CUDA_SUCCESS, "Cleanup destroyed the caller-owned CUDA context");
		}
		std::cout << "Render session stepping, reset, camera, depth, resize, cancellation and cleanup passed\n";
		return 0;
	} catch (const std::exception &error) {
		Renderer::closeActive();
		if (gpContext) cuCtxSetCurrent(gpContext->cudaContext);
		gpContext.reset();
		std::cerr << error.what() << '\n';
		return 1;
	}
}
