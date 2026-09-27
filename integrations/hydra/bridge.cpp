#include "bridge.h"
#include "publication.h"
#include <chrono>
#include <deque>
#include <fstream>
#include <regex>

namespace krr::hydra {
namespace {

class RenderWorker {
public:
	RenderWorker() : mThread([this] { run(); }) {}
	~RenderWorker() {
		{
			std::lock_guard lock(mMutex);
			mQuit = true;
		}
		mWake.notify_all();
		mThread.join();
	}
	std::shared_ptr<RenderState> create() {
		auto state = std::make_shared<RenderState>();
		std::lock_guard lock(mMutex);
		mStates.push_back(state);
		return state;
	}
	void wake() { mWake.notify_all(); }

private:
	std::shared_ptr<RenderState> next() {
		std::shared_ptr<RenderState> result;
		int priority		 = -1;
		bool currentViewport = false;
		if (mCurrent) {
			std::lock_guard stateLock(mCurrent->mutex);
			currentViewport = mCurrent->priority == 1;
		}
		std::lock_guard lock(mMutex);
		for (auto it = mStates.begin(); it != mStates.end();) {
			auto state = it->lock();
			if (!state) {
				it = mStates.erase(it);
				continue;
			}
			++it;
			std::lock_guard stateLock(state->mutex);
			if (state->alive && state->ready && !state->paused && !state->converged &&
				state->error.empty() && state->size[0] > 0 && state->size[1] > 0 &&
				state->priority > priority) {
				if (state->priority == 1 && mCurrent && state != mCurrent && currentViewport) {
					state->error = "KiRaRay supports one rendered viewport at a time; disable the "
								   "other rendered viewport";
					state->converged = true;
					if (!state->statusPath.empty())
						std::ofstream(state->statusPath) << json{{"error", state->error}};
					continue;
				}
				result	 = state;
				priority = state->priority;
			}
		}
		return result;
	}
	void release() {
		mPending.clear();
		mSession.reset();
		mNodes.clear();
		mMaterials.clear();
		mDiagnostics.clear();
		if (mCurrent) {
			std::lock_guard lock(mCurrent->mutex);
			mCurrent->active = false;
			mCurrent->released.notify_all();
		}
		mCurrent.reset();
		mVersion = mGeometryVersion = mTransformVersion = mMaterialVersion = 0;
		mPublication.reset();
	}
	Scene::SharedPtr buildScene(const SceneInput &input) {
		++mSceneBuilds;
		mLights = json::array();
		for (const auto &[id, light] : input.lights)
			mLights.push_back({{"name", id},
							   {"intensity", light.intensity},
							   {"exposure", light.exposure},
							   {"normalize", light.normalize},
							   {"width", light.width},
							   {"height", light.height},
							   {"radius", light.radius}});
		auto scene = std::make_shared<Scene>();
		scene->getSceneGraph()->setScene(scene);
		scene->getSceneGraph()->setRoot(std::make_shared<SceneGraphNode>());
		scene->getSceneGraph()->attachLeaf(scene->getSceneGraph()->getRoot(), scene->getCamera(),
										   "Camera");
		scene->setCamera(scene->getCamera());
		scene->setAnimated(false);
		scene->setCameraController(nullptr);
		mMaterials.clear();
		mDiagnostics.clear();
		for (const auto &[id, network] : input.materials) {
			auto copy					= network;
			copy.emissionLuminanceScale = input.emissionLuminanceScale;
			if (auto found = input.diagnostics.find(id); found != input.diagnostics.end())
				copy.diagnostics.insert(copy.diagnostics.end(), found->second.begin(),
										found->second.end());
			mMaterials[id] = interop::translateMaterial(copy, &mDiagnostics);
		}
		mNodes.clear();
		std::map<std::string, std::vector<Mesh::SharedPtr>> tangentGroups;
		for (const auto &[id, record] : input.meshes) {
			if (!record.visible) continue;
			auto meshInput = record.input;
			for (const auto &binding : record.materials) {
				auto &material = mMaterials[binding];
				if (!material && binding.empty()) {
					material = std::make_shared<Material>();
					material->setName("Hydra default");
					material->mMaterialParams.diffuse = RGBA{.18f, .18f, .18f, 1.f};
				} else if (!material) {
					mDiagnostics.push_back("Missing bound material: " + binding);
					Log(Warning, "%s", mDiagnostics.back().c_str());
					material = interop::errorMaterial(binding);
				}
				meshInput.materials.push_back(material);
			}
			auto meshes = interop::makeMeshes(meshInput);
			if (input.blenderScene) {
				static const std::regex submesh(
					"^(/scene/(?:Instancer/)?O_[0-9A-F]{16})/SM_[0-9]{4,}$");
				std::smatch match;
				if (std::regex_match(id, match, submesh)) {
					auto &group = tangentGroups[match[1].str()];
					group.insert(group.end(), meshes.begin(), meshes.end());
				}
			}
			for (const auto &transform : record.transforms)
				mNodes[id].push_back(interop::attachMeshes(scene, meshes, Affine3f(transform), id));
		}
		for (const auto &[id, meshes] : tangentGroups)
			if (meshes.size() > 1) interop::generateTangents(meshes);
		for (const auto &[id, light] : input.lights) interop::attachLight(scene, light);
		return scene;
	}
	void updateMaterials(const SceneInput &input) {
		++mMaterialUpdates;
		mDiagnostics.clear();
		for (const auto &[id, network] : input.materials) {
			auto found = mMaterials.find(id);
			if (found == mMaterials.end()) continue;
			auto copy					= network;
			copy.emissionLuminanceScale = input.emissionLuminanceScale;
			if (auto diagnostics = input.diagnostics.find(id);
				diagnostics != input.diagnostics.end())
				copy.diagnostics.insert(copy.diagnostics.end(), diagnostics->second.begin(),
										diagnostics->second.end());
			found->second->updateFrom(*interop::translateMaterial(copy, &mDiagnostics));
		}
	}
	void updateTransforms(const SceneInput &input) {
		++mTransformUpdates;
		for (const auto &[id, record] : input.meshes) {
			auto found = mNodes.find(id);
			if (found == mNodes.end()) continue;
			if (found->second.size() != record.transforms.size())
				throw std::runtime_error("Instance count changed without rebuilding the scene");
			for (size_t i = 0; i < record.transforms.size(); ++i)
				found->second[i]->setLocalTransform(Affine3f(record.transforms[i]));
		}
	}
	void render(const std::shared_ptr<RenderState> &state) {
		const auto started = std::chrono::steady_clock::now();
		auto milliseconds  = [](auto from) {
			 return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
															  from)
				 .count();
		};
		SceneInput scene;
		CameraState camera;
		Vector2i size;
		std::string root, graphicsApi;
		uint64_t version, seed;
		uint32_t samples;
		bool depthRequested;
		{
			std::lock_guard lock(state->mutex);
			version		= state->version;
			seed		= state->seed;
			samples		= state->samples;
			camera		= state->camera;
			size		= state->size;
			root		= state->assetRoot;
			graphicsApi = state->graphicsApi;
			depthRequested = state->depthRequested;
			if (version != mVersion || state != mCurrent) scene = state->scene;
		}
		if (mSession && (graphicsApi != mGraphicsApi || root != mAssetRoot)) release();
		if (state != mCurrent) {
			release();
			mCurrent = state;
			{
				std::lock_guard lock(state->mutex);
				state->active = true;
			}
		}
		if (!mSession) {
			auto &context = Context::ensureInitialized();
			if (cuCtxSetCurrent(context.cudaContext) != CUDA_SUCCESS)
				throw std::runtime_error("Could not bind the Hydra worker's CUDA context");
			json config{{"resolution", {size[0], size[1]}},
						{"graphics_api", graphicsApi},
						{"scene", json::object()},
						{"passes",
						 {{{"name", "WavefrontPathTracer"},
						   {"params", {{"nee", true}, {"rr", 0.8}, {"max_depth", 10}}}},
						  {{"name", "AccumulatePass"},
						   {"params",
							{{"spp", 0},
							 {"mode", "accumulate"},
							 {"save_on_finish", false},
							 {"exit_on_finish", false}}}}}}};
			mSession	 = std::make_unique<RenderSession>(config, root, false);
			mGraphicsApi = graphicsApi;
			mAssetRoot	 = root;
			mSession->initialize(seed, buildScene(scene));
			mInitializationMs = milliseconds(started);
			mGeometryVersion  = scene.geometryVersion;
			mTransformVersion = scene.transformVersion;
			mMaterialVersion  = scene.materialVersion;
		}
		if (version != mVersion) {
			mPending.clear();
			mUpdateStarted = started;
			mPublication.reset();
			mPublicationCount = mAsyncPublicationCount = mSnapshotRequests = mPendingPeak = 0;
			mReadbackTotalMs = mRenderStepMs = mWaitMs = 0;
			mDepthCaptureStart = mSession->getDepthCaptureCount();
			if (scene.geometryVersion != mGeometryVersion) {
				mSession->replaceScene(buildScene(scene));
				mGeometryVersion = scene.geometryVersion;
			} else {
				if (scene.transformVersion != mTransformVersion) updateTransforms(scene);
				if (scene.materialVersion != mMaterialVersion) updateMaterials(scene);
			}
			mTransformVersion = scene.transformVersion;
			mMaterialVersion  = scene.materialVersion;
			mSession->resize(size);
			mSession->updateCamera(camera);
			mSession->reset(seed);
			mVersion = version;
		}
		struct CompletedImage {
			std::shared_ptr<RenderSession::Snapshot> image;
			double readbackMs{}, adaptiveMs{};
		};
		auto collect = [&](bool wait) {
			CompletedImage result;
			if (!wait) {
				uint64_t readyFrames = 0;
				for (auto &pending : mPending) {
					if (!mSession->isSnapshotReady(pending.request)) break;
					readyFrames = pending.request.snapshot.completedFrames;
				}
				if (!mPublication.publicationDue(readyFrames, samples, std::chrono::steady_clock::now()))
					return result;
			}
			while (!mPending.empty()) {
				auto &pending = mPending.front();
				bool ready = mSession->isSnapshotReady(pending.request);
				if (!ready && !wait) break;
				const auto collectStarted = std::chrono::steady_clock::now();
				result.image = std::make_shared<RenderSession::Snapshot>(
					mSession->collectSnapshot(pending.request, wait));
				double collectMs = milliseconds(collectStarted);
				mReadbackTotalMs += collectMs;
				result.readbackMs += pending.enqueueMs + collectMs;
				result.adaptiveMs += pending.enqueueMs + (ready ? collectMs : 0.0);
				mPublication.synchronized(result.image->completedFrames);
				mPending.pop_front();
			}
			return result;
		};
		auto publish = [&](CompletedImage result) {
			if (!result.image) return;
			std::lock_guard lock(state->mutex);
			if (state->version != version || !state->alive || state->paused) return;
			const auto frames = result.image->completedFrames;
			mReadbackMs = result.readbackMs;
			mPublication.published(frames, std::chrono::steady_clock::now(), result.adaptiveMs);
			++mPublicationCount;
			if (frames > 1 && frames < samples) ++mAsyncPublicationCount;
			if (frames == 1) mFirstImageMs = milliseconds(mUpdateStarted);
			state->image = std::move(result.image);
			state->imageVersion = version;
			state->converged = frames >= samples;
			if (state->converged && state->priority > 1) mSession->finish();
			if (state->converged && !state->statusPath.empty())
				std::ofstream(state->statusPath)
					<< json{{"version", version},
							{"frames", state->image->completedFrames},
							{"width", size[0]},
							{"height", size[1]},
							{"mesh_count", mNodes.size()},
							{"graphics_api", mGraphicsApi},
							{"seed", seed},
							{"geometry_version", mGeometryVersion},
							{"material_version", mMaterialVersion},
							{"transform_version", mTransformVersion},
							{"diagnostics", mDiagnostics},
							{"scene_builds", mSceneBuilds},
							{"material_updates", mMaterialUpdates},
							{"transform_updates", mTransformUpdates},
							{"lights", mLights},
							{"initialization_ms", mInitializationMs},
							{"first_image_ms", mFirstImageMs},
							{"readback_ms", mReadbackMs},
							{"readback_total_ms", mReadbackTotalMs},
							{"render_step_ms", mRenderStepMs},
							{"wait_ms", mWaitMs},
							{"update_ms", milliseconds(mUpdateStarted)},
							{"publication_count", mPublicationCount},
							{"async_publication_count", mAsyncPublicationCount},
							{"readback_submission_count", mSnapshotRequests},
							{"max_pending_readbacks", mPendingPeak},
							{"publication_interval_ms", mPublication.intervalMs()},
							{"depth_capture_count", mSession->getDepthCaptureCount() - mDepthCaptureStart},
							{"depth_requested", depthRequested},
							{"tracked_cuda_bytes", CUDATrackedMemory::singleton.BytesAllocated()}};
		};
		auto waitForFrames = [&] {
			const auto waitStarted = std::chrono::steady_clock::now();
			mSession->wait();
			mWaitMs += milliseconds(waitStarted);
			mPublication.synchronized(mSession->getCompletedFrames());
		};
		publish(collect(false));
		{
			std::lock_guard lock(state->mutex);
			if (state->version != version || !state->alive || state->paused || state->converged) return;
		}
		if (mSession->getCompletedFrames() < samples) {
			const auto renderStarted = std::chrono::steady_clock::now();
			mSession->step(1);
			mRenderStepMs += milliseconds(renderStarted);
		}
		const auto frames = mSession->getCompletedFrames();
		if (mPublication.due(frames, samples, std::chrono::steady_clock::now())) {
			const bool force = frames == 1 || frames >= samples;
			auto enqueue = [&] {
				const auto enqueueStarted = std::chrono::steady_clock::now();
				auto request = mSession->requestSnapshot(depthRequested);
				double enqueueMs = milliseconds(enqueueStarted);
				mReadbackTotalMs += enqueueMs;
				if (!request) return false;
				mPending.push_back({std::move(*request), enqueueMs});
				mPendingPeak = std::max(mPendingPeak, uint64_t(mPending.size()));
				++mSnapshotRequests;
				mPublication.submitted(frames, std::chrono::steady_clock::now());
				return true;
			};
			bool submitted = enqueue();
			if (!submitted && force) {
				collect(true);
				waitForFrames();
				submitted = enqueue();
				if (!submitted) throw std::runtime_error("No readback slot available after completing pending images");
			}
			if (submitted && force) publish(collect(true));
		}
		publish(collect(false));
		if (mPublication.needsWait(frames)) {
			waitForFrames();
			publish(collect(false));
		}
	}
	void run() {
		for (;;) {
			{
				std::lock_guard lock(mMutex);
				if (mQuit) break;
			}
			if (mCurrent) {
				bool active;
				{
					std::lock_guard lock(mCurrent->mutex);
					active = mCurrent->alive && !mCurrent->paused;
				}
				if (!active) release();
			}
			auto state = next();
			if (!state) {
				std::unique_lock lock(mMutex);
				mWake.wait_for(lock, std::chrono::milliseconds(20));
				continue;
			}
			try {
				render(state);
			} catch (const std::exception &error) {
				Log(Error, "Hydra rendering: %s", error.what());
				{
					std::lock_guard lock(state->mutex);
					state->error	 = error.what();
					state->converged = true;
					if (!state->statusPath.empty())
						std::ofstream(state->statusPath) << json{{"error", state->error}};
				}
				release();
			}
		}
		release();
		gpContext.reset();
	}
	std::mutex mMutex;
	std::condition_variable mWake;
	std::vector<std::weak_ptr<RenderState>> mStates;
	bool mQuit{};
	std::shared_ptr<RenderState> mCurrent;
	std::unique_ptr<RenderSession> mSession;
	std::map<std::string, std::vector<SceneGraphNode::SharedPtr>> mNodes;
	std::map<std::string, Material::SharedPtr> mMaterials;
	std::vector<std::string> mDiagnostics;
	std::string mGraphicsApi, mAssetRoot;
	uint64_t mVersion{}, mGeometryVersion{}, mTransformVersion{}, mMaterialVersion{};
	uint64_t mSceneBuilds{}, mMaterialUpdates{}, mTransformUpdates{};
	uint64_t mPublicationCount{}, mDepthCaptureStart{};
	uint64_t mAsyncPublicationCount{}, mSnapshotRequests{}, mPendingPeak{};
	struct PendingSnapshot {
		RenderSession::SnapshotRequest request;
		double enqueueMs{};
	};
	std::deque<PendingSnapshot> mPending;
	PublicationSchedule mPublication;
	json mLights;
	std::chrono::steady_clock::time_point mUpdateStarted;
	double mInitializationMs{}, mFirstImageMs{}, mReadbackMs{};
	double mReadbackTotalMs{}, mRenderStepMs{}, mWaitMs{};
	std::thread mThread;
};

struct WorkerRegistry {
	std::mutex mutex;
	std::unique_ptr<RenderWorker> worker;
	size_t users{};
};

WorkerRegistry &registry() {
	static WorkerRegistry instance;
	return instance;
}

} // namespace

std::shared_ptr<RenderState> createState() {
	auto &entry = registry();
	RenderWorker *worker;
	{
		std::lock_guard lock(entry.mutex);
		if (!entry.worker) entry.worker = std::make_unique<RenderWorker>();
		++entry.users;
		worker = entry.worker.get();
	}
	return worker->create();
}
void wakeRenderer() {
	auto &entry = registry();
	std::lock_guard lock(entry.mutex);
	if (entry.worker) entry.worker->wake();
}
void releaseState(const std::shared_ptr<RenderState> &state) {
	{
		std::lock_guard lock(state->mutex);
		state->alive = false;
	}
	wakeRenderer();
	{
		std::unique_lock lock(state->mutex);
		state->released.wait(lock, [&] { return !state->active; });
	}
	auto &entry = registry();
	std::lock_guard lock(entry.mutex);
	if (--entry.users == 0) entry.worker.reset();
}

} // namespace krr::hydra
