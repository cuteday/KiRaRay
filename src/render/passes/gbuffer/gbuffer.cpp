#include "gbuffer.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)

extern "C" char GBUFFER_PTX[];

GBufferPass::~GBufferPass() {
	try {
		finalize();
	} catch (...) {
	}
}

void GBufferPass::initialize() {
	if (mOptixBackend || !mScene) return;
	mOptixBackend = std::make_shared<OptixBackend>();
	mOptixBackend->setScene(mScene);
	mOptixBackend->initialize(OptixInitializeParameters()
								  .setPTX(GBUFFER_PTX)
								  .addRayType("Primary", true, true, false)
								  .addRaygenEntry("Primary")
								  .setMaxTraversableDepth(mScene->getMaxGraphDepth()));
}

void GBufferPass::setScene(Scene::SharedPtr scene) {
	if (mScene != scene) mOptixBackend.reset();
	mScene = std::move(scene);
	initialize();
}

void GBufferPass::launch(const Vector2i &size) {
	if (!mScene || size[0] <= 0 || size[1] <= 0)
		throw std::invalid_argument("Depth capture requires a scene and positive dimensions");
	initialize();
	const size_t bytes = size_t(size[0]) * size[1] * sizeof(float);
	mLinearDepth.resize(bytes);
	mProjectedDepth.resize(bytes);
	auto camera					 = mScene->getCamera();
	mLaunchParams.frameSize		 = size;
	mLaunchParams.cameraData	 = camera->getCameraData();
	mLaunchParams.view			 = camera->getViewMatrix();
	mLaunchParams.viewProjection = camera->getViewProjectionMatrix();
	mLaunchParams.nearClip		 = mLaunchParams.cameraData.externalProjection
									   ? mLaunchParams.cameraData.nearClip
									   : (KRR_CLIPSPACE_Z_FROM_ZERO ? 0.f : -1.f);
	mLaunchParams.linearDepth	 = reinterpret_cast<float *>(mLinearDepth.data());
	mLaunchParams.projectedDepth = reinterpret_cast<float *>(mProjectedDepth.data());
	mLaunchParams.traversable	 = mOptixBackend->getRootTraversable();
	mOptixBackend->launch(mLaunchParams, "Primary", size[0], size[1], 1, KRR_DEFAULT_STREAM);
}

void GBufferPass::render(RenderContext *) {
	PROFILE("Primary depth");
	launch(getFrameSize());
}

GBufferPass::Depth GBufferPass::capture(const Vector2i &size) {
	launch(size);
	CUDA_CHECK(cudaStreamSynchronize(KRR_DEFAULT_STREAM));
	Depth result;
	result.linear.resize(size_t(size[0]) * size[1]);
	result.projected.resize(result.linear.size());
	mLinearDepth.copy_to_host(result.linear.data(), result.linear.size());
	mProjectedDepth.copy_to_host(result.projected.data(), result.projected.size());
	return result;
}

void GBufferPass::finalize() {
	mOptixBackend.reset();
	mLinearDepth.free();
	mProjectedDepth.free();
}

NAMESPACE_END(krr)
