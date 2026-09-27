#pragma once
#include "common.h"
#include "device.h"
#include "renderpass.h"
#include "device/optix.h"

NAMESPACE_BEGIN(krr)

class GBufferPass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<GBufferPass>;
	KRR_REGISTER_PASS_DEC(GBufferPass);
	GBufferPass() = default;
	~GBufferPass() override;
	struct Depth {
		std::vector<float> linear;
		std::vector<float> projected;
	};

	void initialize() override;
	void setScene(Scene::SharedPtr scene) override;
	void render(RenderContext *context) override;
	Depth capture(const Vector2i &size);
	void finalize() override;
	std::string getName() const override { return "GBufferPass"; }

private:
	OptixBackend::SharedPtr mOptixBackend;
	LaunchParameters<GBufferPass> mLaunchParams;

	void launch(const Vector2i &size);
	CUDABuffer mLinearDepth;
	CUDABuffer mProjectedDepth;
};

NAMESPACE_END(krr)
