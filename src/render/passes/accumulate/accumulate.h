#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"

#include "device/timer.h"
#include "device/buffer.h"

#include "util/task.h"

NAMESPACE_BEGIN(krr)

class AccumulatePass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<AccumulatePass>;
	KRR_REGISTER_PASS_DEC(AccumulatePass);
	enum class Mode { Accumulate, MovingAverage, Count };
	enum class Precision { Float, Double, Count };

	AccumulatePass() = default;
	void finalize() override;

	void renderUI() override;
	void resize(const Vector2i &size) override;
	string getName() const override { return "AccumulatePass"; }
	void render(RenderContext *context) override;
	void endFrame(RenderContext *context) override;

private:
	void reset();
	size_t getPixelSize() const;

	friend void to_json(json &j, const AccumulatePass &p) {
		j = json{ 
			{ "spp", p.mMaxAccumCount }, 
			{ "mode", p.mMode },
			{ "task", p.mTask }, 
			{ "precision", p.mPrecision },
			{ "save_on_finish", p.mSaveOnFinish }, 
			{ "exit_on_finish", p.mExitOnFinish}
		};
	}

	friend void from_json(const json &j, AccumulatePass &p) {
		p.mMaxAccumCount = j.value("spp", 0);
		p.mMode			 = j.value("mode", Mode::Accumulate);
		p.mPrecision	 = j.value("precision", Precision::Float);
		p.mSaveOnFinish  = j.value("save_on_finish", false);
		p.mExitOnFinish  = j.value("exit_on_finish", false);
		if (j.contains("task"))  j.at("task").get_to(p.mTask);
		
	}

	uint mAccumCount{ 0 };
	Mode mMode{ Mode::Accumulate };
	Precision mPrecision { Precision::Float };
	uint mMaxAccumCount{ 0U };
	CUDABuffer *mAccumBuffer;
	RenderTask mTask;
	bool mSaveOnFinish{}, mExitOnFinish{};
};

KRR_ENUM_DEFINE(AccumulatePass::Mode, {
	{AccumulatePass::Mode::Accumulate, "accumulate"},
	{AccumulatePass::Mode::MovingAverage, "moving average"}	
})

KRR_ENUM_DEFINE(AccumulatePass::Precision, {
	{AccumulatePass::Precision::Float, "float"},
	{AccumulatePass::Precision::Double, "double"}
})

NAMESPACE_END(krr)