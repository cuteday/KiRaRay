#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "window.h"
#include "renderpass.h"

#include "host/timer.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class AccumulatePass : public RenderPass {
public:
	using SharedPtr = std::shared_ptr<AccumulatePass>;
	KRR_REGISTER_PASS_DEC(AccumulatePass);
	enum class Mode { Accumulate, MovingAverage, Count };

	AccumulatePass() = default;
	void renderUI() override;
	void reset();
	void resize(const Vector2i &size) override;
	string getName() const override { return "AccumulatePass"; }
	void render(CUDABuffer& frame);
	CUDABuffer& result() { return *mAccumBuffer; }

private:
	friend void to_json(json &j, const AccumulatePass &p) {
		j = json{ 
			{ "spp", p.mMaxAccumCount }, { "mode", p.mMode }
		};
	}

	friend void from_json(const json &j, AccumulatePass &p) {
		p.mMaxAccumCount = j.value("spp", 0);
		p.mMode			 = j.value("mode", Mode::Accumulate);
	}

	uint mAccumCount{ 0 };
	Mode mMode{ Mode::Accumulate };
	uint mMaxAccumCount{ 0U };
	CUDABuffer *mAccumBuffer;
	CpuTimer::TimePoint mStartTime, mCurrentTime;
};

KRR_ENUM_DEFINE(AccumulatePass::Mode, {
	{AccumulatePass::Mode::Accumulate, "accumulate"},
	{AccumulatePass::Mode::MovingAverage, "moving average"}	
})

KRR_NAMESPACE_END