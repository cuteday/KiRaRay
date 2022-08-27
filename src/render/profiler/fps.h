#pragma once
#include <vector>

#include "common.h"
#include "logger.h"
#include "window.h"
#include "math/math.h"
#include "host/timer.h"
#include "util/ema.h"

KRR_NAMESPACE_BEGIN

class FrameRate {
public:
	FrameRate() {
		mFrameTimes.resize(sFrameWindow);
		reset();
	}

	void reset() {
		mFrameCount = 0;
		mLastTick	= CpuTimer::getCurrentTimePoint();
	}

	void newFrame() {
		double lastFrame = CpuTimer::calcDuration(mLastTick, CpuTimer::getCurrentTimePoint()); 
		mFrameTimes[mFrameCount++ % sFrameWindow] = lastFrame;
		mLastTick								  = CpuTimer::getCurrentTimePoint();
	}

	double getAverageFrameTime() const {
		uint64_t frames	   = min(mFrameCount, sFrameWindow);
		double elapsedTime = 0;
		for (uint64_t i = 0; i < frames; i++)
			elapsedTime += mFrameTimes[i];
		double time = elapsedTime / double(frames);
		return time;
	}

	double getLastFrameTime() const { return mFrameTimes[mFrameCount % sFrameWindow]; }

	uint64_t getFrameCount() const { return mFrameCount; }

	void plotFrameTimeGraph() const {
		ui::PlotLines("Frame time", mFrameTimes.data(), min(mFrameCount, sFrameWindow),
					  mFrameCount < sFrameWindow ? 0 : mFrameCount % sFrameWindow, 0,
					  FLT_MAX, FLT_MAX, ImVec2(0, 50));
	}

private:
	CpuTimer::TimePoint mLastTick;
	std::vector<float> mFrameTimes;
	uint64_t mFrameCount			   = 0;
	static const uint64_t sFrameWindow = 20;
};

KRR_NAMESPACE_END