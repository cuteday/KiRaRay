#pragma once
#include <chrono>
#include "common.h"

KRR_NAMESPACE_BEGIN

class CpuTimer {
public:
	using TimePoint = std::chrono::time_point < std::chrono::high_resolution_clock >;

	/** Returns the current time
	*/
	static TimePoint getCurrentTimePoint() {
		return std::chrono::high_resolution_clock::now();
	}

	/** Update the timer.
		\return The TimePoint of the last update() call.
			This value is meaningless on it's own. Call CCpuTimer#calcDuration() to get the duration that passed between 2 TimePoints
	*/
	TimePoint update() {
		TimePoint now = getCurrentTimePoint();
		mElapsedTime = now - mCurrentTime;
		mCurrentTime = now;
		return mCurrentTime;
	}

	/** Get the time that passed from the last update() call to the one before that.
	*/
	double delta() const {
		return mElapsedTime.count();
	}

	/** Calculate the duration in milliseconds between 2 time points (in micro seconds).
	*/
	static double calcDuration(TimePoint start, TimePoint end) {
		auto delta = end.time_since_epoch() - start.time_since_epoch();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
		return duration.count() * 1.0e-6;
	}

private:
	TimePoint mCurrentTime;
	std::chrono::duration<double> mElapsedTime;
};

KRR_NAMESPACE_END