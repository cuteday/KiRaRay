#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "device/context.h"

NAMESPACE_BEGIN(krr)

class CpuTimer {
public:
	using TimePoint =
		std::chrono::time_point<std::chrono::high_resolution_clock>;

	/** Returns the current time
	 */
	static TimePoint getCurrentTimePoint() {
		return std::chrono::high_resolution_clock::now();
	}

	/** Update the timer.
		\return The TimePoint of the last update() call.
			This value is meaningless on it's own. Call CCpuTimer#calcDuration()
	   to get the duration that passed between 2 TimePoints
	*/
	TimePoint update() {
		TimePoint now = getCurrentTimePoint();
		mElapsedTime  = now - mCurrentTime;
		mCurrentTime  = now;
		return mCurrentTime;
	}

	/** Get the time that passed from the last update() call to the one before
	 * that.
	 */
	double delta() const { return mElapsedTime.count(); }

	/** Calculate the duration in milliseconds between 2 time points (in micro
	 * seconds).
	 */
	static double calcDuration(TimePoint start, TimePoint end) {
		auto delta = end.time_since_epoch() - start.time_since_epoch();
		auto duration =
			std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
		return duration.count() * 1.0e-6;
	}

	static double calcElapsedTime(TimePoint start) {
		return calcDuration(start, getCurrentTimePoint());
	}

private:
	TimePoint mCurrentTime;
	std::chrono::duration<double> mElapsedTime;
};

class GpuTimer {
public:
	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	};
	
	~GpuTimer() {
		// TODO: why this cannot work?
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);
	};

	void begin() {
		/* Why recording timer events on 0 stream is okay?
			Since the default stream (0) has the implicit synchronization mechanism, 
			meaning that any CUDA operation issued into the default stream will not 
			begin executing until ALL prior issued CUDA activity to that device 
			has completed*/
		cudaEventRecord(start, 0 /* default stream (legacy) */);
	}
	
	void end() {
		cudaEventRecord(stop, 0 /* default stream (legacy) */ );
	}

	/** Get the elapsed time in milliseconds between a pair of Begin()/End() calls. \n
		If this function called not after a Begin()/End() pair, zero will be returned and a warning will be logged.
	*/
	double getElapsedTime() {
		/* This may cause an explicit cpu-gpu synchronization and leading to performance loss.
			This is only called if the Profiler is enabled (thus calling Profiler::endFrame). 
			Disable the profiler if you do not need it. */
		float time{ 0 };
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

private:
	cudaEvent_t start, stop;
};

NAMESPACE_END(krr)