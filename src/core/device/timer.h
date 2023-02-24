#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

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

KRR_NAMESPACE_END