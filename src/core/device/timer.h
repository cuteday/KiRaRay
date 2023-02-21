#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"

KRR_NAMESPACE_BEGIN

class GpuTimer {
public:
	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	};
	
	~GpuTimer() {
		// TODO: this cannot work
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);
	};

	void begin(CUstream _stream = 0) {
		stream = _stream;
		cudaEventRecord(start, stream);
	}
	
	void end() {
		cudaEventRecord(stop, stream);
	}

	/** Get the elapsed time in milliseconds between a pair of Begin()/End() calls. \n
		If this function called not after a Begin()/End() pair, zero will be returned and a warning will be logged.
	*/
	double getElapsedTime() {
		float time{ 0 };
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

private:
	CUstream stream{ 0 };
	cudaEvent_t start, stop;
};

class CUDATimer {
public:
	CUDATimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	};

	~CUDATimer() {
		// TODO: this cannot work
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);
	};

	void begin(CUstream _stream = 0) {
		stream = _stream;
		cudaEventRecord(start, stream);
	}

	void end() {
		cudaEventRecord(stop, stream);
	}

	/** Get the elapsed time in milliseconds between a pair of Begin()/End() calls. \n
		If this function called not after a Begin()/End() pair, zero will be returned and a warning will be logged.
	*/
	double getElapsedTime() {
		float time{ 0 };
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		return time;
	}

private:
	CUstream stream{ 0 };
	cudaEvent_t start, stop;
};

KRR_NAMESPACE_END