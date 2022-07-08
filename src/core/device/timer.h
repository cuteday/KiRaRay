#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "common.h"

KRR_NAMESPACE_BEGIN

class GpuTimer {
public:
	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		glGenQueries(1, &glQuery);
	};
	
	~GpuTimer() {
		// TODO: this cannot work
		//cudaEventDestroy(start);
		//cudaEventDestroy(stop);
		glDeleteQueries(1, &glQuery);
	};

	void begin(CUstream _stream = 0) {
		stream = _stream;
		cudaEventRecord(start, stream);
		glBeginQuery(GL_TIME_ELAPSED, glQuery);
	}
	
	void end() {
		cudaEventRecord(stop, stream);
		glEndQuery(GL_TIME_ELAPSED);
	}

	/** Get the elapsed time in milliseconds between a pair of Begin()/End() calls. \n
		If this function called not after a Begin()/End() pair, zero will be returned and a warning will be logged.
	*/
	double getElapsedTime() {
		float time{ 0 };
		GLuint64 glTime{ 0 };
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		glGetQueryObjectui64v(glQuery, GL_QUERY_RESULT, &glTime);
		return time + glTime / 1000000;
	}

private:
	CUstream stream{ 0 };
	cudaEvent_t start, stop;
	GLuint glQuery;
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

class GLTimer {
public:
	GLTimer() {
		glGenQueries(1, &glQuery);
	};

	~GLTimer() {
		glDeleteQueries(1, &glQuery);
	};

	void begin() {
		glBeginQuery(GL_TIME_ELAPSED, glQuery);
	}

	void end() {
		glEndQuery(GL_TIME_ELAPSED);
	}

	/** Get the elapsed time in milliseconds between a pair of Begin()/End() calls. \n
		If this function called not after a Begin()/End() pair, zero will be returned and a warning will be logged.
	*/
	double getElapsedTime() {
		GLuint64 time{ 0 };
		glGetQueryObjectui64v(glQuery, GL_QUERY_RESULT, &time);
		return time / 1000000;
	}

private:
	GLuint glQuery;
};

KRR_NAMESPACE_END