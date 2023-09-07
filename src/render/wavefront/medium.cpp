#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "integrator.h"
#include "wavefront.h"
#include "render/profiler/profiler.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

void WavefrontPathTracer::sampleMediumInteraction(int depth) {
	PROFILE("Sample medium interaction");
	ForAllQueued(mediumSampleQueue, maxQueueSize,
				 KRR_DEVICE_LAMBDA(MediumSampleWorkItem){

	});
}

void WavefrontPathTracer::sampleMediumScattering(int depth) {
	PROFILE("Sample medium scattering");
	ForAllQueued(mediumScatterQueue, maxQueueSize,
				 KRR_DEVICE_LAMBDA(MediumScatterWorkItem){

	});
}

KRR_NAMESPACE_END