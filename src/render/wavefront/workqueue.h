#pragma once
#include "common.h"
#include <atomic>

#include "math/math.h"
#include "device/cuda.h"
#include "logger.h"
#include "workitem.h"

#ifdef KRR_DEVICE_CODE
#if (__CUDA_ARCH__ < 700)
#define KRR_LEGACY_CUDA_ATOMICS
#else 
#include <cuda/atomic>
#endif
#endif

KRR_NAMESPACE_BEGIN

class PixelStateBuffer : public SOA<PixelState> {
public:
    PixelStateBuffer() = default;
    PixelStateBuffer(int n, Allocator alloc) : SOA<PixelState>(n, alloc) {}

    KRR_CALLABLE void setRadiance(int pixelId, color L_val){
        L[pixelId] = L_val;
    }
    KRR_CALLABLE void addRadiance(int pixelId, color L_val) {
        L_val = L_val + color(L[pixelId]);
        L[pixelId] = L_val;
    }
};

template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
public:
    WorkQueue() = default;
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
    WorkQueue& operator=(const WorkQueue& w) {
        SOA<WorkItem>::operator=(w);
#if defined(KRR_DEVICE_CODE) && defined(KRR_LEGACY_CUDA_ATOMICS)
		m_size = w.m_size;
#else
        m_size.store(w.m_size.load());
#endif
        return *this;
    }

    KRR_CALLABLE
        int size() const {
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
        return m_size;
#else
        return m_size.load(cuda::std::memory_order_relaxed);
#endif
#else
        return m_size.load(std::memory_order_relaxed);
#endif
    }
    KRR_CALLABLE
        void reset() {
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
        m_size = 0;
#else
        m_size.store(0, cuda::std::memory_order_relaxed);
#endif
#else
        m_size.store(0, std::memory_order_relaxed);
#endif
    }

    KRR_CALLABLE
        int push(WorkItem w) {
        int index = allocateEntry();
        (*this)[index] = w;
        return index;
    }

protected:
    KRR_CALLABLE
        int allocateEntry() {
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
        return atomicAdd(&m_size, 1);
#else
        return m_size.fetch_add(1, cuda::std::memory_order_relaxed);
#endif
#else
        return m_size.fetch_add(1, std::memory_order_relaxed);
#endif
    }

private:
#ifdef KRR_DEVICE_CODE
#ifdef KRR_LEGACY_CUDA_ATOMICS
    int m_size{ 0 };
#else
    cuda::atomic<int, cuda::thread_scope_device> m_size{ 0 };
#endif
#else
    std::atomic<int> m_size{ 0 };
#endif 
};

template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem>* q, int nElements,
    F&& func) {
    GPUParallelFor(nElements, [=] KRR_DEVICE(int index) mutable {
        if (index >= q->size())
            return;
        func((*q)[index]);
    });
}

class RayQueue : public WorkQueue<RayWorkItem> {
public:
    using WorkQueue::WorkQueue;     // use parent constructor
    using WorkQueue::push;

    KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
        int index = allocateEntry();
        this->depth[index] = 0;
		this->thp[index]	 = vec3f::Ones();
        this->pixelId[index] = pixelId;
        this->ray[index] = ray;
        return index;
    }
};

class MissRayQueue : public WorkQueue<MissRayWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;

    KRR_CALLABLE int push(RayWorkItem w) {
        return push(MissRayWorkItem{ w.ray, w.ctx, w.pdf, w.thp, w.depth, w.pixelId });
    }
};

class HitLightRayQueue : public WorkQueue<HitLightWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;
};

class ShadowRayQueue : public WorkQueue<ShadowRayWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;
};

class ScatterRayQueue : public WorkQueue<ScatterRayWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;
};

KRR_NAMESPACE_END