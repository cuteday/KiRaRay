#pragma once
#include "common.h"
#include <atomic>

#ifdef __CUDACC__
#include <cuda/atomic>
#endif

#include "math/math.h"
#include "device/cuda.h"
#include "logger.h"
#include "workitem.h"

KRR_NAMESPACE_BEGIN

template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
public:
    WorkQueue() = default;
    WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
    WorkQueue& operator=(const WorkQueue& w) {
        SOA<WorkItem>::operator=(w);
        m_size.store(w.m_size.load());
        return *this;
    }

    KRR_CALLABLE
        int size() const {
#ifdef KRR_DEVICE_CODE
        return m_size.load(cuda::std::memory_order_relaxed);
#else
        return m_size.load(std::memory_order_relaxed);
#endif
    }
    KRR_CALLABLE
        void reset() {
#ifdef KRR_DEVICE_CODE
        m_size.store(0, cuda::std::memory_order_relaxed);
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
        return m_size.fetch_add(1, cuda::std::memory_order_relaxed);
#else
        return m_size.fetch_add(1, std::memory_order_relaxed);
#endif
    }

private:
#ifdef KRR_DEVICE_CODE
    cuda::atomic<int, cuda::thread_scope_device> m_size{ 0 };
#else
    std::atomic<int> m_size{ 0 };
#endif 
};

template <typename F, typename WorkItem>
void ForAllQueued(const char* desc, const WorkQueue<WorkItem>* q, int maxQueued,
    F&& func) {
        // Launch GPU threads to process _q_ using _func_
#ifdef KRR_DEVICE_CODE
    //GPUParallelFor(desc, maxQueued, [=] PBRT_GPU(int index) mutable {
    //    if (index >= q->Size())
    //        return;
    //    func((*q)[index]);
    //});
#endif
}

class RayQueue : public WorkQueue<RayWorkItem> {
public:
    using WorkQueue::WorkQueue;     // use parent constructor

    KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
        int index = allocateEntry();
        this->depth[index] = 0;
        this->pixelId[index] = pixelId;
        this->ray[index] = ray;
        return index;
    }

    KRR_CALLABLE int pushIndirectRay() {
        int index = allocateEntry();
        return index;
    }
};

KRR_NAMESPACE_END