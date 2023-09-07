#pragma once
#include "common.h"
#include <atomic>


#include "device/cuda.h"
#include "device/atomic.h"
#include "logger.h"
#include "workitem.h"

KRR_NAMESPACE_BEGIN

class PixelStateBuffer : public SOA<PixelState> {
public:
    PixelStateBuffer() = default;
    PixelStateBuffer(int n, Allocator alloc) : SOA<PixelState>(n, alloc) {}

    KRR_CALLABLE void setRadiance(int pixelId, Color L_val){
        L[pixelId] = L_val;
    }
	KRR_CALLABLE void addRadiance(int pixelId, Color L_val) {
		L_val	   = L_val + Color(L[pixelId]);
        L[pixelId] = L_val;
    }
};

template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
public:
    WorkQueue() = default;
	KRR_HOST WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
    KRR_HOST WorkQueue& operator=(const WorkQueue& w) {
        SOA<WorkItem>::operator=(w);
		m_size.store(w.m_size);
        return *this;
    }

    KRR_CALLABLE int size() const {
		return m_size.load();
    }
    KRR_CALLABLE void reset() {
		m_size.store(0);
    }

    KRR_CALLABLE int push(WorkItem w) {
        int index = allocateEntry();
        (*this)[index] = w;
        return index;
    }

protected:
    KRR_CALLABLE int allocateEntry() {
		return m_size.fetch_add(1);
    }

private:
	atomic<int> m_size{ 0 };
};

template <typename T> 
class MultiWorkQueue;

template <typename... Ts> 
class MultiWorkQueue<types::TypePack<Ts...>> {
public:
    template <typename T> 
    KRR_CALLABLE WorkQueue<T>* get() {
        return &inter::get<WorkQueue<T>>(m_queues);
    }

    MultiWorkQueue(int n, Allocator alloc, inter::span<const bool> haveType) {
        int index = 0;
		((*get<Ts>() = WorkQueue<Ts>(haveType[index++] ? n : 1, alloc)), ...);
    }

    template <typename T>
	KRR_CALLABLE int size() const {
		return get<T>()->size();
	}

	template <typename T>
    KRR_CALLABLE int push(const T& value) { 
        return get<T>()->push(value);
    }

    KRR_CALLABLE void reset() { 
        (get<Ts>()->reset(), ...);
    }

private:
    inter::tuple<WorkQueue<Ts>...> m_queues;
};

// Helper functions and basic classes
template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem>* q, int nElements,
    F&& func, CUstream stream = 0) {
    GPUParallelFor(nElements, [=] KRR_DEVICE(int index) mutable {
        if (index >= q->size())
            return;
        func((*q)[index]);
    }, stream);
}

class RayQueue : public WorkQueue<RayWorkItem> {
public:
    using WorkQueue::WorkQueue;     // use parent constructor
    using WorkQueue::push;

    KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
        int index = allocateEntry();
        this->depth[index] = 0;
		this->thp[index]	 = Color::Ones();
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
        return push(MissRayWorkItem{ w.ray, w.ctx, w.pdf, w.thp, w.bsdfType, w.depth, w.pixelId });
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

class MediumSampleQueue : public WorkQueue<MediumSampleWorkItem> {
public:
    using WorkQueue::WorkQueue;
	using WorkQueue::push;
};

class MediumScatterQueue : public WorkQueue<MediumScatterWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;
};


KRR_NAMESPACE_END