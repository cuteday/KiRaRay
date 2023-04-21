#pragma once
#include "common.h"
#include <atomic>

#include "device/soa.h"
#include "device/cuda.h"
#include "device/atomic.h"
#include "logger.h"
#include "workitem.h"

KRR_NAMESPACE_BEGIN

class PixelState : public SoA<Color, PCGSampler> {
public:
	using SoA<Color, PCGSampler>::SoA;
    enum { eColor, eSampler };

	KRR_CALLABLE void setRadiance(int pixelId, Color L) {
		SoA::get<eColor>(pixelId) = L;
	}

	KRR_CALLABLE void addRadiance(int pixelId, Color L) {
		SoA::get<eColor>(pixelId) += L;
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

template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem>* q, int nElements,
    F&& func, CUstream stream = 0) {
    GPUParallelFor(nElements, [=] KRR_DEVICE(int index) mutable {
        if (index >= q->size())
            return;
        func((*q)[index]);
    }, stream);
}

class RayQUEUE : public SoAQueue<Ray, LightSampleContext, 
    float, Color, BSDFType, uint, uint> {
public:
	using SoAQueue::SoAQueue;	
	enum {
		eRay,
		eContext,
		ePdf,
		eThroughput,
		eBsdfType,
		eDepth,
		ePixelId
	};
	
    KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
		int index					 = allocate();
		SoA::get<eRay>(index)		 = ray;
		SoA::get<ePixelId>(index)	 = pixelId;
		SoA::get<eDepth>(index)		 = 0;
		SoA::get<eThroughput>(index) = Color(1.f);
		return index;
    }
};

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

class MissRayQUEUE : public SoAQueue<Ray, LightSampleContext,
    float, Color, BSDFType, uint, uint> {
	using SoAQueue::SoAQueue;
	enum {
		eRay,
		eContext,
		ePdf,
		eThroughput,
		eBsdfType,
		eDepth,
		ePixelId
	};
};

class MissRayQueue : public WorkQueue<MissRayWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;

    KRR_CALLABLE int push(RayWorkItem w) {
        return push(MissRayWorkItem{ w.ray, w.ctx, w.pdf, w.thp, w.bsdfType, w.depth, w.pixelId });
    }
};

class HitLightRayQUEUE : public SoAQueue<Light, LightSampleContext,
    float, Vector3f, Vector3f, Vector3f, Vector2f, Color, BSDFType, uint, uint> {
public:
    using SoAQueue::SoAQueue;
	enum {
	    eLight, 
        eContext, 
        ePdf, 
        ePosition,
        eWo,
        eNormal,
        eUV,
        eThroughput, 
        eBsdfType,
        eDepth,
        ePixelId
    };
};

class HitLightRayQueue : public WorkQueue<HitLightWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;
};

class ShadowRayQUEUE : public SoAQueue<Ray, float, Color, Color, uint> {
public:
    using SoAQueue::SoAQueue;
	enum {
	    eRay,
        eMaxT,
        eLi,
        eA,
        ePixelId
    };
};

class ShadowRayQueue : public WorkQueue<ShadowRayWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;
};

class ScatterRayQUEUE : public SoAQueue<Color, ShadingData, uint, uint> {
public:
    using SoAQueue::SoAQueue;
	enum {
	    eThroughput,
        eShadingData,
        eDepth,
        ePixelId
    };
};

class ScatterRayQueue : public WorkQueue<ScatterRayWorkItem> {
public:
    using WorkQueue::WorkQueue;
    using WorkQueue::push;
};

KRR_NAMESPACE_END