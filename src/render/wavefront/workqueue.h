#pragma once
#include "common.h"
#include <atomic>
#ifdef __NVCC__
#include <thrust/sort.h>
#endif

#include "device/cuda.h"
#include "device/atomic.h"
#include "logger.h"
#include "workitem.h"

NAMESPACE_BEGIN(krr)

class PixelStateBuffer : public SOA<PixelState> {
public:
	PixelStateBuffer() = default;
	PixelStateBuffer(int n, Allocator alloc) : SOA<PixelState>(n, alloc) {}

	KRR_CALLABLE void setRadiance(int pixelId, const Spectrum& L_val){
		L[pixelId] = L_val;
	}
	KRR_CALLABLE void addRadiance(int pixelId, const Spectrum& L_val) {
		L[pixelId] = L_val + Spectrum(L[pixelId]);
	}
};

template <typename WorkItem>
class WorkQueue : public SOA<WorkItem> {
public:
	using value_type			 = WorkItem;
	using iterator				 = typename SOA<WorkItem>::iterator;
	using const_iterator		 = typename SOA<WorkItem>::const_iterator;
	using reverse_iterator		 = typename SOA<WorkItem>::reverse_iterator;
	using const_reverse_iterator = typename SOA<WorkItem>::const_reverse_iterator;

	WorkQueue() = default;
	KRR_HOST WorkQueue(int n, Allocator alloc) : SOA<WorkItem>(n, alloc) {}
	KRR_HOST WorkQueue& operator=(const WorkQueue& w) {
		SOA<WorkItem>::operator=(w);
		m_size.store(w.m_size);
		return *this;
	}

	KRR_CALLABLE iterator begin() { return iterator(this, 0); }
	KRR_CALLABLE const_iterator begin() const { return const_iterator(this, 0); }
	KRR_CALLABLE iterator end() { return iterator(this, m_size.load()); }
	KRR_CALLABLE const_iterator end() const { return const_iterator(this, m_size.load()); }

	KRR_CALLABLE int size() const { return m_size.load(); }
	KRR_CALLABLE int capacity() const { return nAlloc; }
	KRR_CALLABLE void reset() { m_size.store(0); }

	KRR_CALLABLE int push(const WorkItem& w) {
		int index = allocateEntry();
		(*this)[index] = w;
		return index;
	}

	template <typename ...Args>
	KRR_CALLABLE int emplace(Args&&... args) {
		int index = allocateEntry();
		new (&(*this)[index]) WorkItem(std::forward<Args>(args)...);
		return index;
	}

protected:
	KRR_CALLABLE int allocateEntry() {
		return m_size.fetch_add(1);
	}

	atomic<int> m_size{ 0 };
};

template <typename WorkItem, typename Key> 
class SortableWorkQueue : public WorkQueue<WorkItem> {
public:
	using KeyType = Key;
	SortableWorkQueue() = default;
	KRR_HOST SortableWorkQueue(int n, Allocator alloc) 
		: WorkQueue<WorkItem>(n, alloc) {
		m_keys = TypedBuffer<Key>(n);
	}

	KRR_HOST SortableWorkQueue &operator=(const SortableWorkQueue &w) {
		WorkQueue<WorkItem>::operator=(w);
		m_keys = w.m_keys;
		return *this;
	}

	KRR_CALLABLE TypedBuffer<Key>& keys() { return m_keys; }

	template <typename F> 
	void updateKeys(F mapping, size_t max_elements, const Key& oob_val, CUstream stream) {
		auto* queue = this;
		GPUParallelFor(max_elements, [=] KRR_DEVICE (int index) {
				if (index >= queue->size()) queue->keys()[index] = oob_val;
				else queue->keys()[index] = mapping(queue->operator[](index));
			}, stream);
	}

	template <typename Compare> 
	void sort(Compare comp, size_t max_elements, CUstream stream) {
#ifdef __NVCC__
		thrust::sort_by_key(thrust::device.on(stream), m_keys.data(),
							m_keys.data() + max_elements, this->begin(), comp);
#endif
	}

	void resize(int n, Allocator alloc) {
		WorkQueue<WorkItem>::resize(n, alloc);
		m_keys.resize(n);
	}

protected:
	TypedBuffer<Key> m_keys;
};

template <typename T> class MultiWorkQueue;

template <typename... Ts> 
class MultiWorkQueue<TypePack<Ts...>> {
public:
	template <typename T> 
	KRR_CALLABLE WorkQueue<T>* get() {
		return &gpu::get<WorkQueue<T>>(m_queues);
	}

	MultiWorkQueue(int n, Allocator alloc, gpu::span<const bool> haveType) {
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

	template <typename T, typename... Args>
	KRR_CALLABLE int emplace(Args&&... args) {
		return get<T>()->emplace(std::forward<Args>(args)...);
	}

	KRR_CALLABLE void reset() { 
		(get<Ts>()->reset(), ...);
	}

private:
	gpu::tuple<WorkQueue<Ts>...> m_queues;
};

// Helper functions and basic classes

template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem> *q, int nElements, F &&func, CUstream stream = 0);

#ifdef __NVCC__
template <typename F, typename WorkItem>
void ForAllQueued(const WorkQueue<WorkItem> *q, int nElements, F &&func, CUstream stream) {
	GPUParallelFor(
		nElements,
		[=] KRR_DEVICE(int index) mutable {
			if (index >= q->size()) return;
			func((*q)[index]);
		},
		stream);
}
#endif

class RayQueue : public WorkQueue<RayWorkItem> {
public:
	using WorkQueue::WorkQueue;     // use parent constructor
	using WorkQueue::push;

	KRR_CALLABLE int pushCameraRay(Ray ray, uint pixelId) {
		int index = allocateEntry();
		this->depth[index]	 = 0;
		this->thp[index]	 = Spectrum::Ones();
		this->pu[index]		 = Spectrum::Ones();
		this->pl[index]		 = Spectrum::Ones();
		this->pixelId[index] = pixelId;
		this->ray[index]	 = ray;
		return index;
	}

	KRR_CALLABLE int push(const Ray& ray, const LightSampleContext& ctx, const Spectrum& thp, 
						  const Spectrum &pu, const Spectrum &pl, uint depth,
						  uint pixelId, BSDFType bsdfType) {
		int index = allocateEntry();
		this->ray[index]	  = ray;
		this->depth[index]	  = depth;
		this->thp[index]	  = thp;
		this->ctx[index]	  = ctx;
		this->pu[index]		  = pu;
		this->pl[index]		  = pl;
		this->pixelId[index]  = pixelId;
		this->bsdfType[index] = bsdfType;
		return index;
	}
};

class MissRayQueue : public WorkQueue<MissRayWorkItem> {
public:
	using WorkQueue::WorkQueue;
	using WorkQueue::push;

	KRR_CALLABLE int push(const RayWorkItem &w) {
		int index = allocateEntry();
		this->ray[index]	  = w.ray;
		this->ctx[index]	  = w.ctx;
		this->thp[index]	  = w.thp;
		this->pu[index]		  = w.pu;
		this->pl[index]		  = w.pl;
		this->bsdfType[index] = w.bsdfType;
		this->depth[index]	  = w.depth;
		this->pixelId[index]  = w.pixelId;
		return index;
	}

	KRR_CALLABLE int push(const Ray &ray, const LightSampleContext &ctx, const Spectrum &thp,
						  const Spectrum &pu, const Spectrum &pl, BSDFType bsdfType,
						  uint depth, uint pixelId) {
		int index			  = allocateEntry();
		this->ray[index]	  = ray;
		this->ctx[index]	  = ctx;
		this->thp[index]	  = thp;
		this->pu[index]		  = pu;
		this->pl[index]		  = pl;
		this->bsdfType[index] = bsdfType;
		this->depth[index]	  = depth;
		this->pixelId[index]  = pixelId;
		return index;
	}
};

class HitLightRayQueue : public WorkQueue<HitLightWorkItem> {
public:
	using WorkQueue::WorkQueue;
	using WorkQueue::push;

	KRR_CALLABLE int push(const RayWorkItem &r, const SurfaceInteraction& intr) {
		int index			  = allocateEntry();
		this->depth[index]	  = r.depth;
		this->thp[index]	  = r.thp;
		this->ctx[index]	  = r.ctx;
		this->pixelId[index]  = r.pixelId;
		this->pu[index]		  = r.pu;
		this->pl[index]		  = r.pl;
		this->bsdfType[index] = r.bsdfType;
		this->light[index]	  = intr.light;
		this->p[index]		  = intr.p;
		this->wo[index]		  = intr.wo;
		this->n[index]		  = intr.n;
		this->uv[index]		  = intr.uv;
		return index;
	}
	
	KRR_CALLABLE int push(const SurfaceInteraction& intr, const LightSampleContext& prevCtx,
						  BSDFType bsdfType, uint depth, uint pixelId, const Spectrum &thp,
						  const Spectrum &pu, const Spectrum &pl) {
		int index			  = allocateEntry();
		this->depth[index]	  = depth;
		this->thp[index]	  = thp;
		this->ctx[index]	  = prevCtx;
		this->pixelId[index]  = pixelId;
		this->pu[index]		  = pu;
		this->pl[index]		  = pl;
		this->bsdfType[index] = bsdfType;
		this->light[index]	  = intr.light;
		this->p[index]		  = intr.p;
		this->wo[index]		  = intr.wo;
		this->n[index]		  = intr.n;
		this->uv[index]		  = intr.uv;
		return index;
	}
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

	KRR_CALLABLE int push(const SurfaceInteraction &intr, const Spectrum &thp, const Spectrum &pu,
						  uint depth, uint pixelId) {
		int index			 = allocateEntry();
		this->intr[index]	 = intr;
		this->thp[index]	 = thp;
		this->pu[index]		 = pu;
		this->depth[index]	 = depth;
		this->pixelId[index] = pixelId;
		return index;
	}
};

class MediumSampleQueue : public WorkQueue<MediumSampleWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(const RayWorkItem &r, float tMax = M_FLOAT_INF) {
		int index			 = allocateEntry();
		this->tMax[index]	 = tMax;
		this->ray[index]	 = r.ray;
		this->ctx[index]	 = r.ctx;
		this->thp[index]	 = r.thp;
		this->pu[index]		 = r.pu;
		this->pl[index]		 = r.pl;
		this->depth[index]	 = r.depth;
		this->pixelId[index] = r.pixelId;
		return index;
	}

	KRR_CALLABLE int push(const RayWorkItem &r, const SurfaceInteraction &intr,
						  float tMax = M_FLOAT_INF) {
		int index			 = allocateEntry();
		this->intr[index]	 = intr;
		this->tMax[index]	 = tMax;
		this->ray[index]	 = r.ray;
		this->thp[index]	 = r.thp;
		this->pu[index]		 = r.pu;
		this->pl[index]		 = r.pl;
		this->depth[index]	 = r.depth;
		this->pixelId[index] = r.pixelId;
		return index;
	}
};

class MediumScatterQueue : public WorkQueue<MediumScatterWorkItem> {
public:
	using WorkQueue::push;
	using WorkQueue::WorkQueue;

	KRR_CALLABLE int push(Vector3f p, const Spectrum &thp, const Spectrum &pu,
						  Vector3f wo, float time, Medium medium, PhaseFunction phase, uint depth,
						  uint pixelId) {
		int index = allocateEntry();
		this->p[index]		 = p;
		this->wo[index]		 = wo;
		this->time[index]	 = time;
		this->medium[index]	 = medium;
		this->phase[index]	 = phase;
		this->thp[index]	 = thp;
		this->pu[index]		 = pu;
		this->depth[index]	 = depth;
		this->pixelId[index] = pixelId;
		return index;
	}
};

NAMESPACE_END(krr)