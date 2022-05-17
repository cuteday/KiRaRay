#pragma once
#include <atomic>
#include <array>
#include <stack>
#ifdef KRR_DEVICE_CODE
#include <cuda/atomic>
#endif

#include "common.h"
#include "sampler.h"
#include "logger.h"
#include "host/synchronize.h"
#include "math/math.h"
#include "util/check.h"

KRR_NAMESPACE_BEGIN

using namespace math;

static void addToAtomicfloat(std::atomic<float>& var, float val) {
	auto current = var.load();
	while (!var.compare_exchange_weak(current, current + val));
} 

inline float logistic(float x) {
	return 1 / (1 + std::exp(-x));
}

// Implements the stochastic-gradient-based Adam optimizer [Kingma and Ba 2014]
class AdamOptimizer {
public:
	AdamOptimizer(float learningRate, int batchSize = 1, float epsilon = 1e-08f, float beta1 = 0.9f, float beta2 = 0.999f) {
		m_hparams = { learningRate, batchSize, epsilon, beta1, beta2 };
	}

	AdamOptimizer& operator=(const AdamOptimizer& arg) {
		m_state = arg.m_state;
		m_hparams = arg.m_hparams;
		return *this;
	}

	AdamOptimizer(const AdamOptimizer& arg) {
		*this = arg;
	}

	void append(float gradient, float statisticalWeight) {
		m_state.batchGradient += gradient * statisticalWeight;
		m_state.batchAccumulation += statisticalWeight;

		if (m_state.batchAccumulation > m_hparams.batchSize) {
			step(m_state.batchGradient / m_state.batchAccumulation);

			m_state.batchGradient = 0;
			m_state.batchAccumulation = 0;
		}
	}

	void step(float gradient) {
		++m_state.iter;

		float actualLearningRate = m_hparams.learningRate * std::sqrt(1 - pow(m_hparams.beta2, m_state.iter)) / (1 - pow(m_hparams.beta1, m_state.iter));
		m_state.firstMoment = m_hparams.beta1 * m_state.firstMoment + (1 - m_hparams.beta1) * gradient;
		m_state.secondMoment = m_hparams.beta2 * m_state.secondMoment + (1 - m_hparams.beta2) * gradient * gradient;
		m_state.variable -= actualLearningRate * m_state.firstMoment / (std::sqrt(m_state.secondMoment) + m_hparams.epsilon);

		// Clamp the variable to the range [-20, 20] as a safeguard to avoid numerical instability:
		// since the sigmoid involves the exponential of the variable, value of -20 or 20 already yield
		// in *extremely* small and large results that are pretty much never necessary in practice.
		m_state.variable = clamp(m_state.variable, -20.0f, 20.0f);
	}

	float variable() const {
		return m_state.variable;
	}

private:
	struct State {
		int iter = 0;
		float firstMoment = 0;
		float secondMoment = 0;
		float variable = 0;

		float batchAccumulation = 0;
		float batchGradient = 0;
	} m_state;

	struct Hyperparameters {
		float learningRate;
		int batchSize;
		float epsilon;
		float beta1;
		float beta2;
	} m_hparams;
};

enum class ESampleCombination {
	EDiscard,
	EDiscardWithAutomaticBudget,
	EInverseVariance,
};

enum class EBsdfSamplingFractionLoss {
	ENone,
	EKL,
	EVariance,
};

enum class ESpatialFilter {
	ENearest,
	EStochasticBox,
	EBox,
};

enum class EDirectionalFilter {
	ENearest,
	EBox,
};

class QuadTreeNode {
public:
	QuadTreeNode() {
		m_children = {};
		for (size_t i = 0; i < m_sum.size(); ++i) {
			m_sum[i].store(0, std::memory_order_relaxed);
		}
	}

	KRR_CALLABLE void setSum(int index, float val) {
		m_sum[index].store(val, std::memory_order_relaxed);
	}

	KRR_CALLABLE float sum(int index) const {
		return m_sum[index].load(std::memory_order_relaxed);
	}

	void copyFrom(const QuadTreeNode& arg) {
		for (int i = 0; i < 4; ++i) {
			setSum(i, arg.sum(i));
			m_children[i] = arg.m_children[i];
		}
	}

	QuadTreeNode(const QuadTreeNode& arg) {
		copyFrom(arg);
	}

	KRR_CALLABLE QuadTreeNode& operator=(const QuadTreeNode& arg) {
		copyFrom(arg);
		return *this;
	}

	KRR_CALLABLE void setChild(int idx, uint16_t val) {
		m_children[idx] = val;
	}

	KRR_CALLABLE uint16_t child(int idx) const {
		return m_children[idx];
	}

	KRR_CALLABLE void setSum(float val) {
		for (int i = 0; i < 4; ++i) {
			setSum(i, val);
		}
	}

	KRR_CALLABLE int childIndex(vec2f& p) const {
		int res = 0;
		for (int i = 0; i < vec2f::dims; ++i) {
			if (p[i] < 0.5f) {
				p[i] *= 2;
			}
			else {
				p[i] = (p[i] - 0.5f) * 2;
				res |= 1 << i;
			}
		}

		return res;
	}

	// Evaluates the directional irradiance *sum density* (i.e. sum / area) at a given location p.
	// To obtain radiance, the sum density (result of this function) must be divided
	// by the total statistical weight of the estimates that were summed up.
	KRR_CALLABLE float eval(vec2f& p, const std::vector<QuadTreeNode>& nodes) const {
		CHECK(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
		const int index = childIndex(p);
		if (isLeaf(index)) {
			return 4 * sum(index);
		}
		else {
			return 4 * nodes[child(index)].eval(p, nodes);
		}
	}

	KRR_CALLABLE float pdf(vec2f& p, const std::vector<QuadTreeNode>& nodes) const {
		CHECK(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
		const int index = childIndex(p);
		if (!(sum(index) > 0)) {
			return 0;
		}

		const float factor = 4 * sum(index) / (sum(0) + sum(1) + sum(2) + sum(3));
		if (isLeaf(index)) {
			return factor;
		}
		else {
			return factor * nodes[child(index)].pdf(p, nodes);
		}
	}

	KRR_CALLABLE int depthAt(vec2f& p, const std::vector<QuadTreeNode>& nodes) const {
		CHECK(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
		const int index = childIndex(p);
		if (isLeaf(index)) {
			return 1;
		}
		else {
			return 1 + nodes[child(index)].depthAt(p, nodes);
		}
	}

	KRR_CALLABLE vec2f sample(Sampler sampler, const std::vector<QuadTreeNode>& nodes) const {
		int index = 0;

		float topLeft = sum(0);
		float topRight = sum(1);
		float partial = topLeft + sum(2);
		float total = partial + topRight + sum(3);

		// Should only happen when there are numerical instabilities.
		if (!(total > 0.0f)) {
			return sampler.get2D();
		}

		float boundary = partial / total;
		vec2f origin = vec2f{ 0.0f, 0.0f };

		float sample = sampler.get1D();

		if (sample < boundary) {
			CHECK(partial > 0);
			sample /= boundary;
			boundary = topLeft / partial;
		}
		else {
			partial = total - partial;
			CHECK(partial > 0);
			origin.x = 0.5f;
			sample = (sample - boundary) / (1.0f - boundary);
			boundary = topRight / partial;
			index |= 1 << 0;
		}

		if (sample < boundary) {
			sample /= boundary;
		}
		else {
			origin.y = 0.5f;
			sample = (sample - boundary) / (1.0f - boundary);
			index |= 1 << 1;
		}

		if (isLeaf(index)) {
			return origin + 0.5f * sampler.get2D();
		}
		else {
			return origin + 0.5f * nodes[child(index)].sample(sampler, nodes);
		}
	}

	KRR_CALLABLE void record(vec2f& p, float irradiance, std::vector<QuadTreeNode>& nodes) {
		CHECK(p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
		int index = childIndex(p);

		if (isLeaf(index)) {
			addToAtomicfloat(m_sum[index], irradiance);
		}
		else {
			nodes[child(index)].record(p, irradiance, nodes);
		}
	}

	KRR_CALLABLE float computeOverlappingArea(const vec2f& min1, const vec2f& max1, const vec2f& min2, const vec2f& max2) {
		float lengths[2];
		for (int i = 0; i < 2; ++i) {
			lengths[i] = max(min(max1[i], max2[i]) - max(min1[i], min2[i]), 0.0f);
		}
		return lengths[0] * lengths[1];
	}

	KRR_CALLABLE void record(const vec2f& origin, float size, vec2f nodeOrigin, float nodeSize, float value, std::vector<QuadTreeNode>& nodes) {
		float childSize = nodeSize / 2;
		for (int i = 0; i < 4; ++i) {
			vec2f childOrigin = nodeOrigin;
			if (i & 1) { childOrigin[0] += childSize; }
			if (i & 2) { childOrigin[1] += childSize; }

			float w = computeOverlappingArea(origin, origin + vec2f(size), childOrigin, childOrigin + vec2f(childSize));
			if (w > 0.0f) {
				if (isLeaf(i)) {
					addToAtomicfloat(m_sum[i], value * w);
				}
				else {
					nodes[child(i)].record(origin, size, childOrigin, childSize, value, nodes);
				}
			}
		}
	}

	KRR_CALLABLE bool isLeaf(int index) const {
		return child(index) == 0;
	}

	// Ensure that each quadtree node's sum of irradiance estimates
	// equals that of all its children.
	void build(std::vector<QuadTreeNode>& nodes) {
		for (int i = 0; i < 4; ++i) {
			// During sampling, all irradiance estimates are accumulated in
			// the leaves, so the leaves are built by definition.
			if (isLeaf(i)) {
				continue;
			}

			QuadTreeNode& c = nodes[child(i)];

			// Recursively build each child such that their sum becomes valid...
			c.build(nodes);

			// ...then sum up the children's sums.
			float sum = 0;
			for (int j = 0; j < 4; ++j) {
				sum += c.sum(j);
			}
			setSum(i, sum);
		}
	}

private:
	std::array<std::atomic<float>, 4> m_sum;
	std::array<uint16_t, 4> m_children;
};


class DTree {
public:
	DTree() {
		m_atomic.sum.store(0, std::memory_order_relaxed);
		m_maxDepth = 0;
		m_nodes.emplace_back();
		m_nodes.front().setSum(0.0f);
	}

	KRR_CALLABLE const QuadTreeNode& node(size_t i) const {
		return m_nodes[i];
	}

	KRR_CALLABLE float mean() const {
		if (m_atomic.statisticalWeight == 0) {
			return 0;
		}
		const float factor = 1 / (M_PI * 4 * m_atomic.statisticalWeight);
		return factor * m_atomic.sum;
	}

	KRR_CALLABLE void recordIrradiance(vec2f p, float irradiance, float statisticalWeight, EDirectionalFilter directionalFilter) {
		if (std::isfinite(statisticalWeight) && statisticalWeight > 0) {
			addToAtomicfloat(m_atomic.statisticalWeight, statisticalWeight);

			if (std::isfinite(irradiance) && irradiance > 0) {
				if (directionalFilter == EDirectionalFilter::ENearest) {
					m_nodes[0].record(p, irradiance * statisticalWeight, m_nodes);
				}
				else {
					int depth = depthAt(p);
					float size = pow(0.5f, depth);

					vec2f origin = p;
					origin.x -= size / 2;
					origin.y -= size / 2;
					m_nodes[0].record(origin, size, vec2f(0.0f), 1.0f, irradiance * statisticalWeight / (size * size), m_nodes);
				}
			}
		}
	}

	KRR_CALLABLE float pdf(vec2f p) const {
		if (!(mean() > 0)) {
			return 1 / (4 * M_PI);
		}

		return m_nodes[0].pdf(p, m_nodes) / (4 * M_PI);
	}

	KRR_CALLABLE int depthAt(vec2f p) const {
		return m_nodes[0].depthAt(p, m_nodes);
	}

	KRR_CALLABLE int depth() const {
		return m_maxDepth;
	}

	KRR_CALLABLE vec2f sample(Sampler sampler) const {
		if (!(mean() > 0)) {
			return sampler.get2D();
		}

		vec2f res = m_nodes[0].sample(sampler, m_nodes);

		res.x = clamp(res.x, 0.0f, 1.0f);
		res.y = clamp(res.y, 0.0f, 1.0f);

		return res;
	}

	KRR_CALLABLE size_t numNodes() const {
		return m_nodes.size();
	}

	KRR_CALLABLE float statisticalWeight() const {
		return m_atomic.statisticalWeight;
	}

	KRR_CALLABLE void setStatisticalWeight(float statisticalWeight) {
		m_atomic.statisticalWeight = statisticalWeight;
	}

	void reset(const DTree& previousDTree, int newMaxDepth, float subdivisionThreshold) {
		m_atomic = Atomic{};
		m_maxDepth = 0;
		m_nodes.clear();
		m_nodes.emplace_back();

		struct StackNode {
			size_t nodeIndex;
			size_t otherNodeIndex;
			const DTree* otherDTree;
			int depth;
		};

		std::stack<StackNode> nodeIndices;
		nodeIndices.push({ 0, 0, &previousDTree, 1 });

		const float total = previousDTree.m_atomic.sum;

		// Create the topology of the new DTree to be the refined version
		// of the previous DTree. Subdivision is recursive if enough energy is there.
		while (!nodeIndices.empty()) {
			StackNode sNode = nodeIndices.top();
			nodeIndices.pop();

			m_maxDepth = max(m_maxDepth, sNode.depth);

			for (int i = 0; i < 4; ++i) {
				const QuadTreeNode& otherNode = sNode.otherDTree->m_nodes[sNode.otherNodeIndex];
				const float fraction = total > 0 ? (otherNode.sum(i) / total) : pow(0.25f, sNode.depth);
				CHECK(fraction <= 1.0f + M_EPSILON);

				if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
					if (!otherNode.isLeaf(i)) {
						CHECK(sNode.otherDTree == &previousDTree);
						nodeIndices.push({ m_nodes.size(), otherNode.child(i), &previousDTree, sNode.depth + 1 });
					}
					else {
						nodeIndices.push({ m_nodes.size(), m_nodes.size(), this, sNode.depth + 1 });
					}

					m_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(m_nodes.size()));
					m_nodes.emplace_back();
					m_nodes.back().setSum(otherNode.sum(i) / 4);

					if (m_nodes.size() > std::numeric_limits<uint16_t>::max()) {
						logWarning("DTreeWrapper hit maximum children count.");
						nodeIndices = std::stack<StackNode>();
						break;
					}
				}
			}
		}

		// Uncomment once memory becomes an issue.
		//m_nodes.shrink_to_fit();

		for (auto& node : m_nodes) {
			node.setSum(0);
		}
	}

	KRR_CALLABLE size_t approxMemoryFootprint() const {
		return m_nodes.capacity() * sizeof(QuadTreeNode) + sizeof(*this);
	}

	void build() {
		auto& root = m_nodes[0];

		// Build the quadtree recursively, starting from its root.
		root.build(m_nodes);

		// Ensure that the overall sum of irradiance estimates equals
		// the sum of irradiance estimates found in the quadtree.
		float sum = 0;
		for (int i = 0; i < 4; ++i) {
			sum += root.sum(i);
		}
		m_atomic.sum.store(sum);
	}

private:
	std::vector<QuadTreeNode> m_nodes;

	struct Atomic {
		Atomic() {
			sum.store(0, std::memory_order_relaxed);
			statisticalWeight.store(0, std::memory_order_relaxed);
		}

		Atomic(const Atomic& arg) {
			*this = arg;
		}

		Atomic& operator=(const Atomic& arg) {
			sum.store(arg.sum.load(std::memory_order_relaxed), std::memory_order_relaxed);
			statisticalWeight.store(arg.statisticalWeight.load(std::memory_order_relaxed), std::memory_order_relaxed);
			return *this;
		}

		std::atomic<float> sum;
		std::atomic<float> statisticalWeight;

	} m_atomic;

	int m_maxDepth;
};

struct DTreeRecord {
	vec3f d;
	float radiance, product;
	float woPdf, bsdfPdf, dTreePdf;
	float statisticalWeight;
	bool isDelta;
};

struct DTreeWrapper {
public:
	DTreeWrapper() {}

	KRR_CALLABLE void record(const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
		if (!rec.isDelta) {
			float irradiance = rec.radiance / rec.woPdf;
			building.recordIrradiance(dirToCanonical(rec.d), irradiance, rec.statisticalWeight, directionalFilter);
		}

		if (bsdfSamplingFractionLoss != EBsdfSamplingFractionLoss::ENone && rec.product > 0) {
			optimizeBsdfSamplingFraction(rec, bsdfSamplingFractionLoss == EBsdfSamplingFractionLoss::EKL ? 1.0f : 2.0f);
		}
	}

	KRR_CALLABLE static vec3f canonicalToDir(vec2f p) {
		const float cosTheta = 2 * p.x - 1;
		const float phi = 2 * M_PI * p.y;

		const float sinTheta = sqrt(1 - cosTheta * cosTheta);
		float sinPhi = sin(phi), cosPhi = cos(phi);

		return { sinTheta * cosPhi, sinTheta * sinPhi, cosTheta };
	}

	KRR_CALLABLE static vec2f dirToCanonical(const vec3f& d) {
		if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.z)) {
			return { 0, 0 };
		}

		const float cosTheta = min(max(d.z, -1.0f), 1.0f);
		float phi = std::atan2(d.y, d.x);
		while (phi < 0)
			phi += 2.0 * M_PI;

		return { (cosTheta + 1) / 2, phi / (2 * M_PI) };
	}

	void build() {
		building.build();
		sampling = building;
	}

	void reset(int maxDepth, float subdivisionThreshold) {
		building.reset(sampling, maxDepth, subdivisionThreshold);
	}

	KRR_CALLABLE vec3f sample(Sampler sampler) const {
		return canonicalToDir(sampling.sample(sampler));
	}

	KRR_CALLABLE float pdf(const vec3f& dir) const {
		return sampling.pdf(dirToCanonical(dir));
	}

	KRR_CALLABLE float diff(const DTreeWrapper& other) const {
		return 0.0f;
	}

	KRR_CALLABLE int depth() const {
		return sampling.depth();
	}

	KRR_CALLABLE size_t numNodes() const {
		return sampling.numNodes();
	}

	KRR_CALLABLE float meanRadiance() const {
		return sampling.mean();
	}

	KRR_CALLABLE float statisticalWeight() const {
		return sampling.statisticalWeight();
	}

	KRR_CALLABLE float statisticalWeightBuilding() const {
		return building.statisticalWeight();
	}

	KRR_CALLABLE void setStatisticalWeightBuilding(float statisticalWeight) {
		building.setStatisticalWeight(statisticalWeight);
	}

	KRR_CALLABLE size_t approxMemoryFootprint() const {
		return building.approxMemoryFootprint() + sampling.approxMemoryFootprint();
	}

	KRR_CALLABLE float bsdfSamplingFraction(float variable) const {
		return logistic(variable);
	}

	KRR_CALLABLE float dBsdfSamplingFraction_dVariable(float variable) const {
		float fraction = bsdfSamplingFraction(variable);
		return fraction * (1 - fraction);
	}

	KRR_CALLABLE float bsdfSamplingFraction() const {
		return bsdfSamplingFraction(bsdfSamplingFractionOptimizer.variable());
	}

	void optimizeBsdfSamplingFraction(const DTreeRecord& rec, float ratioPower) {
		m_lock.lock();

		// GRADIENT COMPUTATION
		float variable = bsdfSamplingFractionOptimizer.variable();
		float samplingFraction = bsdfSamplingFraction(variable);

		// Loss gradient w.r.t. sampling fraction
		float mixPdf = samplingFraction * rec.bsdfPdf + (1 - samplingFraction) * rec.dTreePdf;
		float ratio = pow(rec.product / mixPdf, ratioPower);
		float dLoss_dSamplingFraction = -ratio / rec.woPdf * (rec.bsdfPdf - rec.dTreePdf);

		// Chain rule to get loss gradient w.r.t. trainable variable
		float dLoss_dVariable = dLoss_dSamplingFraction * dBsdfSamplingFraction_dVariable(variable);

		// We want some regularization such that our parameter does not become too big.
		// We use l2 regularization, resulting in the following linear gradient.
		float l2RegGradient = 0.01f * variable;

		float lossGradient = l2RegGradient + dLoss_dVariable;

		// ADAM GRADIENT DESCENT
		bsdfSamplingFractionOptimizer.append(lossGradient, rec.statisticalWeight);

		m_lock.unlock();
	}

private:
	DTree building;
	DTree sampling;

	AdamOptimizer bsdfSamplingFractionOptimizer{ 0.01f };

	SpinLock m_lock;
};

struct STreeNode {
	STreeNode() {
		//children = {};
		isLeaf = true;
		axis = 0;
	}

	KRR_CALLABLE int childIndex(vec3f& p) const {
		if (p[axis] < 0.5f) {
			p[axis] *= 2;
			return 0;
		}
		else {
			p[axis] = (p[axis] - 0.5f) * 2;
			return 1;
		}
	}

	KRR_CALLABLE int nodeIndex(vec3f& p) const {
		return children[childIndex(p)];
	}

	KRR_CALLABLE DTreeWrapper* dTreeWrapper(vec3f& p, vec3f& size, std::vector<STreeNode>& nodes) {
		CHECK(p[axis] >= 0 && p[axis] <= 1);
		if (isLeaf) {
			return &dTree;
		}
		else {
			size[axis] /= 2;
			return nodes[nodeIndex(p)].dTreeWrapper(p, size, nodes);
		}
	}

	KRR_CALLABLE const DTreeWrapper* dTreeWrapper() const {
		return &dTree;
	}

	KRR_CALLABLE int depth(vec3f& p, const std::vector<STreeNode>& nodes) const {
		CHECK(p[axis] >= 0 && p[axis] <= 1);
		if (isLeaf) {
			return 1;
		}
		else {
			return 1 + nodes[nodeIndex(p)].depth(p, nodes);
		}
	}

	KRR_CALLABLE int depth(const std::vector<STreeNode>& nodes) const {
		int result = 1;

		if (!isLeaf) {
			for (auto c : children) {
				result = max(result, 1 + nodes[c].depth(nodes));
			}
		}

		return result;
	}

	void forEachLeaf(
		std::function<void(const DTreeWrapper*, const vec3f&, const vec3f&)> func,
		vec3f p, vec3f size, const std::vector<STreeNode>& nodes) const {

		if (isLeaf) {
			func(&dTree, p, size);
		}
		else {
			size[axis] /= 2;
			for (int i = 0; i < 2; ++i) {
				vec3f childP = p;
				if (i == 1) {
					childP[axis] += size[axis];
				}

				nodes[children[i]].forEachLeaf(func, childP, size, nodes);
			}
		}
	}

	KRR_CALLABLE float computeOverlappingVolume(const vec3f& min1, const vec3f& max1, const vec3f& min2, const vec3f& max2) {
		float lengths[3];
		for (int i = 0; i < 3; ++i) {
			lengths[i] = max(min(max1[i], max2[i]) - max(min1[i], min2[i]), 0.0f);
		}
		return lengths[0] * lengths[1] * lengths[2];
	}

	KRR_CALLABLE void record(const vec3f& min1, const vec3f& max1, vec3f min2, vec3f size2, const DTreeRecord& rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss, std::vector<STreeNode>& nodes) {
		float w = computeOverlappingVolume(min1, max1, min2, min2 + size2);
		if (w > 0) {
			if (isLeaf) {
				dTree.record({ rec.d, rec.radiance, rec.product, rec.woPdf, rec.bsdfPdf, rec.dTreePdf, rec.statisticalWeight * w, rec.isDelta }, directionalFilter, bsdfSamplingFractionLoss);
			}
			else {
				size2[axis] /= 2;
				for (int i = 0; i < 2; ++i) {
					if (i & 1) {
						min2[axis] += size2[axis];
					}

					nodes[children[i]].record(min1, max1, min2, size2, rec, directionalFilter, bsdfSamplingFractionLoss, nodes);
				}
			}
		}
	}

	bool isLeaf;
	DTreeWrapper dTree;
	int axis;
	uint32_t children[2];
	//std::array<uint32_t, 2> children;
};


class STree {
public:
	STree() = default;

	STree(const AABB& aabb) {
		clear();

		m_aabb = aabb;

		// Enlarge AABB to turn it into a cube. This has the effect
		// of nicer hierarchical subdivisions.
		vec3f size = m_aabb.upper - m_aabb.lower;
		float maxSize = max(max(size.x, size.y), size.z);
		m_aabb.upper = m_aabb.lower + vec3f(maxSize);
	}

	void clear() {
		m_nodes.clear();
		m_nodes.emplace_back();
	}

	void subdivideAll() {
		int nNodes = (int)m_nodes.size();
		for (int i = 0; i < nNodes; ++i) {
			if (m_nodes[i].isLeaf) {
				subdivide(i, m_nodes);
			}
		}
	}

	void subdivide(int nodeIdx, std::vector<STreeNode>& nodes) {
		// Add 2 child nodes
		nodes.resize(nodes.size() + 2);

		if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
			logWarning("DTreeWrapper hit maximum children count.");
			return;
		}

		STreeNode& cur = nodes[nodeIdx];
		for (int i = 0; i < 2; ++i) {
			uint32_t idx = (uint32_t)nodes.size() - 2 + i;
			cur.children[i] = idx;
			nodes[idx].axis = (cur.axis + 1) % 3;
			nodes[idx].dTree = cur.dTree;
			nodes[idx].dTree.setStatisticalWeightBuilding(nodes[idx].dTree.statisticalWeightBuilding() / 2);
		}
		cur.isLeaf = false;
		cur.dTree = {}; // Reset to an empty dtree to save memory.
	}

	DTreeWrapper* dTreeWrapper(vec3f p, vec3f& size) {
		size = m_aabb.extent();
		p = vec3f(p - m_aabb.lower);
		p.x /= size.x;
		p.y /= size.y;
		p.z /= size.z;

		return m_nodes[0].dTreeWrapper(p, size, m_nodes);
	}

	DTreeWrapper* dTreeWrapper(vec3f p) {
		vec3f size;
		return dTreeWrapper(p, size);
	}

	void forEachDTreeWrapperConst(std::function<void(const DTreeWrapper*)> func) const {
		for (auto& node : m_nodes) {
			if (node.isLeaf) {
				func(&node.dTree);
			}
		}
	}

	void forEachDTreeWrapperConstP(std::function<void(const DTreeWrapper*, const vec3f&, const vec3f&)> func) const {
		m_nodes[0].forEachLeaf(func, m_aabb.lower, m_aabb.upper - m_aabb.lower, m_nodes);
	}

	void forEachDTreeWrapperParallel(std::function<void(DTreeWrapper*)> func) {
		int nDTreeWrappers = static_cast<int>(m_nodes.size());

		// TODO: cuda-parallelize this
		for (int i = 0; i < nDTreeWrappers; ++i) {
			if (m_nodes[i].isLeaf) {
				func(&m_nodes[i].dTree);
			}
		}
	}

	KRR_CALLABLE void record(const vec3f& p, const vec3f& dTreeVoxelSize, DTreeRecord rec, EDirectionalFilter directionalFilter, EBsdfSamplingFractionLoss bsdfSamplingFractionLoss) {
		float volume = 1;
		for (int i = 0; i < 3; ++i) {
			volume *= dTreeVoxelSize[i];
		}

		rec.statisticalWeight /= volume;
		m_nodes[0].record(p - dTreeVoxelSize * 0.5f, p + dTreeVoxelSize * 0.5f, m_aabb.lower, m_aabb.extent(), rec, directionalFilter, bsdfSamplingFractionLoss, m_nodes);
	}

	KRR_CALLABLE bool shallSplit(const STreeNode& node, int depth, size_t samplesRequired) {
		return m_nodes.size() < std::numeric_limits<uint32_t>::max() - 1 && node.dTree.statisticalWeightBuilding() > samplesRequired;
	}

	void refine(size_t sTreeThreshold, int maxMB) {
		if (maxMB >= 0) {
			size_t approxMemoryFootprint = 0;
			for (const auto& node : m_nodes) {
				approxMemoryFootprint += node.dTreeWrapper()->approxMemoryFootprint();
			}

			if (approxMemoryFootprint / 1000000 >= (size_t)maxMB) {
				return;
			}
		}

		struct StackNode {
			size_t index;
			int depth;
		};

		std::stack<StackNode> nodeIndices;
		nodeIndices.push({ 0,  1 });
		while (!nodeIndices.empty()) {
			StackNode sNode = nodeIndices.top();
			nodeIndices.pop();

			// Subdivide if needed and leaf
			if (m_nodes[sNode.index].isLeaf) {
				if (shallSplit(m_nodes[sNode.index], sNode.depth, sTreeThreshold)) {
					subdivide((int)sNode.index, m_nodes);
				}
			}

			// Add children to stack if we're not
			if (!m_nodes[sNode.index].isLeaf) {
				const STreeNode& node = m_nodes[sNode.index];
				for (int i = 0; i < 2; ++i) {
					nodeIndices.push({ node.children[i], sNode.depth + 1 });
				}
			}
		}

		// Uncomment once memory becomes an issue.
		//m_nodes.shrink_to_fit();
	}

	KRR_CALLABLE const AABB& aabb() const {
		return m_aabb;
	}

private:
	std::vector<STreeNode> m_nodes;
	AABB m_aabb;
};


KRR_NAMESPACE_END