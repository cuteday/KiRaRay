/*  This code should be run on host side. 
	All the data manipulated by this code should reside on host memory. */
#include "tree.h"
#include "integrator.h"

/*	Max d-tree depth: 20; max d-tree nodes: 2e16.
 *	Max s-tree nodes: 2e32; max s-tree depth is unlimited.
 */

KRR_NAMESPACE_BEGIN

KRR_HOST void QuadTreeNode::initialize() {
	for (size_t i = 0; i < 4/*m_sum.size()*/; ++i) {
		m_children[i] = 0;
		storeAtomic(m_sum[i], (AtomicType) 0.0);
	}
}

KRR_HOST void QuadTreeNode::copyFrom(const QuadTreeNode& arg) {
	for (int i = 0; i < 4; ++i) {
		setSum(i, arg.sum(i));
		m_children[i] = arg.m_children[i];
	}
}

KRR_HOST QuadTreeNode::QuadTreeNode(const QuadTreeNode& arg) {
	copyFrom(arg);
}

KRR_HOST QuadTreeNode& QuadTreeNode::operator=(const QuadTreeNode& arg) {
	copyFrom(arg);
	return *this;
}

/*	Ensure that each quadtree node's sum of irradiance estimates equals that of all its children.
	This function do not change the topology of the D-Tree. */
KRR_HOST void QuadTreeNode::build(std::vector<QuadTreeNode>& nodes) {		// [called by host]
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

KRR_HOST void DTree::initialize() {
	storeAtomic(m_sum, (AtomicType) 0.0);
	storeAtomic(m_statisticalWeight, (AtomicType) 0.0);
	m_maxDepth = 0;
	std::vector<QuadTreeNode> nodes(1);
	nodes.front().initialize();
	m_nodes.alloc_and_copy_from_host(nodes);
	CUDA_SYNC_CHECK();
}

KRR_HOST void DTree::clear() {
	CUDA_SYNC(m_nodes.clear());
	m_sum.store(0);
	m_statisticalWeight.store(0);
	m_maxDepth = 0;
}

KRR_HOST DTree& DTree::operator = (const DTree& other) {
	m_nodes = other.m_nodes;	/* Deep copies the D-Tree nodes. */
	m_maxDepth = other.m_maxDepth;
	m_sum.store(other.m_sum);
	m_statisticalWeight.store(other.m_statisticalWeight);
	return *this;
}

KRR_HOST DTree::DTree(const DTree& other) {
	*this = other;
}

/* This function adaptively subdivides / prunes the D-Tree, recursively in a sequential manner. */
KRR_HOST void DTree::reset(const DTree& previousDTree, int newMaxDepth, float subdivisionThreshold) {
	struct StackNode {
		size_t nodeIndex;
		size_t otherNodeIndex;
		const std::vector<QuadTreeNode>* nodes;
		int depth;
	};

	/* Do the adaptive subdivision on host-side. */
	std::vector<QuadTreeNode> this_nodes(1);
	std::vector<QuadTreeNode> other_nodes(previousDTree.m_nodes.size());
	storeAtomic(m_sum, (AtomicType) 0.0);
	storeAtomic(m_statisticalWeight, (AtomicType) 0.0);
	m_maxDepth = 0;
	this_nodes.back().initialize();
	previousDTree.m_nodes.copy_to_host(other_nodes.data(), previousDTree.m_nodes.size());

	std::stack<StackNode> nodeIndices;
	nodeIndices.push({ 0, 0, &other_nodes, 1 });

	const float total = previousDTree.m_sum;

	// Create the topology of the new DTree to be the refined version
	// of the previous DTree. Subdivision is recursive if enough energy is there.
	while (!nodeIndices.empty()) {
		StackNode sNode = nodeIndices.top();
		nodeIndices.pop();

		m_maxDepth = max(m_maxDepth, sNode.depth);

		for (int i = 0; i < 4; ++i) {
			const QuadTreeNode& otherNode = (*sNode.nodes)[sNode.otherNodeIndex];
			/* This makes each d-tree have 85+ nodes (4 layers) even without any radiance records. */
			const float fraction = total > 0 ? (otherNode.sum(i) / total) : pow(0.25f, sNode.depth);
			CHECK_LE(fraction, 1.f + M_EPSILON);

			if (sNode.depth < newMaxDepth && fraction > subdivisionThreshold) {
				if (!otherNode.isLeaf(i)) {
					CHECK_EQ(sNode.nodes, &other_nodes);
					nodeIndices.push({ this_nodes.size(), otherNode.child(i), &other_nodes, sNode.depth + 1 });
				}
				else {
					nodeIndices.push({ this_nodes.size(), this_nodes.size(), &this_nodes, sNode.depth + 1 });
				}

				this_nodes[sNode.nodeIndex].setChild(i, static_cast<uint16_t>(this_nodes.size()));
				this_nodes.emplace_back();
				this_nodes.back().initialize();
				this_nodes.back().setSum(otherNode.sum(i) / 4);

				if (this_nodes.size() > std::numeric_limits<uint16_t>::max()) {
					logWarning("[ResetDTree] DTreeWrapper hit maximum children count (65536).");
					nodeIndices = std::stack<StackNode>();
					break;
				}
			}
		}
	}

	for (auto& node : this_nodes) {	/* zeros all the radiance of nodes, rebuild them later. */
		node.setSum(0);
	}

	/* Copy the processed host nodes to device side. */
	m_nodes.alloc_and_copy_from_host(this_nodes);
	CUDA_SYNC_CHECK();
}

/* Make sure *this is on host memory now. This function would not change the topology of the D-Tree. */
KRR_HOST void DTree::build() {
	size_t n_nodes = m_nodes.size();
	std::vector<QuadTreeNode> nodes(n_nodes);
	m_nodes.copy_to_host(nodes.data(), n_nodes);

	QuadTreeNode& root = nodes[0];
	// Build the quadtree recursively, starting from its root.
	root.build(nodes);

	// Ensure that the overall sum of irradiance estimates equals
	// the sum of irradiance estimates found in the quadtree.
	float sum = 0;
	for (int i = 0; i < 4; ++i) {
		sum += root.sum(i);
	}
	storeAtomic(m_sum, (AtomicType) sum);
	m_nodes.copy_from_host(nodes.data(), n_nodes);
}

KRR_HOST void DTreeWrapper::initialize() {
	CUDA_SYNC(sampling.initialize());
	CUDA_SYNC(building.initialize());
}

KRR_HOST void DTreeWrapper::clear() {
	CUDA_SYNC(sampling.clear());
	CUDA_SYNC(building.clear());
}

KRR_HOST DTreeWrapper& DTreeWrapper::operator = (const DTreeWrapper& other) {
	building = other.building;
	sampling = other.sampling;
	return *this;
}

KRR_HOST void DTreeWrapper::build() {
	building.build();
	sampling.clear();
	sampling = building;
}

KRR_HOST void DTreeWrapper::reset(int maxDepth, float subdivisionThreshold) {
	building.reset(sampling, maxDepth, subdivisionThreshold);
}

KRR_HOST void STreeNode::initialize() {
	isLeaf = true;
	dTree.initialize();
	CUDA_SYNC_CHECK();
}

KRR_HOST void STreeNode::forEachLeaf(
	std::function<void(const DTreeWrapper*, const Vector3f&, const Vector3f&)> func,
	Vector3f p, Vector3f size, const TypedBuffer<STreeNode>& nodes) const {

	if (isLeaf) {
		func(&dTree, p, size);
	}
	else {
		size[axis] /= 2;
		for (int i = 0; i < 2; ++i) {
			Vector3f childP = p;
			if (i == 1) {
				childP[axis] += size[axis];
			}

			nodes[children[i]].forEachLeaf(func, childP, size, nodes);
		}
	}
}

KRR_HOST STree::STree(const AABB& aabb, Allocator alloc) {
	clear();
	m_aabb = aabb;
	// Enlarge AABB to turn it into a cube. This has the effect of nicer hierarchical subdivisions.
	Vector3f size = m_aabb.diagonal();
	m_aabb.extend(m_aabb.min() + Vector3f(size.maxCoeff()));
}

KRR_HOST void STree::clear() {
	forEachDTreeWrapper([](DTreeWrapper* dtree) {
		dtree->clear();	/* free the memories of the quadtrees */
		});
	std::vector<STreeNode> nodes(1);
	nodes.front().initialize();
	m_nodes.alloc_and_copy_from_host(nodes);	// initialize the super root tree node
}

KRR_HOST void STree::subdivideAll() {
	int nNodes = (int)m_nodes.size();
	std::vector<STreeNode> nodes(nNodes);
	m_nodes.copy_to_host(nodes.data(), nNodes);
	for (int i = 0; i < nNodes; ++i) {
		if (m_nodes[i].isLeaf) {
			subdivide(i, nodes);
		}
	}
	m_nodes.alloc_and_copy_from_host(nodes);
}

/* This is the actual function that directly changes the topology of the S-Tree. */
KRR_HOST void STree::subdivide(int nodeIdx, std::vector<STreeNode>& nodes) {
	// Add 2 child nodes
	nodes.resize(nodes.size() + 2);

	if (nodes.size() > std::numeric_limits<uint32_t>::max()) {
		logWarning("[SubdivideSTree] DTreeWrapper hit maximum children count.");
		return;
	}

	STreeNode& cur = nodes[nodeIdx];
	for (int i = 0; i < 2; ++i) {
		uint32_t idx = (uint32_t)nodes.size() - 2 + i;
		STreeNode& child = nodes[idx];
		cur.children[i] = idx;
		child.initialize();
		child.axis = (cur.axis + 1) % 3;
		child.dTree = cur.dTree;
		child.dTree.setStatisticalWeightBuilding(child.dTree.statisticalWeightBuilding() / 2);
	}
	cur.isLeaf = false;
	cur.dTree.clear(); // Reset to an empty dtree to save memory.
	CUDA_SYNC_CHECK();
}

KRR_HOST void STree::forEachDTreeWrapper(std::function<void(DTreeWrapper*)> func) {
	int n_nodes = static_cast<int>(m_nodes.size());
	Log(Info, "[ForEachDTreeWrapper] There are %d S-Tree nodes to process...", n_nodes);
	// TODO: parallelize this on cpu
	std::vector<STreeNode> nodes(n_nodes);
	m_nodes.copy_to_host(nodes.data(), n_nodes);

	for (int i = 0; i < n_nodes; ++i) {
		//Log(Info, "Processing the %d-th D-Tree...", i);
		if (nodes[i].isLeaf) {
			func(&nodes[i].dTree);
		}
	}
	m_nodes.copy_from_host(nodes.data(), nodes.size());
}

KRR_HOST bool STree::shallSplit(const STreeNode& node,size_t samplesRequired) {
	//Log(Info, "The node at depth %d has a statistical weight of %f, with the specified required sample is %zd",
	//	depth, node.dTree.statisticalWeightBuilding(), samplesRequired);
	return node.dTree.statisticalWeightBuilding() > samplesRequired;
}

/*	Only this function would change the topology of the S-Tree, sequentially executed.
	To adaptively subdivision, should be run on hostcode? */
KRR_HOST void STree::refine(size_t sTreeThreshold, int maxMB) {
	// These work are done on CPU!
	size_t n_nodes = m_nodes.size();
	Log(Info, "[REFINE] There are %zd S-Tree nodes to traverse...", n_nodes);
	std::vector<STreeNode> nodes(n_nodes);
	m_nodes.copy_to_host(nodes.data(), n_nodes);

	if (maxMB >= 0) {
		size_t approxMemoryFootprint = 0;
		for (const auto& node : nodes) {
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
		//Log(Info, "Traversing the %zd-th S-Tree node at depth %d", sNode.index, sNode.depth);

		// Subdivide if needed and leaf
		if (nodes[sNode.index].isLeaf) {
			if (nodes.size() < std::numeric_limits<uint32_t>::max()
				&& shallSplit(nodes[sNode.index], sTreeThreshold)) {
				subdivide((int)sNode.index, nodes);
			}
		}

		// Add children to stack if we're not
		if (!nodes[sNode.index].isLeaf) {
			const STreeNode& node = nodes[sNode.index];
			for (int i = 0; i < 2; ++i) {
				nodeIndices.push({ node.children[i], sNode.depth + 1 });
			}
		}
	}

	/* Copy the host-side new S-Tree to device side */
	m_nodes.alloc_and_copy_from_host(nodes);
	CUDA_SYNC_CHECK();
}

/* [At the begining of each iteration]
	Adaptively subdivide the S-Tree,
	and resets the distribution within the (building) D-Tree.
	The irradiance records within D-Tree is cleared after this. */
void PPGPathTracer::resetSDTree() {
	cudaDeviceSynchronize();
	/* About 18k at the first iteration. */
	float sTreeSplitThres = sqrt(pow(2, m_iter) * m_sppPerPass / 4) * m_sTreeThreshold;
	Log(Info, "Adaptively subdividing the S-Tree. Current split threshould: %.2f", sTreeSplitThres);
	m_sdTree->refine((size_t) sTreeSplitThres, m_sdTreeMaxMemory);
	cudaDeviceSynchronize();
	Log(Info, "Adaptively subdividing the D-Tree...");
	m_sdTree->forEachDTreeWrapper([this](DTreeWrapper* dTree) {
		dTree->reset(20 /* max d-tree depth */, m_dTreeThreshold);
		});
	CUDA_SYNC_CHECK();
}

/* [At the end of each iteration] Build the sampling distribution with statistics, in the current iteration of the building tree, */
/* Then use it, on the sampling tree, in the next iteration. */
void PPGPathTracer::buildSDTree() {
	cudaDeviceSynchronize();
	// Build distributions
	Log(Info, "Building distributions for each D-Tree node...");
	m_sdTree->forEachDTreeWrapper([](DTreeWrapper* dTree) {
		dTree->build();
		});
	m_isBuilt = true;
	CUDA_SYNC_CHECK();
}

/* Collect the statistics for the sampling tree (not the building tree that get reset). */
KRR_HOST void STree::gatherStatistics() const {
	cudaDeviceSynchronize();
	size_t n_nodes = m_nodes.size();
	std::vector<STreeNode> nodes(n_nodes);
	m_nodes.copy_to_host(nodes.data(), n_nodes);
	
	int maxDepth = 0;
	int minDepth = std::numeric_limits<int>::max();
	float avgDepth = 0;
	float maxAvgRadiance = 0;
	float minAvgRadiance = std::numeric_limits<float>::max();
	float avgAvgRadiance = 0;
	size_t maxNodes = 0;
	size_t minNodes = std::numeric_limits<size_t>::max();
	float avgNodes = 0;
	float maxStatisticalWeight = 0;
	float minStatisticalWeight = std::numeric_limits<float>::max();
	float avgStatisticalWeight = 0;

	int nPoints = 0;
	int nPointsNodes = 0;

	for (const auto& node : nodes) {
		if (node.isLeaf) {
			/* These statistics are all gathered from the sampling tree. */
			const DTreeWrapper* dTree = node.dTreeWrapper();
			const int depth = dTree->depth();
			maxDepth = max(maxDepth, depth);
			minDepth = min(minDepth, depth);
			avgDepth += depth;
			
			const float avgRadiance = dTree->meanRadiance();
			maxAvgRadiance = max(maxAvgRadiance, avgRadiance);
			minAvgRadiance = min(minAvgRadiance, avgRadiance);
			avgAvgRadiance += avgRadiance;

			if (dTree->numNodes() > 1) {
				const size_t cur_nodes = dTree->numNodes();
				maxNodes = max(maxNodes, cur_nodes);
				minNodes = min(minNodes, cur_nodes);
				avgNodes += cur_nodes;
				++nPointsNodes;
			}

			const float statisticalWeight = dTree->statisticalWeight();
			maxStatisticalWeight = max(maxStatisticalWeight, statisticalWeight);
			minStatisticalWeight = min(minStatisticalWeight, statisticalWeight);
			avgStatisticalWeight += statisticalWeight;
			++nPoints;
		}
	}
	
	if (nPoints > 0) {
		avgDepth /= nPoints;
		avgAvgRadiance /= nPoints;
		if (nPointsNodes > 0) {
			avgNodes /= nPointsNodes;
		}
		avgStatisticalWeight /= nPoints;
	}

	printf(
		"Distribution statistics:\n"
		"  Depth of D-Trees         = [%d, %f, %d]\n"
		"  Mean radiance            = [%f, %f, %f]\n"
		"  Node count of D-Trees    = [%d, %f, %d]\n"
		"  Stat. weight	of D-Trees  = [%f, %f, %f]\n",
		minDepth, avgDepth, maxDepth,
		minAvgRadiance, avgAvgRadiance, maxAvgRadiance,
		static_cast<int>(minNodes), avgNodes, static_cast<int>(maxNodes),
		minStatisticalWeight, avgStatisticalWeight, maxStatisticalWeight
	);
}

KRR_NAMESPACE_END