#include "scenegraph.h"

KRR_NAMESPACE_BEGIN

const string& SceneGraphLeaf::getName() const { 
	auto node = getNode();
	if (node) return node->getName();
	static const string emptyString = "";
	return emptyString;
}

void SceneGraphLeaf::setName(const std::string &name) const {
	auto node = getNode();
	if (node) node->setName(name);
	else Log(Fatal, "Set name for a leaf without a node");
}

void SceneGraphNode::setTransform(const Vector3f *translation, const Quaternionf *rotation,
								  const Vector3f *scale) {
	if (scale) mScale = *scale;
	if (rotation) mRotation = *rotation;
	if (translation) mTranslation = *translation;

	mHasLocalTransform = true;
}

void SceneGraphNode::setScale(const Vector3f &scale) { setTransform(nullptr, nullptr, &scale); }

void SceneGraphNode::setRotation(const Quaternionf &rotation) {
	setTransform(nullptr, &rotation, nullptr);
}

void SceneGraphNode::setTranslation(const Vector3f &translation) {
	setTransform(&translation, nullptr, nullptr);
}

void SceneGraphNode::setLeaf(const SceneGraphLeaf::SharedPtr &leaf) {
	auto graph = mGraph.lock();

	if (mLeaf) {
		mLeaf->mNode.reset();
		if (graph) graph->unregisterLeaf(mLeaf);
	}

	mLeaf		 = leaf;
	leaf->mNode = weak_from_this();
	if (graph) graph->registerLeaf(leaf);
}

void SceneGraphNode::reverseChildren() {
	// in-place linbked list inversion
	std::shared_ptr<SceneGraphNode> current, prev, next;
	current = mFirstChild;
	while (current) {
		next = current->mNextSibling;
		current->mNextSibling = prev;
		prev = current;
		current = next;
	}
	mFirstChild = prev;
}

int SceneGraphWalker::next(bool allowChildren) {
	if (!mCurrent) return 0;
	if (allowChildren) {
		auto firstChild = mCurrent->getFirstChild();
		if (firstChild) {
			mCurrent = firstChild;
			return 1;
		}
	}
	int depth = 0;
	while (mCurrent) {
		if (mCurrent == mScope) {
			mCurrent = nullptr;
			return depth;
		}
		auto nextSibling = mCurrent->getNextSibling();
		if (nextSibling) {
			mCurrent = nextSibling;
			return depth;
		}
		mCurrent = mCurrent->getParent();
		--depth;
	}
	return depth;
}

int SceneGraphWalker::up() {
	if (mCurrent == mScope) mCurrent = nullptr;
	if (!mCurrent) return 0;
	mCurrent = mCurrent->getParent();
	return -1;
}

SceneGraphNode::SharedPtr SceneGraph::setRoot(const SceneGraphNode::SharedPtr &root) {
	auto oldRoot = mRoot;
	if (oldRoot) detach(oldRoot);
	attach(nullptr, root);
	return oldRoot;
}

SceneGraphNode::SharedPtr SceneGraph::attach(const SceneGraphNode::SharedPtr &parent,
											 const SceneGraphNode::SharedPtr &child) {
	SceneGraph::SharedPtr parentGraph = parent ? parent->mGraph.lock() : shared_from_this();
	SceneGraph::SharedPtr childGraph  = child->mGraph.lock();
	
	if (!parentGraph && !childGraph) {
		// neither nodes has belonged to a graph...
		DCHECK(parent);
		child->mNextSibling = parent->mFirstChild;
		child->mParent		= parent.get();
		parent->mFirstChild = child;
		return child;
	}

	DCHECK(parentGraph.get() == this);
	SceneGraphNode::SharedPtr attachedChild;

	if (childGraph) {
		// subgraph belongs to another graph, copy the entire subgraph
		SceneGraphNode *currentParent = parent.get();
		SceneGraphWalker walker(child.get());
		while (walker) {
			// copy each node and their topology
			SceneGraphNode::SharedPtr copy = std::make_shared<SceneGraphNode>();
			// the root of the copied subgraph...
			if (!attachedChild) attachedChild = copy;

			copy->mName					   = walker->mName;
			copy->mParent				   = currentParent;
			copy->mGraph				   = weak_from_this();
			if (walker->mHasLocalTransform) {
				copy->setTransform(&walker->getTranslation(), &walker->getRotation(),
								   &walker->getScale());
			}
			if (walker->mLeaf) {
				copy->setLeaf(walker->mLeaf->clone());
			}
			if (currentParent) {
				copy->mNextSibling		   = parent->mFirstChild;
				currentParent->mFirstChild = copy;
			} else {
				// parent do not exist at the beginning...
				mRoot = copy;
			}

			if (!attachedChild) attachedChild = copy;

			int deltaDepth = walker.next(true);
			if (deltaDepth > 0) {
				currentParent = copy.get();
			} else {
				while (deltaDepth++ < 0) {
					// make the children linked list same as original...
					currentParent->reverseChildren();
					currentParent = currentParent->mParent;
				}
			}
		}

	} else {
		// attach an orphaned subgraph...
		SceneGraphWalker walker(child.get());
		while (walker) {
			walker->mGraph = weak_from_this();
			auto leaf	   = walker->getLeaf();
			if (leaf) registerLeaf(leaf);
			walker.next(true);
		}
		child->mParent = parent.get();
		if (parent) {
			child->mNextSibling = parent->mFirstChild;
			parent->mFirstChild = child;
		} else {
			// parent == nullptr, i.e. set root.
			mRoot = child;
		}
		attachedChild = child;
	}
	return attachedChild;
}

SceneGraphNode::SharedPtr SceneGraph::attachLeaf(const SceneGraphNode::SharedPtr &parent,
												 const SceneGraphLeaf::SharedPtr &leaf) {
	auto node = std::make_shared<SceneGraphNode>();
	if (leaf->getNode()) node->setLeaf(leaf->clone());
	else node->setLeaf(leaf);
	return attach(parent, node);
}

SceneGraphNode::SharedPtr SceneGraph::detach(const SceneGraphNode::SharedPtr &node) {
	SceneGraph::SharedPtr graph = node->mGraph.lock();
	if (graph) {
		DCHECK(graph.get() == this);
		SceneGraphWalker walker(node.get());
		while (walker) {
			// traverse the subgraph, unregister all leaves.
			walker->mGraph.reset();
			auto leaf = walker->getLeaf();
			if (leaf) unregisterLeaf(leaf);
			walker.next(true);
		}
		if (node->mParent) {
			SceneGraphNode::SharedPtr *sibling = &node->mParent->mFirstChild;
			// delete this node from the linked list.
			while (*sibling && *sibling != node)
				sibling = & (*sibling)->mNextSibling;
			if (*sibling) *sibling = node->mNextSibling;
		}

		node->mParent = nullptr;
		node->mNextSibling.reset();

		if (mRoot == node) {
			mRoot.reset();
			mRoot = std::make_shared<SceneGraphNode>();
		}

		node->mParent = nullptr;
	}
	return node;
}

void SceneGraph::registerLeaf(const SceneGraphLeaf::SharedPtr &leaf) {
	
}

void SceneGraph::unregisterLeaf(const SceneGraphLeaf::SharedPtr &leaf) {

}

KRR_NAMESPACE_END