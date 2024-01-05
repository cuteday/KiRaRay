#include "scenenode.h"
#include "logger.h"

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
	else Log(Error, "Set name %s for a leaf without a node", name.c_str());
}

void SceneGraphNode::setTransform(const Vector3f *translation, const Quaternionf *rotation,
								  const Vector3f *scaling) {
	if (scaling) mScaling = *scaling;
	if (rotation) mRotation = *rotation;
	if (translation) mTranslation = *translation;

	mHasLocalTransform = true;
	mUpdateFlags |= UpdateFlags::LocalTransform;
	propagateUpdateFlags(UpdateFlags::SubgraphTransform);
}

void SceneGraphNode::setScaling(const Vector3f &scaling) {
	setTransform(nullptr, nullptr, &scaling);
}

void SceneGraphNode::setRotation(const Quaternionf &rotation) {
	setTransform(nullptr, &rotation, nullptr);
}

void SceneGraphNode::setTranslation(const Vector3f &translation) {
	setTransform(&translation, nullptr, nullptr);
}

void SceneGraphNode::updateLocalTransform() {
	mLocalTransform = Affine3f().translate(mTranslation).rotate(mRotation).scale(mScaling);
}

void SceneGraphNode::propagateUpdateFlags(UpdateFlags flags) { 
	SceneGraphWalker walker(this, nullptr);
	while (walker) {
		walker->mUpdateFlags |= flags;	
		walker.up();
	}
}

void SceneGraphNode::propagateContentFlags(ContentFlags flags) {
	SceneGraphWalker walker(this, nullptr);
	while (walker) {
		walker->mContentFlags = flags;
		/*aggregate children contents*/
		SceneGraphWalker child(walker->getFirstChild(), walker.get());
		while (child) {
			walker->mContentFlags |= child->mContentFlags;
			child.next(false);
		}
		walker.up();
	}
}

void SceneGraphNode::reverseChildren() {
	// in-place linked list inversion...
	std::shared_ptr<SceneGraphNode> current, prev, next;
	current = mFirstChild;
	while (current) {
		next				  = current->mNextSibling;
		current->mNextSibling = prev;
		prev				  = current;
		current				  = next;
	}
	mFirstChild = prev;
}


int SceneGraphWalker::next(bool allowChildren) {
	// if allowChildren == true, then next() follows DFS traversal order.
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


KRR_NAMESPACE_END