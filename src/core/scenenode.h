#pragma once

#include "common.h"
#include "krrmath/math.h"

NAMESPACE_BEGIN(krr)

class SceneGraph;
class SceneGraphNode;

class SceneGraphLeaf {
public:
	using SharedPtr			  = std::shared_ptr<SceneGraphLeaf>;
	virtual ~SceneGraphLeaf() = default;

	enum struct ContentFlags : uint32_t {
		None	  = 0,
		Mesh	  = 1 << 0,
		Material  = 1 << 1,
		Animation = 1 << 2,
		Light	  = 1 << 3,
		Volume	  = 1 << 4,
		Camera	  = 1 << 5,
		All		  = (Mesh | Material | Animation | Light | Volume | Camera),
	};

	SceneGraphNode *getNode() const { return mNode.lock().get(); }
	std::shared_ptr<SceneGraphNode> getNodeSharedPtr() const { return mNode.lock(); }
	virtual AABB getLocalBoundingBox() const { return AABB::Zero(); }
	virtual std::shared_ptr<SceneGraphLeaf> clone() = 0;
	virtual void renderUI() {}
	virtual ContentFlags getContentFlags() const { return ContentFlags::None; }

	virtual void setUpdated(bool updated = true);
	virtual bool isUpdated() const { return mUpdated; }

	const std::string &getName() const;
	void setName(const std::string &name) const;

	// Non-copyable and non-movable.
	SceneGraphLeaf(const SceneGraphLeaf &)			   = delete;
	SceneGraphLeaf(const SceneGraphLeaf &&)			   = delete;
	SceneGraphLeaf &operator=(const SceneGraphLeaf &)  = delete;
	SceneGraphLeaf &operator=(const SceneGraphLeaf &&) = delete;

protected:
	SceneGraphLeaf() = default;
	bool mUpdated{};

private:
	friend class SceneGraphNode;
	std::weak_ptr<SceneGraphNode> mNode;
};

class SceneGraphNode final : public std::enable_shared_from_this<SceneGraphNode> {
public:
	using SharedPtr = std::shared_ptr<SceneGraphNode>;
	using ContentFlags = SceneGraphLeaf::ContentFlags;

	enum struct UpdateFlags : uint32_t {
		None			  = 0,
		LocalTransform	  = 1 << 0,
		Leaf			  = 1 << 1,		// A leaf update subjects to a subgraph structure change.
		SubgraphStructure = 1 << 2,
		SubgraphContent	  = 1 << 3,
		SubgraphTransform = 1 << 4,
		SubgraphUpdates	  = (SubgraphStructure | SubgraphContent | SubgraphTransform),
	};
	
	SceneGraphNode() = default;
	SceneGraphNode(std::string name) : mName(name) {}
	~SceneGraphNode() = default;
	
	const std::string &getName() const { return mName; }
	const std::shared_ptr<SceneGraph> getGraph() const { return mGraph.lock(); }

	const Quaternionf &getRotation() const { return mRotation; }
	const Vector3f &getScaling() const { return mScaling; }
	const Vector3f &getTranslation() const { return mTranslation; }

	// Topology for traversal
	SceneGraphNode *getParent() const { return mParent; }
	SceneGraphNode *getFirstChild() const { return mFirstChild.get(); }
	SceneGraphNode *getNextSibling() const { return mNextSibling.get(); }
	const SceneGraphLeaf::SharedPtr getLeaf() const { return mLeaf; }
	
	void setTransform(const Vector3f *translation, const Quaternionf *rotation,
					  const Vector3f *scaling);
	void setScaling(const Vector3f &scaling);
	void setRotation(const Quaternionf& rotation);
	void setTranslation(const Vector3f &translation);
	void setName(const std::string &name) { mName = name; }
	void setLeaf(const SceneGraphLeaf::SharedPtr &leaf);
	
	Affine3f getLocalTransform() const { return mLocalTransform; }
	Affine3f getGlobalTransform() const { return mGlobalTransform; }
	AABB getGlobalBoundingBox() const { return mGlobalBoundingBox; }

	void updateLocalTransform();
	void propagateUpdateFlags(UpdateFlags flags);
	void propagateContentFlags(ContentFlags flags);
	void reverseChildren();
	UpdateFlags getUpdateFlags() const { return mUpdateFlags; }
	ContentFlags getContentFlags() const { return mContentFlags; }
	
	// Non-copyable and non-movable
	SceneGraphNode(const SceneGraphNode &)			   = delete;
	SceneGraphNode(const SceneGraphNode &&)			   = delete;
	SceneGraphNode &operator=(const SceneGraphNode &)  = delete;
	SceneGraphNode &operator=(const SceneGraphNode &&) = delete;

	void renderUI();
private:
	friend class SceneGraph;
	std::weak_ptr<SceneGraph> mGraph;
	SceneGraphNode *mParent{nullptr};
		
	SceneGraphNode::SharedPtr mFirstChild;
	SceneGraphNode::SharedPtr mNextSibling;
	SceneGraphLeaf::SharedPtr mLeaf;

	std::string mName;
	Affine3f mLocalTransform = Affine3f::Identity();
	Affine3f mGlobalTransform = Affine3f::Identity();
	// S.R.T. transformation
	Quaternionf mRotation = Quaternionf::Identity();
	Vector3f mScaling	  = Vector3f::Ones();
	Vector3f mTranslation = Vector3f::Zero();
	AABB3f mGlobalBoundingBox = AABB3f::Zero();
	bool mHasLocalTransform{false};
	
	UpdateFlags mUpdateFlags = UpdateFlags::None;
	ContentFlags mContentFlags = ContentFlags::None;
};

KRR_ENUM_OPERATORS(SceneGraphNode::UpdateFlags)
KRR_ENUM_OPERATORS(SceneGraphLeaf::ContentFlags)


class SceneGraphWalker final {
public:
	SceneGraphWalker() = default;
	
	explicit SceneGraphWalker(SceneGraphNode *scope)
		: mCurrent(scope), mScope(scope) {}

	SceneGraphWalker(SceneGraphNode *current, SceneGraphNode* scope)
		: mCurrent(current), mScope(scope) {}
	
	SceneGraphNode *get() const { return mCurrent; }
	SceneGraphNode *operator->() const { return mCurrent; }
	operator bool() const { return mCurrent != nullptr; }

    // Moves the pointer to the first child of the current node, if it exists & allowChildren.
	// Otherwise, moves the pointer to the next sibling of the current node, if it exists.
	// Otherwise, goes up and tries to find the next sibiling up the hierarchy.
	// Returns the depth of the new node relative to the current node (negative means go upper).
	int next(bool allowChildren);
	// Moves to parent, up to the specified scope (root of a supergraph). 
	int up();

private:
	SceneGraphNode *mCurrent{nullptr};
	SceneGraphNode *mScope{nullptr};
};

NAMESPACE_END(krr)