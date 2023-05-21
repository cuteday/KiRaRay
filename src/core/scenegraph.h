#pragma once
#include "common.h"
#include "logger.h"
#include "mesh.h"
#include "texture.h"
#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN

class SceneGraph;
class SceneGraphNode;
class SceneGraphLeaf;
class SceenGraphWalker;

class SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<SceneGraphLeaf>;
	virtual ~SceneGraphLeaf() = default;

	SceneGraphNode *getNode() const { return mNode.lock().get(); }
	std::shared_ptr<SceneGraphNode> getNodeSharedPtr() const { return mNode.lock(); }
	virtual AABB getLocalBoundingBox() { return AABB(); }
	virtual std::shared_ptr<SceneGraphLeaf> clone() = 0;

	const std::string &getName() const;
	void setName(const std::string &name) const;

	// Non-copyable and non-movable.
	SceneGraphLeaf(const SceneGraphLeaf &)			   = delete;
	SceneGraphLeaf(const SceneGraphLeaf &&)			   = delete;
	SceneGraphLeaf &operator=(const SceneGraphLeaf &)  = delete;
	SceneGraphLeaf &operator=(const SceneGraphLeaf &&) = delete;

protected:
	SceneGraphLeaf() = default;

private:
	friend class SceneGraphNode;
	std::weak_ptr<SceneGraphNode> mNode;
};

class MeshInstance : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<MeshInstance>;
	explicit MeshInstance(Mesh::SharedPtr mesh) : mMesh(mesh){};

	virtual std::shared_ptr<SceneGraphLeaf> clone() override;
	const Mesh::SharedPtr &getMesh() const { return mMesh; }
	int getInstanceId() const { return mInstanceId; }
	virtual AABB getLocalBoundingBox() override { return mMesh->getBoundingBox(); }

private:
	friend class SceneGraph;
	Mesh::SharedPtr mMesh;
	int mInstanceId{-1};
};

class SceneGraphNode final : public std::enable_shared_from_this<SceneGraphNode> {
public:
	using SharedPtr = std::shared_ptr<SceneGraphNode>;
	
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
	void reverseChildren();
	UpdateFlags getUpdateFlags() const { return mUpdateFlags; }
	
	// Non-copyable and non-movable
	SceneGraphNode(const SceneGraphNode &)			   = delete;
	SceneGraphNode(const SceneGraphNode &&)			   = delete;
	SceneGraphNode &operator=(const SceneGraphNode &)  = delete;
	SceneGraphNode &operator=(const SceneGraphNode &&) = delete;

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
	AABB3f mGlobalBoundingBox;
	bool mHasLocalTransform{false};
	
	UpdateFlags mUpdateFlags = UpdateFlags::None;
};

KRR_ENUM_OPERATORS(SceneGraphNode::UpdateFlags)

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

class SceneGraph : public std::enable_shared_from_this<SceneGraph> {
public:
	using SharedPtr = std::shared_ptr<SceneGraph>;
	SceneGraph()		  = default;
	virtual ~SceneGraph() = default;

	void update(size_t frameIndex);

	const SceneGraphNode::SharedPtr &getRoot() const { return mRoot; }
	std::vector<MeshInstance::SharedPtr> &getMeshInstances() { return mMeshInstances; }
	std::vector<Mesh::SharedPtr> &getMeshes() { return mMeshes; }
	std::vector<Material::SharedPtr> &getMaterials() { return mMaterials; }
	void addMesh(Mesh::SharedPtr mesh);
	void addMaterial(Material::SharedPtr material);

	SceneGraphNode::SharedPtr setRoot(const SceneGraphNode::SharedPtr &root);
	SceneGraphNode::SharedPtr attach(const SceneGraphNode::SharedPtr &parent,
									 const SceneGraphNode::SharedPtr &child);
	// Attach a leaf to a new node, then attach that node to specified parent.
	SceneGraphNode::SharedPtr attachLeaf(const SceneGraphNode::SharedPtr &parent,
										const SceneGraphLeaf::SharedPtr &leaf);
	// Detach a node and its subgraph from the graph, then unregister all its resources.
	SceneGraphNode::SharedPtr detach(const SceneGraphNode::SharedPtr &node);

	void printSceneGraph() const;
	void renderUI();

protected:
	virtual void registerLeaf(const SceneGraphLeaf::SharedPtr &leaf);
	virtual void unregisterLeaf(const SceneGraphLeaf::SharedPtr &leaf);

private:
	friend class SceneGraphNode;
	friend class SceneGraphLeaf;
	friend class SceneGraphWalker;

	SceneGraphNode::SharedPtr mRoot;
	
	std::vector<Mesh::SharedPtr> mMeshes;
	std::vector<Material::SharedPtr> mMaterials;
	std::vector<MeshInstance::SharedPtr> mMeshInstances;
};

KRR_NAMESPACE_END