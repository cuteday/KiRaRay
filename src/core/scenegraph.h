#pragma once

#include "common.h"
#include "logger.h"
#include "mesh.h"
#include "animation.h"
#include "texture.h"
#include "krrmath/math.h"
#include "util/volume.h"

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
	virtual AABB getLocalBoundingBox() const { return AABB(); }
	virtual std::shared_ptr<SceneGraphLeaf> clone() = 0;
	virtual void renderUI() {}

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
	virtual AABB getLocalBoundingBox() const override { return mMesh->getBoundingBox(); }
	virtual void renderUI() override;

private:
	friend class SceneGraph;
	Mesh::SharedPtr mMesh;
	int mInstanceId{-1};
};

class SceneLight : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<SceneLight>;
	enum class Type : uint32_t {
		Undefined		 = 0,
		PointLight		 = 1,
		DirectionalLight = 2,
		InfiniteLight	 = 3
	};
	SceneLight()		= default;
	SceneLight(const Color &color, const float scale) : color(color), scale(scale) {}
	virtual Type getType() const { return Type::Undefined; }
	virtual void renderUI() override;

	void setColor(const Color &_color) { color = _color; setUpdated();}
	void setScale(const float _scale) { scale = _scale; setUpdated();}
	void setPosition(const Vector3f& position);
	void setDirection(const Vector3f& direction);
	Color getColor() const { return color; }
	float getScale() const { return scale; }
	Vector3f getPosition() const;
	Vector3f getDirection() const;
	bool isUpdated() const { return updated; }
	void setUpdated(bool _updated = true) { updated = _updated; }

protected:
	Color color;
	float scale;
	bool updated{false};
};

class PointLight : public SceneLight {
public:
	using SharedPtr = std::shared_ptr<PointLight>;
	using SceneLight::SceneLight;

	virtual std::shared_ptr<SceneGraphLeaf> clone() override;
	virtual Type getType() const override { return Type::PointLight; }
};

class DirectionalLight : public SceneLight {
public:
	using SharedPtr = std::shared_ptr<DirectionalLight>;
	using SceneLight::SceneLight;

	virtual std::shared_ptr<SceneGraphLeaf> clone() override;
	virtual Type getType() const override { return Type::DirectionalLight; }
	virtual void renderUI() override;
};

class InfiniteLight : public SceneLight {
public:
	using SharedPtr = std::shared_ptr<InfiniteLight>;
	using SceneLight::SceneLight;
	InfiniteLight(Texture::SharedPtr texture, const float scale = 1) :
		SceneLight(Color::Ones(), 1), texture(texture) {}

	virtual std::shared_ptr<SceneGraphLeaf> clone() override;
	virtual Type getType() const override { return Type::InfiniteLight; }

	void setTexture(Texture::SharedPtr _texture) { texture = texture; }
	Texture::SharedPtr getTexture() const { return texture; }

protected:
	Texture::SharedPtr texture;
};

class Volume : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<Volume>;
	Volume()		= default;

	int getMediumId() const { return mediumId; }
	void setMediumId(int id) { mediumId = id; }

protected:
	friend class SceneGraph;
	int mediumId{-1};
};

class HomogeneousVolume : public Volume {
public:
	using SharedPtr = std::shared_ptr<HomogeneousVolume>;

	virtual std::shared_ptr<SceneGraphLeaf> clone() override;
	virtual void renderUI() override;

	bool isEmissive() const { return !Le.isZero(); }

	Color sigma_a;
	Color sigma_s;
	Color Le;
	float g;

private:
	friend class SceneGraph;
};

class VDBVolume : public Volume {
public:
	using SharedPtr = std::shared_ptr<VDBVolume>;

	VDBVolume(Color sigma_a, Color sigma_s, float g, fs::path density) :
		sigma_a(sigma_a), sigma_s(sigma_s), g(g) {
		densityGrid = loadNanoVDB(density);
	}

	VDBVolume(Color sigma_a, Color sigma_s, float g, NanoVDBGrid::SharedPtr density) :
		sigma_a(sigma_a), sigma_s(sigma_s), g(g), densityGrid(density) {}

	virtual std::shared_ptr<SceneGraphLeaf> clone() override;
	virtual void renderUI() override;

	Color sigma_a;
	Color sigma_s;
	float g;
	NanoVDBGrid::SharedPtr densityGrid;

protected:
	friend class SceneGraph;
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

class SceneAnimationChannel {
public:
	using SharedPtr = std::shared_ptr<SceneAnimationChannel>;
	SceneAnimationChannel(anime::Sampler::SharedPtr sampler,
						  const SceneGraphNode::SharedPtr targetNode,
						  anime::AnimationAttribute attribute) :
		mSampler(std::move(sampler)), mTargetNode(targetNode), mAttribute(attribute) {}
	
	bool isValid() const { return !mTargetNode.expired(); }
	const anime::Sampler::SharedPtr &getSampler() const { return mSampler; }
	anime::AnimationAttribute getAttribute() const { return mAttribute; }
	SceneGraphNode::SharedPtr getTargetNode() const { return mTargetNode.lock(); }
	void setTargetNode(const SceneGraphNode::SharedPtr &node) { mTargetNode = node; }
	void renderUI();
	bool apply(float time) const;

private:
	anime::Sampler::SharedPtr mSampler;
	anime::AnimationAttribute mAttribute;
	std::weak_ptr<SceneGraphNode> mTargetNode;
};

class SceneAnimation : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<SceneAnimation>;
	SceneAnimation() = default;

	const std::vector<SceneAnimationChannel::SharedPtr> &getChannels() const { return mChannels; }
	float getDuration() const { return mDuration; }
	bool isValid() const;
	bool apply(float time) const;
	void addChannel(const SceneAnimationChannel::SharedPtr &channel);
	virtual void renderUI() override;
	virtual std::shared_ptr<SceneGraphLeaf> clone() override;

private:
	std::vector<SceneAnimationChannel::SharedPtr> mChannels;
	float mDuration = 0.f;
};

class SceneGraph : public std::enable_shared_from_this<SceneGraph> {
public:
	using UpdateRecord = struct { size_t frameIndex; SceneGraphNode::UpdateFlags updateFlags; };
	using SharedPtr = std::shared_ptr<SceneGraph>;
	SceneGraph()		  = default;
	virtual ~SceneGraph() = default;

	void update(size_t frameIndex);
	void animate(double currentTime);

	const SceneGraphNode::SharedPtr &getRoot() const { return mRoot; }
	std::vector<MeshInstance::SharedPtr> &getMeshInstances() { return mMeshInstances; }
	std::vector<Mesh::SharedPtr> &getMeshes() { return mMeshes; }
	std::vector<Material::SharedPtr> &getMaterials() { return mMaterials; }
	std::vector<SceneAnimation::SharedPtr> &getAnimations() { return mAnimations; }
	std::vector<SceneLight::SharedPtr> &getLights() { return mLights; }
	std::vector<Volume::SharedPtr> &getMedia() { return mMedia; }
	UpdateRecord getLastUpdateRecord() const { return mLastUpdateRecord; };
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
	std::vector<SceneAnimation::SharedPtr> mAnimations;
	std::vector<SceneLight::SharedPtr> mLights;
	std::vector<Volume::SharedPtr> mMedia;
	UpdateRecord mLastUpdateRecord;
};

KRR_NAMESPACE_END