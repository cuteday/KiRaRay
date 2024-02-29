#pragma once

#include "common.h"
#include "logger.h"
#include "mesh.h"
#include "animation.h"
#include "texture.h"
#include "scenenode.h"
#include "krrmath/math.h"
#include "util/volume.h"

NAMESPACE_BEGIN(krr)

class SceneGraph;

class MeshInstance : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<MeshInstance>;
	explicit MeshInstance(Mesh::SharedPtr mesh) : mMesh(mesh){};

	std::shared_ptr<SceneGraphLeaf> clone() override;
	const Mesh::SharedPtr &getMesh() const { return mMesh; }
	int getInstanceId() const { return mInstanceId; }
	AABB getLocalBoundingBox() const override { return mMesh->getBoundingBox(); }
	void renderUI() override;
	ContentFlags getContentFlags() const override { return ContentFlags::Mesh; }

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
	SceneLight(const RGB &color, const float scale) : color(color), scale(scale) {}
	virtual Type getType() const { return Type::Undefined; }
	void renderUI() override;
	ContentFlags getContentFlags() const override { return ContentFlags::Light; }

	void setColor(const RGB &_color) { color = _color; setUpdated();}
	void setScale(const float _scale) { scale = _scale; setUpdated();}
	void setPosition(const Vector3f& position);
	void setDirection(const Vector3f& direction);
	RGB getColor() const { return color; }
	float getScale() const { return scale; }
	Vector3f getPosition() const;
	Vector3f getDirection() const;
	bool isUpdated() const { return updated; }
	void setUpdated(bool _updated = true) { updated = _updated; }

protected:
	RGB color;
	float scale;
	bool updated{false};
};

class PointLight : public SceneLight {
public:
	using SharedPtr = std::shared_ptr<PointLight>;
	using SceneLight::SceneLight;

	std::shared_ptr<SceneGraphLeaf> clone() override;
	Type getType() const override { return Type::PointLight; }
};

class DirectionalLight : public SceneLight {
public:
	using SharedPtr = std::shared_ptr<DirectionalLight>;
	using SceneLight::SceneLight;

	std::shared_ptr<SceneGraphLeaf> clone() override;
	Type getType() const override { return Type::DirectionalLight; }
	void renderUI() override;
};

class InfiniteLight : public SceneLight {
public:
	using SharedPtr = std::shared_ptr<InfiniteLight>;
	using SceneLight::SceneLight;
	InfiniteLight(Texture::SharedPtr texture, const float scale = 1) :
		SceneLight(RGB::Ones(), 1), texture(texture) {}

	std::shared_ptr<SceneGraphLeaf> clone() override;
	Type getType() const override { return Type::InfiniteLight; }

	void setTexture(Texture::SharedPtr _texture) { texture = _texture; }
	Texture::SharedPtr getTexture() const { return texture; }

protected:
	Texture::SharedPtr texture;
};

class Volume : public SceneGraphLeaf {
public:
	using SharedPtr = std::shared_ptr<Volume>;
	Volume()		= default;
	Volume(RGB sigma_t, RGB albedo, float g) : sigma_t(sigma_t), albedo(albedo), g(g) {}

	int getMediumId() const { return mediumId; }
	void setMediumId(int id) { mediumId = id; }
	ContentFlags getContentFlags() const override { return ContentFlags::Volume; }

	RGB sigma_t;
	RGB albedo;
	float g;
protected:
	friend class SceneGraph;
	int mediumId{-1};
};

class HomogeneousVolume : public Volume {
public:
	using SharedPtr = std::shared_ptr<HomogeneousVolume>;

	HomogeneousVolume(RGB sigma_t, RGB albedo, float g, RGB Le = RGB::Zero(), 
		AABB boundingBox = AABB{std::numeric_limits<float>::max(),
								std::numeric_limits<float>::min()}) :
		Volume(sigma_t, albedo, g), Le(Le), boundingBox(boundingBox) {}

	std::shared_ptr<SceneGraphLeaf> clone() override;
	void renderUI() override;

	AABB getLocalBoundingBox() const override { return boundingBox; }
	bool isEmissive() const { return !Le.isZero(); }

	RGB Le;
	AABB boundingBox;

private:
	friend class SceneGraph;
};

class VDBVolume : public Volume {
public:
	using SharedPtr = std::shared_ptr<VDBVolume>;

	VDBVolume(RGB sigma_t, RGB albedo, float g, NanoVDBGridBase::SharedPtr density,
			  NanoVDBGrid<float>::SharedPtr temperature	 = nullptr,
			  NanoVDBGrid<Array3f>::SharedPtr albedoGrid = nullptr, float scale = 1,
			  float LeScale = 1, float temperatureScale = 1, float temperatureOffset = 0) :
		Volume(sigma_t, albedo, g),
		densityGrid(density),
		temperatureGrid(temperature),
		albedoGrid(albedoGrid),
		scale(scale),
		LeScale(LeScale),
		temperatureScale(temperatureScale),
		temperatureOffset(temperatureOffset) {}

	SceneGraphLeaf::SharedPtr clone() override;
	void renderUI() override;

	AABB getLocalBoundingBox() const override { return densityGrid->getBounds(); }

	float scale, LeScale;
	float temperatureScale, temperatureOffset;
	NanoVDBGridBase::SharedPtr densityGrid;
	NanoVDBGrid<float>::SharedPtr temperatureGrid;
	NanoVDBGrid<Array3f>::SharedPtr albedoGrid;

protected:
	friend class SceneGraph;
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
	float getDuration() const { return std::max(0.f, mEndTime - mStartTime); }
	float getStartTime() const { return mStartTime; }
	float getEndTime() const { return mEndTime; }
	bool isValid() const;
	bool apply(float time) const;
	void addChannel(const SceneAnimationChannel::SharedPtr &channel);
	void renderUI() override;
	ContentFlags getContentFlags() const override { return ContentFlags::Animation; }

	std::shared_ptr<SceneGraphLeaf> clone() override;

private:
	std::vector<SceneAnimationChannel::SharedPtr> mChannels;
	float mStartTime = std::numeric_limits<float>::max();
	float mEndTime	 = std::numeric_limits<float>::min();
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
	unsigned int evaluateMaxTraversalDepth() const;
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

NAMESPACE_END(krr)