#pragma once

#include "common.h"
#include "medium.h"
#include "texture.h"

#include "device/memory.h"
#include "device/buffer.h"

NAMESPACE_BEGIN(krr)

class Triangle;
class Volume;
namespace rt { class DiffuseAreaLight; }

enum class VertexAttribute {
	Position,
	Normal,
	Texcoord,
	Tangent,
	Count
};

namespace rt {
struct MeshData {
	KRR_CALLABLE const MaterialData &getMaterial() const { return *material; }

	TypedBuffer<Vector3f> positions;
	TypedBuffer<Vector3f> normals;
	TypedBuffer<Vector2f> texcoords;
	TypedBuffer<Vector3f> tangents;
	TypedBuffer<Vector3i> indices;
	MaterialData *material;
	MediumInterface mediumInterface;
};

struct InstanceData {
	KRR_CALLABLE const MeshData &getMesh() const { return *mesh; }
	KRR_CALLABLE const MaterialData &getMaterial() const { return mesh->getMaterial(); }
	KRR_CALLABLE const Affine3f getTransform() const { return transform.matrix(); }
	KRR_CALLABLE const Matrix3f getTransposedInverseTransform() const {
		return transform.transposedInverse();
	}

	TypedBuffer<Triangle> primitives;		// Only emissive instances have these
	TypedBuffer<DiffuseAreaLight> lights;

	/* TODO: maybe we can get animated transform(t) by traversable? can we? */
	Transformation transform;
	MeshData *mesh = nullptr;
};
}

class Mesh {
public:
	using SharedPtr = std::shared_ptr<Mesh>;

	AABB computeBoundingBox();
	int getMeshId() const { return meshId; }
	std::shared_ptr<Material> getMaterial() const { return material; }
	AABB getBoundingBox() const { return aabb; }
	std::string getName() const { return name; }

	void setName(const std::string& name) { this->name = name; }
	void setMaterial(std::shared_ptr<Material> material) { this->material = material; }
	void setMedium(std::shared_ptr<Volume> mi, std::shared_ptr<Volume> mo) { inside = mi; outside = mo; }
	void renderUI();

	std::vector<Vector3f> positions;
	std::vector<Vector3f> normals;
	std::vector<Vector2f> texcoords;
	std::vector<Vector3f> tangents;
	std::vector<Vector3i> indices;

	std::shared_ptr<Material> material;
	std::shared_ptr<Volume> inside, outside;

	int meshId{-1};
	std::string name;
	AABB aabb{};
	RGB Le{};		/* A mesh-specific area light, used when importing pbrt formats. */
};

NAMESPACE_END(krr)