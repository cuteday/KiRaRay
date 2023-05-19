#pragma once

#include "common.h"
#include "texture.h"

#include "device/memory.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class Triangle;
class DiffuseAreaLight;

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
	TypedBuffer<Triangle> primitives;
	TypedBuffer<DiffuseAreaLight> lights;
	MaterialData *material;
	uint materialId;
};

struct InstanceData {
	KRR_CALLABLE const MeshData &getMesh() const { return *mesh; }
	KRR_CALLABLE const MaterialData &getMaterial() const { return mesh->getMaterial(); }
	KRR_CALLABLE const Affine3f &getTransform() const { return transform; }
	KRR_CALLABLE const Vector3f &getScaling() const { return scaling; }
	KRR_CALLABLE const Quaternionf &getRotation() const { return rotation; }
	KRR_CALLABLE const Vector3f &getTranslation() const { return translation; }

	Affine3f transform;
	Vector3f scaling;
	Quaternionf rotation;
	Vector3f translation;
	MeshData *mesh;
	uint meshId;
};
}

class Mesh {
public:
	using SharedPtr = std::shared_ptr<Mesh>;

	AABB computeAABB();
	int getMeshId() const { return meshId; }
	std::shared_ptr<Material> getMaterial() const { return material; }
	AABB getAABB() const { return aabb; }

	std::vector<Vector3f> positions;
	std::vector<Vector3f> normals;
	std::vector<Vector2f> texcoords;
	std::vector<Vector3f> tangents;
	std::vector<Vector3i> indices;

	std::shared_ptr<Material> material;
	int meshId{-1};
	AABB aabb{};
	Color Le{};		/* A mesh-specific area light, used when importing pbrt formats. */
};

KRR_NAMESPACE_END