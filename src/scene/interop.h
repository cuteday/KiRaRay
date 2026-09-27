#pragma once

#include "scene.h"

NAMESPACE_BEGIN(krr)
namespace interop {

enum class Interpolation { Constant, Uniform, Vertex, FaceVarying };

template <typename T> struct Primvar {
	std::vector<T> values;
	std::vector<int> indices;
	Interpolation interpolation{Interpolation::Vertex};
};

struct MeshInput {
	std::string name;
	std::vector<Vector3f> points;
	std::vector<int> faceCounts;
	std::vector<int> faceIndices;
	std::vector<int> holeFaces;
	Primvar<Vector3f> normals;
	Primvar<Vector2f> texcoords;
	std::vector<Material::SharedPtr> materials;
	std::vector<int> faceMaterials;
	bool leftHanded{};
};

std::vector<Mesh::SharedPtr> makeMeshes(const MeshInput &input);
// Recompute tangents across partitions of one logical mesh in the same local space.
void generateTangents(const std::vector<Mesh::SharedPtr> &meshes);
SceneGraphNode::SharedPtr attachMeshes(Scene::SharedPtr scene,
	const std::vector<Mesh::SharedPtr> &meshes, const Affine3f &transform,
	const std::string &name, SceneGraphNode::SharedPtr parent = nullptr);
Material::SharedPtr errorMaterial(const std::string &name);

struct LightInput {
	enum class Type { Point, Spot, Distant, Rectangle, Disk, Sphere, Dome };
	Type type{Type::Point};
	std::string name;
	Affine3f transform{Affine3f::Identity()};
	RGB color{1.f};
	float intensity{1.f};
	float exposure{};
	float width{1.f}, height{1.f}, radius{0.5f};
	float coneAngle{45.f}, coneSoftness{};
	float distantAngle{};
	bool normalize{};
	std::string texture;
};

SceneGraphNode::SharedPtr attachLight(Scene::SharedPtr scene, const LightInput &input,
	SceneGraphNode::SharedPtr parent = nullptr);

}
NAMESPACE_END(krr)
