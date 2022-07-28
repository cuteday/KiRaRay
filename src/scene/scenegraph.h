#pragma once
#include <vector>

#include "common.h"
#include "interop.h"
#include "math/math.h"
#include "mesh.h"
#include "shape.h"

KRR_NAMESPACE_BEGIN

/**
 * To use scene graph and mesh instancing in ray tracing, transformations must be applied within RTX program.
 * Specifically, the normal, tangents and bitangents are needed to be rotated.
 * The position, and the localToWorld transformation, could be obtained via optix instrinsic calls.
 * To conclude, 3 more rotations are needed in each intersection.
 */


enum class NodeType {
	GROUP,
	INSTANCE,
	MESH
};

class SceneNode {
public:
	using SharedPtr = std::shared_ptr<SceneNode>;
	SceneNode(const uint id) : mId(id){};
	
	virtual NodeType getType() const = 0;
	uint getId() const { return mId; }

private:
	uint mId;
};

class SceneMesh : public SceneNode {
public:
	using SharedPtr = std::shared_ptr<SceneMesh>;
	struct MeshData {
		VertexAttribute *vertices;
		Vector3i *indices;
	};

	SceneMesh(const uint id) : SceneNode(id) {}
	NodeType getType() const override { return NodeType::MESH; }
	std::vector<Triangle> createTriangles() {
		std::vector<Triangle> triangles;
		for (int i = 0; i < indices.size(); i ++) {
			Triangle triangle;
			triangles.push_back(triangle);
		}
		return triangles;
	}
	
	const inter::vector<VertexAttribute> &getVertices() const { return vertices; }
	const inter::vector<Vector3i> &getIndices() const { return indices; }

	inter::vector<VertexAttribute> vertices;
	inter::vector<Vector3i> indices;
};

class SceneInstance : public SceneNode {
public:
	using SharedPtr = std::shared_ptr<SceneInstance>;
	struct InstanceData {
		int material{};
		int mesh{};
		inter::vector<DiffuseAreaLight>* lights;	// multiple instance may have different materials over the same mesh
	};
	
	SceneInstance(const uint id) : SceneNode(id) {}
	NodeType getType() const override { return NodeType::INSTANCE; }

	void setChild(SceneNode::SharedPtr c) { child = c; }
	SceneNode::SharedPtr getChild() { return child; }

	void setMaterial(int material) { data.material = material; }
	int getMaterial() const { return data.material; }

	void setTransform(const Matrixf<4, 4>& t) { transform = t; }
	Matrixf<4, 4> getTransform() const { return transform; }

	InstanceData data;
	Matrixf<4, 4> transform;
	SceneNode::SharedPtr child;
};

class SceneGroup : public SceneNode {
public:
	using SharedPtr = std::shared_ptr<SceneGroup>;
	
	SceneGroup(const uint id) : SceneNode(id) {}
	NodeType getType() const override { return NodeType::GROUP; }

	void addChild(SceneInstance::SharedPtr instance) { mChildren.push_back(instance); }
	SceneInstance::SharedPtr getChild(const size_t index) { return mChildren[index]; }
	size_t getNumChildren() const { return mChildren.size(); }
	
private:
	std::vector<SceneInstance::SharedPtr> mChildren;
};

KRR_NAMESPACE_END