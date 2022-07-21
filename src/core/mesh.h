#pragma once

#include "common.h"
#include "math/math.h"
#include "device/memory.h"
#include "device/buffer.h"
//#include "light.h"

KRR_NAMESPACE_BEGIN

class Triangle;
class DiffuseAreaLight;

struct MeshData {
	Vector3f* vertices{ nullptr };
	Vector3i* indices{ nullptr };
	Vector3f* normals{ nullptr };
	Vector2f* texcoords{ nullptr };
	Vector3f* tangents{ nullptr };
	Vector3f* bitangents{ nullptr };
	uint materialId{ 0 };

	DiffuseAreaLight* lights{ nullptr };
};

class Mesh {
public:
	void toDevice() {
		mData.vertices = (Vector3f*)vertices.data();
		mData.normals = (Vector3f*)normals.data();
		mData.indices = (Vector3i*)indices.data();
		mData.texcoords = (Vector2f*)texcoords.data();
		mData.tangents = (Vector3f*)tangents.data();
		mData.bitangents = (Vector3f*)bitangents.data();
		mData.materialId = materialId;

		mData.lights = lights.data();
	}

	std::vector<Triangle> createTriangles(MeshData* mesh);

	inter::vector<Vector3f> vertices;
	inter::vector<Vector3f> normals;
	inter::vector<Vector2f> texcoords;
	inter::vector<Vector3i> indices;
	inter::vector<Vector3f> tangents;
	inter::vector<Vector3f> bitangents;

	inter::vector<Triangle> emissiveTriangles;
	inter::vector<DiffuseAreaLight> lights;

	uint materialId = 0;
	MeshData mData;
};

KRR_NAMESPACE_END