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
	Vec3f* vertices{ nullptr };
	Vec3i* indices{ nullptr };
	Vec3f* normals{ nullptr };
	Vec2f* texcoords{ nullptr };
	Vec3f* tangents{ nullptr };
	Vec3f* bitangents{ nullptr };
	uint materialId{ 0 };

	DiffuseAreaLight* lights{ nullptr };
};

class Mesh {
public:
	void toDevice() {
		mData.vertices = (Vec3f*)vertices.data();
		mData.normals = (Vec3f*)normals.data();
		mData.indices = (Vec3i*)indices.data();
		mData.texcoords = (Vec2f*)texcoords.data();
		mData.tangents = (Vec3f*)tangents.data();
		mData.bitangents = (Vec3f*)bitangents.data();
		mData.materialId = materialId;

		mData.lights = lights.data();
	}

	std::vector<Triangle> createTriangles(MeshData* mesh);

	inter::vector<Vec3f> vertices;
	inter::vector<Vec3f> normals;
	inter::vector<Vec2f> texcoords;
	inter::vector<Vec3i> indices;
	inter::vector<Vec3f> tangents;
	inter::vector<Vec3f> bitangents;

	inter::vector<Triangle> emissiveTriangles;
	inter::vector<DiffuseAreaLight> lights;

	uint materialId = 0;
	MeshData mData;
};

KRR_NAMESPACE_END