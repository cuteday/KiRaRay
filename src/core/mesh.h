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
	vec3f* vertices{ nullptr };
	vec3i* indices{ nullptr };
	vec3f* normals{ nullptr };
	vec2f* texcoords{ nullptr };
	vec3f* tangents{ nullptr };
	vec3f* bitangents{ nullptr };
	uint materialId{ 0 };

	DiffuseAreaLight* lights{ nullptr };
};

class Mesh {
public:
	void toDevice() {
		mData.vertices = (vec3f*)vertices.data();
		mData.normals = (vec3f*)normals.data();
		mData.indices = (vec3i*)indices.data();
		mData.texcoords = (vec2f*)texcoords.data();
		mData.tangents = (vec3f*)tangents.data();
		mData.bitangents = (vec3f*)bitangents.data();
		mData.materialId = materialId;

		mData.lights = lights.data();
	}

	std::vector<Triangle> createTriangles(MeshData* mesh);

	inter::vector<vec3f> vertices;
	inter::vector<vec3f> normals;
	inter::vector<vec2f> texcoords;
	inter::vector<vec3i> indices;
	inter::vector<vec3f> tangents;
	inter::vector<vec3f> bitangents;

	inter::vector<Triangle> emissiveTriangles;
	inter::vector<DiffuseAreaLight> lights;

	uint materialId = 0;
	MeshData mData;
};

KRR_NAMESPACE_END