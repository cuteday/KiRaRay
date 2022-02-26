#pragma once

#include "common.h"
#include "math/math.h"
#include "device/buffer.h"
#include "shape.h"
#include "light.h"

KRR_NAMESPACE_BEGIN

struct MeshData {
	vec3f* vertices = nullptr;
	vec3i* indices = nullptr;
	vec3f* normals = nullptr;
	vec2f* texcoords = nullptr;
	vec3f* tangents = nullptr;
	vec3f* bitangents = nullptr;
	uint materialId = 0;
	Light* lights = nullptr;
};

class Mesh {
public:
	struct DeviceMemory {
		CUDABuffer vertices;
		CUDABuffer normals;
		CUDABuffer texcoords;
		CUDABuffer indices;
		CUDABuffer tangents;
		CUDABuffer bitangents;
		CUDABuffer material;

		TypedBuffer<Triangle> emissiveTriangles;
		TypedBuffer<DiffuseAreaLight> areaLights;
	};

	void toDevice() {
		mDeviceMemory.vertices.alloc_and_copy_from_host(vertices);
		mDeviceMemory.normals.alloc_and_copy_from_host(normals);
		mDeviceMemory.texcoords.alloc_and_copy_from_host(texcoords);
		mDeviceMemory.indices.alloc_and_copy_from_host(indices);
		mDeviceMemory.tangents.alloc_and_copy_from_host(tangents);
		mDeviceMemory.bitangents.alloc_and_copy_from_host(bitangents);

		mMeshData.vertices = (vec3f*)mDeviceMemory.vertices.data();
		mMeshData.normals = (vec3f*)mDeviceMemory.normals.data();
		mMeshData.indices = (vec3i*)mDeviceMemory.indices.data();
		mMeshData.texcoords = (vec2f*)mDeviceMemory.texcoords.data();
		mMeshData.tangents = (vec3f*)mDeviceMemory.tangents.data();
		mMeshData.bitangents = (vec3f*)mDeviceMemory.bitangents.data();
		mMeshData.materialId = mMaterialId;
	
		// move diffuse emissives to device memory
		uint lightNum = emissiveTriangles.size();
		mDeviceMemory.emissiveTriangles.alloc_and_copy_from_host(emissiveTriangles);
		mDeviceMemory.areaLights.resize(lightNum);
		for (int i = 0; i < lightNum; i++) {
			mDeviceMemory.areaLights[i];

		}
	}

	static std::vector<Triangle> createTriangles(uint meshId, Mesh& mesh) {
		uint nTriangles = mesh.indices.size();
		std::vector<Triangle> tris(nTriangles);
		for (uint i = 0; i < nTriangles; i++) {
			tris[i] = Triangle(meshId, i);
		}
		return tris;
	}

	std::vector<vec3f> vertices;
	std::vector<vec3f> normals;
	std::vector<vec2f> texcoords;
	std::vector<vec3i> indices;
	std::vector<vec3f> tangents;
	std::vector<vec3f> bitangents;

	std::vector<Triangle> emissiveTriangles;
	std::vector<DiffuseAreaLight> areaLights;

	uint mMaterialId = 0;
	MeshData mMeshData;
	DeviceMemory mDeviceMemory;
};

KRR_NAMESPACE_END