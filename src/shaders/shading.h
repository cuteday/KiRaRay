#pragma once

#include "shared.h"
#include "common.h"

KRR_NAMESPACE_BEGIN

	KRR_DEVICE_FUNCTION vec4f sampleTexture(Texture& texture, vec2f uv, vec4f fallback) {
		if (texture.isValid()) {
			cudaTextureObject_t cudaTexture = texture.getCudaTexture();
			return tex2D<float4>(cudaTexture, uv.x, uv.y);
		}
		return fallback;
	}

	KRR_DEVICE_FUNCTION void prepareShadingData(ShadingData& sd, const HitInfo& hitInfo, Material& material) {
		vec3f bc = hitInfo.barycentric;
		MeshData& mesh = *hitInfo.mesh;
		vec3i triangle = mesh.indices[hitInfo.primitiveId];

		sd.wi = hitInfo.wi;

		sd.pos = bc.x * mesh.vertices[triangle.x] +
			bc.y * mesh.vertices[triangle.y] +
			bc.z * mesh.vertices[triangle.z];

		sd.geoN = normalize(cross(mesh.vertices[triangle.y] - mesh.vertices[triangle.x],
			mesh.vertices[triangle.z] - mesh.vertices[triangle.x]));

		sd.N = normalize(
			bc.x * mesh.normals[triangle.x] +
			bc.y * mesh.normals[triangle.y] +
			bc.z * mesh.normals[triangle.z]);

		sd.frontFacing = dot(sd.wi, sd.N) > 0.f;
		if (!sd.frontFacing) 
			sd.N = -sd.N;

		if (mesh.tangents != nullptr && mesh.bitangents != nullptr) {
			sd.T = normalize(
				bc.x * mesh.tangents[triangle.x] +
				bc.y * mesh.tangents[triangle.y] +
				bc.z * mesh.tangents[triangle.z]);
			sd.B = normalize(
				bc.x * mesh.bitangents[triangle.x] +
				bc.y * mesh.bitangents[triangle.y] +
				bc.z * mesh.bitangents[triangle.z]);
		}
		else {
			// generate a fake tbn frame for now...
			sd.T = getPerpendicular(sd.N);
			sd.B = normalize(cross(sd.N, sd.T));
		}

		if (mesh.texcoords) {
			sd.uv = (
				bc.x * mesh.texcoords[triangle.x] +
				bc.y * mesh.texcoords[triangle.y] +
				bc.z * mesh.texcoords[triangle.z]);
		}

		Material::MaterialParams& materialParams = material.mMaterialParams;
		
		sd.IoR = materialParams.IoR;
		sd.shadingModel = material.mShadingModel;

		Texture& diffuseTexture = material.mTextures[(uint)Material::TextureType::Diffuse];
		Texture& specularTexture = material.mTextures[(uint)Material::TextureType::Specular];

		sd.diffuse = (vec3f)sampleTexture(diffuseTexture, sd.uv, materialParams.diffuse);
	}


KRR_NAMESPACE_END