#pragma once

#include "shared.h"
#include "common.h"
#include "bsdf.h"

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

		sd.wo = hitInfo.wo;

		sd.pos = bc.x * mesh.vertices[triangle.x] +
			bc.y * mesh.vertices[triangle.y] +
			bc.z * mesh.vertices[triangle.z];

		sd.geoN = normalize(cross(mesh.vertices[triangle.y] - mesh.vertices[triangle.x],
			mesh.vertices[triangle.z] - mesh.vertices[triangle.x]));

		sd.N = normalize(
			bc.x * mesh.normals[triangle.x] +
			bc.y * mesh.normals[triangle.y] +
			bc.z * mesh.normals[triangle.z]);

		sd.frontFacing = dot(sd.wo, sd.N) > 0.f;
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

		vec4f diff = sampleTexture(diffuseTexture, sd.uv, materialParams.diffuse);
		vec4f spec = sampleTexture(specularTexture, sd.uv, materialParams.specular);

		if (sd.shadingModel == Material::ShadingModel::Diffuse) {
			sd.diffuse = (vec3f)diff;
		}
		else if (sd.shadingModel == Material::ShadingModel::MetallicRoughness) {
			// this is the default except for OBJ or when user specified 
			// G - Roughness; B - Metallic
			vec3f baseColor = (vec3f)diff;
			sd.diffuse = lerp(sd.diffuse, vec3f(0), spec.b);

			// Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
			// UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
			float f = (sd.IoR - 1.f) / (sd.IoR + 1.f);
			float F0 = f * f;

			sd.specular = lerp(vec3f(F0), baseColor, spec.b);
			sd.roughness = spec.g;
			sd.metallic = spec.b;
		}
		else if (sd.shadingModel == Material::ShadingModel::SpecularGlossiness) {
		
		}
	}


KRR_NAMESPACE_END