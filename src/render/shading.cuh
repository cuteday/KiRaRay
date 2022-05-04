#pragma once

#include "shared.h"
#include "common.h"
#include "bsdf.h"

#include <optix_device.h>

KRR_NAMESPACE_BEGIN

namespace {
	KRR_DEVICE_FUNCTION float getMetallic(vec3f diffuse, vec3f spec)
	{
		// This is based on the way that UE4 and Substance Painter 2 converts base+metallic+specular level to diffuse/spec colors
		// We don't have the specular level information, so the assumption is that it is equal to 0.5 (based on the UE4 documentation)
		float d = luminance(diffuse);
		float s = luminance(spec);
		if (s == 0) return 0;
		float b = s + d - 0.08;
		float c = 0.04 - s;
		float root = sqrt(b * b - 0.16 * c);
		float m = (root - b) * 12.5;
		return max(0.f, m);
	}
}

template <typename T>
KRR_DEVICE_FUNCTION T sampleTexture(Texture& texture, vec2f uv, T fallback) {
	cudaTextureObject_t cudaTexture = texture.getCudaTexture();
	if (cudaTexture) {
		return tex2D<float4>(cudaTexture, uv.x, uv.y);
	}
	return fallback;
}

KRR_DEVICE_FUNCTION HitInfo getHitInfo() {
	HitInfo hitInfo = {};
	HitgroupSBTData* hitData = (HitgroupSBTData*)optixGetSbtDataPointer();
	vec2f barycentric = (vec2f)optixGetTriangleBarycentrics();
	hitInfo.primitiveId = optixGetPrimitiveIndex();
	hitInfo.mesh = hitData->mesh;
	hitInfo.wo = -normalize((vec3f)optixGetWorldRayDirection());
	hitInfo.hitKind = optixGetHitKind();
	hitInfo.barycentric = { 1 - barycentric.x - barycentric.y, barycentric.x, barycentric.y };
	return hitInfo;
}

KRR_DEVICE_FUNCTION void prepareShadingData(ShadingData& sd, const HitInfo& hitInfo, Material& material) {
	// [NOTE] about local shading frame (tangent space, TBN, etc.)
	// The shading normal sd.N and face normal sd.geoN is always points towards the outside of the object,
	// we can use this convention to determine whether an incident ray is coming from outside of the object.
	
	vec3f b = hitInfo.barycentric;
	MeshData& mesh = *hitInfo.mesh;
	vec3i v = mesh.indices[hitInfo.primitiveId];

	sd.wo = normalize(hitInfo.wo);

	sd.pos = b[0] * mesh.vertices[v[0]] +
		b[1] * mesh.vertices[v[1]] +
		b[2] * mesh.vertices[v[2]];

	sd.geoN = normalize(cross(mesh.vertices[v[1]] - mesh.vertices[v[0]],
		mesh.vertices[v[2]] - mesh.vertices[v[0]]));

	sd.N = normalize(
		b[0] * mesh.normals[v[0]] +
		b[1] * mesh.normals[v[1]] +
		b[2] * mesh.normals[v[2]]);

	sd.frontFacing = dot(sd.wo, sd.N) > 0.f;
//	if (!sd.frontFacing) sd.N *= -1;

//	if (mesh.tangents.data() && mesh.bitangents.data()) {
	if (mesh.tangents && mesh.bitangents) {
		sd.T = normalize(
			b[0] * mesh.tangents[v[0]] +
			b[1] * mesh.tangents[v[1]] +
			b[2] * mesh.tangents[v[2]]);
		//sd.B = normalize(
		//	b[0] * mesh.bitangents[v[0]] +
		//	b[1] * mesh.bitangents[v[1]] +
		//	b[2] * mesh.bitangents[v[2]]);
		
		// re-orthogonize the tangent space 
		// since tbn may become not orthogonal after the interpolation process.
		// or they are not orthogonal at the beginning (when we import the models)
		sd.T = normalize(sd.T - sd.N * dot(sd.N, sd.T));
		sd.B = normalize(cross(sd.N, sd.T));
	}
	else {
		// generate a fake tbn frame for now...
		sd.T = getPerpendicular(sd.N);
		sd.B = normalize(cross(sd.N, sd.T));
	}

	vec2f uv[3];
	if (mesh.texcoords) {
		uv[0] = mesh.texcoords[v[0]],
			uv[1] = mesh.texcoords[v[1]],
			uv[2] = mesh.texcoords[v[2]];
	}
	else {
		uv[0] = { 0,0 }, uv[1] = { 1,0 }, uv[1] = { 1,1 };
	}
	sd.uv = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];

	if (mesh.lights) {
		sd.light = &mesh.lights[hitInfo.primitiveId];
	}

	Material::MaterialParams& materialParams = material.mMaterialParams;
	
	sd.IoR = materialParams.IoR;
	sd.bsdfType = material.mBsdfType;
	sd.diffuseTransmission = materialParams.diffuseTransmission;
	sd.specularTransmission = materialParams.specularTransmission;

	Texture& diffuseTexture = material.mTextures[(uint)Material::TextureType::Diffuse];
	Texture& specularTexture = material.mTextures[(uint)Material::TextureType::Specular];

	vec4f diff = sampleTexture(diffuseTexture, sd.uv, materialParams.diffuse);
	vec4f spec = sampleTexture(specularTexture, sd.uv, materialParams.specular);

	vec3f baseColor = (vec3f)diff;

	if (material.mShadingModel == Material::ShadingModel::MetallicRoughness) {
		// this is the default except for OBJ or when user specified 
		// G - Roughness; B - Metallic
		sd.diffuse = lerp(baseColor, vec3f(0), spec.b);

		// Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
		// UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
		float f = (sd.IoR - 1.f) / (sd.IoR + 1.f);
		float F0 = f * f;

		sd.specular = lerp(vec3f(F0), baseColor, spec.b);
		sd.metallic = spec.b;
		sd.roughness = spec.g;
	}
	else if (material.mShadingModel == Material::ShadingModel::SpecularGlossiness) {
		sd.diffuse = baseColor;
		sd.specular = (vec3f)spec;			// specular reflectance
		sd.roughness = 1 - spec.a;			// 
		sd.metallic = getMetallic(sd.diffuse, sd.specular);
	}
	else {
		assert(false);
	}
}


KRR_NAMESPACE_END