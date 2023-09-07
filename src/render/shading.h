#pragma once

#include "bsdf.h"
#include "common.h"

#include "shared.h"
#include "raytracing.h"
#include "util/hash.h"

#include <optix_device.h>

KRR_NAMESPACE_BEGIN
using namespace rt;

namespace {
KRR_DEVICE_FUNCTION float getMetallic(Color diffuse, Color spec) {
	// This is based on the way that UE4 and Substance Painter 2 converts base+metallic+specular
	// level to diffuse/spec colors We don't have the specular level information, so the assumption
	// is that it is equal to 0.5 (based on the UE4 documentation)
	float d = luminance(diffuse);
	float s = luminance(spec);
	if (s == 0)
		return 0;
	float b	   = s + d - 0.08f;
	float c	   = 0.04f - s;
	float root = sqrt(b * b - 0.16f * c);
	float m	   = (root - b) * 12.5f;
	return max(0.f, m);
}

KRR_DEVICE_FUNCTION Color rgbToNormal(Color rgb) { return 2 * rgb - Color::Ones(); }

} 

template <typename T>
KRR_DEVICE_FUNCTION T sampleTexture(const rt::TextureData &texture, Vector2f uv,
									T fallback = {}) {
	if (texture.isValid()) {
		if (cudaTextureObject_t cudaTexture = texture.getCudaTexture()) 
			return tex2D<float4>(cudaTexture, uv[0], uv[1]);
		else return texture.getConstant();
	}
	return fallback;
}

KRR_DEVICE_FUNCTION HitInfo getHitInfo() {
	HitInfo hitInfo			 = {};
	HitgroupSBTData *hitData = (HitgroupSBTData *) optixGetSbtDataPointer();
	Vector2f barycentric	 = optixGetTriangleBarycentrics();
	hitInfo.primitiveId		 = optixGetPrimitiveIndex();
	hitInfo.instance		 = hitData->instance;
	hitInfo.wo				 = -normalize((Vector3f) optixGetWorldRayDirection());
	hitInfo.barycentric = { 1 - barycentric[0] - barycentric[1], barycentric[0], barycentric[1] };
	return hitInfo;
}

KRR_DEVICE_FUNCTION bool alphaKilled() {
	HitInfo hitInfo			   = getHitInfo();
	const rt::MaterialData &material = hitInfo.getMaterial();
	const rt::TextureData &opaticyTexture =
		material.mTextures[(uint) Material::TextureType::Transmission];
	if (!opaticyTexture.isValid()) return false;

	Vector3f b				 = hitInfo.barycentric;
	const rt::MeshData &mesh = hitInfo.getMesh();
	Vector3i v				 = mesh.indices[hitInfo.primitiveId];
	Vector2f uv{};
	if (mesh.texcoords.size())
		uv = b[0] * mesh.texcoords[v[0]] + b[1] * mesh.texcoords[v[1]] +
					  b[2] * mesh.texcoords[v[2]];

	Vector3f opacity = sampleTexture(opaticyTexture, uv, Color3f(1));
	float alpha		 = 1 - luminance(opacity);
	if (alpha >= 1) return false;
	if (alpha <= 0) return true;
	float3 o = optixGetWorldRayOrigin();
	float3 d = optixGetWorldRayDirection();
	float u	 = HashFloat(o, d);
	return u > alpha;
}

KRR_DEVICE_FUNCTION void prepareSurfaceInteraction(SurfaceInteraction &intr, const HitInfo &hitInfo) {
	// [NOTE] about local shading frame (tangent space, TBN, etc.)
	// The shading normal intr.n and face normal is always points towards the outside of
	// the object, we can use this convention to determine whether an incident ray is coming from
	// outside of the object.

	const rt::InstanceData &instance = hitInfo.getInstance();
	const rt::MeshData &mesh		 = hitInfo.getMesh();
	const rt::MaterialData &material = instance.getMaterial();
	Vector3f b						 = hitInfo.barycentric;
	Vector3i v						 = mesh.indices[hitInfo.primitiveId];
	
	intr.wo  = normalize(hitInfo.wo);

	Vector3f p0 = mesh.positions[v[0]], p1 = mesh.positions[v[1]],
			 p2 = mesh.positions[v[2]];
	
	intr.p = b[0] * p0 + b[1] * p1  + b[2] * p2;

	if (mesh.normals.size())
		intr.n = normalize(b[0] * mesh.normals[v[0]] +
							   b[1] * mesh.normals[v[1]] +
							   b[2] * mesh.normals[v[2]]);
	else  // if the model does not contain normal...
		intr.n = normalize(cross(p1 - p0, p2 - p0));

	if (mesh.tangents.size()) {
		intr.tangent = normalize(b[0] * mesh.tangents[v[0]] + b[1] * mesh.tangents[v[1]] +
							   b[2] * mesh.tangents[v[2]]);
		// re-orthogonize the tangent space
		intr.tangent = normalize(intr.tangent - intr.n * dot(intr.n, intr.tangent));
	} else intr.tangent = getPerpendicular(intr.n);
	intr.bitangent = normalize(cross(intr.n, intr.tangent));

	if (mesh.texcoords.size()) {
		Vector2f uv[3] = {mesh.texcoords[v[0]], mesh.texcoords[v[1]],
						  mesh.texcoords[v[2]]};
		intr.uv		   = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];
	}

	if (instance.lights.size())
		intr.light = &instance.lights[hitInfo.primitiveId];
	else intr.light = nullptr;

	const Material::MaterialParams &materialParams = material.mMaterialParams;

	intr.sd.IoR					 = materialParams.IoR;
	intr.sd.bsdfType			 = material.mBsdfType;
	intr.sd.specularTransmission = materialParams.specularTransmission;

	const rt::TextureData &diffuseTexture = 
		material.mTextures[(uint) Material::TextureType::Diffuse];
	const rt::TextureData &specularTexture =
		material.mTextures[(uint) Material::TextureType::Specular];
	const rt::TextureData &normalTexture =
		material.mTextures[(uint) Material::TextureType::Normal];

	Color4f diff	   = sampleTexture(diffuseTexture, intr.uv, materialParams.diffuse);
	Color4f spec	   = sampleTexture(specularTexture, intr.uv, materialParams.specular);
	Color3f baseColor  = (Color3f) diff;

	if (normalTexture.isValid() && mesh.texcoords.size()) { // be cautious if we have TBN info
		Vector3f normal = sampleTexture(normalTexture, intr.uv, Color3f{ 0, 0, 1 });
		normal			= rgbToNormal(normal);

		intr.n =
			normalize(intr.tangent * normal[0] + intr.bitangent * normal[1] + intr.n * normal[2]);
		intr.tangent = normalize(intr.tangent - intr.n * dot(intr.tangent, intr.n));
		intr.bitangent = normalize(cross(intr.n, intr.tangent));
	}

	if (material.mShadingModel == Material::ShadingModel::MetallicRoughness) {
		// [SPECULAR] G - Roughness; B - Metallic
		intr.sd.diffuse	  = lerp(baseColor, Color3f::Zero(), spec[2]);
		intr.sd.specular  = lerp(Color3f::Zero(), baseColor, spec[2]);
		intr.sd.metallic  = spec[2];
		intr.sd.roughness = spec[1];
	} else if (material.mShadingModel == Material::ShadingModel::SpecularGlossiness) {
		// [SPECULAR] RGB - Specular Color; A - Glossiness
		intr.sd.diffuse	  = baseColor;
		intr.sd.specular  = (Color3f) spec; // specular reflectance
		intr.sd.roughness = 1.f - spec[3];	//
		intr.sd.metallic  = getMetallic(intr.sd.diffuse, intr.sd.specular);
	} else assert(false);
	
	// transform local interaction to world space 
	// [TODO: refactor this, maybe via an integrated SurfaceInteraction struct]
	intr.p		   = hitInfo.instance->getTransform() * intr.p;
	intr.n		   = hitInfo.instance->getTransposedInverseTransform() * intr.n;
	intr.tangent   = hitInfo.instance->getTransposedInverseTransform() * intr.tangent;
	intr.bitangent = hitInfo.instance->getTransposedInverseTransform() * intr.bitangent;
}

KRR_NAMESPACE_END