#pragma once

#include "bsdf.h"
#include "common.h"

#include "shared.h"
#include "raytracing.h"
#include "util/hash.h"

#include <optix_device.h>

KRR_NAMESPACE_BEGIN

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
	hitInfo.hitKind			 = optixGetHitKind();
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

KRR_DEVICE_FUNCTION void prepareShadingData(ShadingData &sd, const HitInfo &hitInfo,
											const rt::MaterialData &material) {
	// [NOTE] about local shading frame (tangent space, TBN, etc.)
	// The shading normal sd.frame.N and face normal sd.geoN is always points towards the outside of
	// the object, we can use this convention to determine whether an incident ray is coming from
	// outside of the object.

	const rt::InstanceData &instance = hitInfo.getInstance();
	const rt::MeshData &mesh		 = hitInfo.getMesh();
	Vector3f b						 = hitInfo.barycentric;
	Vector3i v						 = mesh.indices[hitInfo.primitiveId];
	
	sd.wo  = normalize(hitInfo.wo);

	Vector3f p0 = mesh.positions[v[0]], p1 = mesh.positions[v[1]],
			 p2 = mesh.positions[v[2]];
	
	sd.pos = b[0] * p0 + b[1] * p1  + b[2] * p2;

	Vector3f face_normal = normalize(cross(p1 - p0, p2 - p0));

	if (mesh.normals.size())
		sd.frame.N = normalize(b[0] * mesh.normals[v[0]] +
							   b[1] * mesh.normals[v[1]] +
							   b[2] * mesh.normals[v[2]]);
	else  // if the model does not contain normal...
		sd.frame.N = face_normal;

	if (mesh.tangents.size()) {
		sd.frame.T = normalize(b[0] * mesh.tangents[v[0]] + b[1] * mesh.tangents[v[1]] +
							   b[2] * mesh.tangents[v[2]]);
		// re-orthogonize the tangent space
		sd.frame.T = normalize(sd.frame.T - sd.frame.N * dot(sd.frame.N, sd.frame.T));
	} else sd.frame.T = getPerpendicular(sd.frame.N);
	sd.frame.B = normalize(cross(sd.frame.N, sd.frame.T));

	if (mesh.texcoords.size()) {
		Vector2f uv[3] = {mesh.texcoords[v[0]], mesh.texcoords[v[1]],
						  mesh.texcoords[v[2]]};
		sd.uv		   = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];
	}

	if (instance.lights.size())
		sd.light = &instance.lights[hitInfo.primitiveId];
	else sd.light = nullptr;

	const Material::MaterialParams &materialParams = material.mMaterialParams;

	sd.IoR					= materialParams.IoR;
	sd.bsdfType				= material.mBsdfType;
	sd.specularTransmission = materialParams.specularTransmission;

	const rt::TextureData &diffuseTexture = 
		material.mTextures[(uint) Material::TextureType::Diffuse];
	const rt::TextureData &specularTexture =
		material.mTextures[(uint) Material::TextureType::Specular];
	const rt::TextureData &normalTexture =
		material.mTextures[(uint) Material::TextureType::Normal];

	Color4f diff	   = sampleTexture(diffuseTexture, sd.uv, materialParams.diffuse);
	Color4f spec	   = sampleTexture(specularTexture, sd.uv, materialParams.specular);
	Color3f baseColor  = (Color3f) diff;

	if (normalTexture.isValid() && mesh.texcoords.size()) { // be cautious if we have TBN info
		Vector3f normal = sampleTexture(normalTexture, sd.uv, Color3f{ 0, 0, 1 });
		normal			= rgbToNormal(normal);

		sd.frame.N =
			normalize(sd.frame.T * normal[0] + sd.frame.B * normal[1] + sd.frame.N * normal[2]);
		sd.frame.T = normalize(sd.frame.T - sd.frame.N * dot(sd.frame.T, sd.frame.N));
		sd.frame.B = normalize(cross(sd.frame.N, sd.frame.T));
	}

	if (material.mShadingModel == Material::ShadingModel::MetallicRoughness) {
		// [SPECULAR] G - Roughness; B - Metallic
		sd.diffuse	 = lerp(baseColor, Color3f::Zero(), spec[2]);
		sd.specular	 = lerp(Color3f::Zero(), baseColor, spec[2]);
		sd.metallic	 = spec[2];
		sd.roughness = spec[1];
	} else if (material.mShadingModel == Material::ShadingModel::SpecularGlossiness) {
		// [SPECULAR] RGB - Specular Color; A - Glossiness
		sd.diffuse	 = baseColor;
		sd.specular	 = (Color3f) spec; // specular reflectance
		sd.roughness = 1.f - spec[3];	//
		sd.metallic	 = getMetallic(sd.diffuse, sd.specular);
	} else assert(false);
	
	// transform local interaction to world space 
	// [TODO: refactor this, maybe via an integrated SurfaceInteraction struct]
	sd.pos	   = hitInfo.instance->getTransform() * sd.pos;
	sd.frame.N = hitInfo.instance->getTransposedInverseTransform() * sd.frame.N;
	sd.frame.T = hitInfo.instance->getTransposedInverseTransform() * sd.frame.T;
	sd.frame.B = hitInfo.instance->getTransposedInverseTransform() * sd.frame.B;
}

KRR_NAMESPACE_END