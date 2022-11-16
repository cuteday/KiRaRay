#pragma once

#include "shared.h"
#include "common.h"
#include "math/math.h"
#include "util/hash.h"
#include "bsdf.h"

#include <optix_device.h>

KRR_NAMESPACE_BEGIN

namespace {
KRR_DEVICE_FUNCTION float getMetallic(Color diffuse, Color spec)
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

	KRR_DEVICE_FUNCTION Color rgbToNormal(Color rgb) { 
		return 2 * rgb - Color::Ones();
	}

}

template <typename T>
KRR_DEVICE_FUNCTION T sampleTexture(Texture &texture, Vector2f uv, T fallback) {
	cudaTextureObject_t cudaTexture = texture.getCudaTexture();
	if (cudaTexture) {
		return tex2D<float4>(cudaTexture, uv[0], uv[1]);
	}
	return fallback;
}

KRR_DEVICE_FUNCTION HitInfo getHitInfo() {
	HitInfo hitInfo = {};
	HitgroupSBTData* hitData = (HitgroupSBTData*)optixGetSbtDataPointer();
	Vector2f barycentric = optixGetTriangleBarycentrics();
	hitInfo.primitiveId = optixGetPrimitiveIndex();
	hitInfo.mesh = hitData->mesh;
	hitInfo.wo = -normalize((Vector3f)optixGetWorldRayDirection());
	hitInfo.hitKind = optixGetHitKind();
	hitInfo.barycentric = { 1 - barycentric[0] - barycentric[1], barycentric[0], barycentric[1] };
	return hitInfo;
}

KRR_DEVICE_FUNCTION bool alphaKilled(inter::vector<Material>* materials) {
	HitInfo hitInfo			= getHitInfo();
	Material &material		= (*materials)[hitInfo.mesh->materialId];
	Texture &opaticyTexture = material.mTextures[(uint) Material::TextureType::Transmission];
	if (!opaticyTexture.isValid())
		return false;

	Vector3f b				  = hitInfo.barycentric;
	const MeshData &mesh	  = *hitInfo.mesh;
	Vector3i v				  = mesh.indices[hitInfo.primitiveId];
	const VertexAttribute &v0 = mesh.vertices[v[0]], v1 = mesh.vertices[v[1]],
						  v2 = mesh.vertices[v[2]];
	Vector2f uv				 = b[0] * v0.texcoord + b[1] * v1.texcoord + b[2] * v2.texcoord;

	Vector3f opacity = sampleTexture(opaticyTexture, uv, Vector3f(1));
	float alpha		 = 1 - luminance(opacity);
	if (alpha >= 1)
		return false;
	if (alpha <= 0)
		return true;
	float3 o = optixGetWorldRayOrigin();
	float3 d = optixGetWorldRayDirection();
	float u	 = HashFloat(o, d);
	return u > alpha;
}

KRR_DEVICE_FUNCTION void prepareShadingData(ShadingData &sd, const HitInfo &hitInfo,
											Material &material) {
	// [NOTE] about local shading frame (tangent space, TBN, etc.)
	// The shading normal sd.frame.N and face normal sd.geoN is always points towards the outside of
	// the object, we can use this convention to determine whether an incident ray is coming from
	// outside of the object.

	Vector3f b		   = hitInfo.barycentric;
	MeshData &mesh	   = *hitInfo.mesh;
	Vector3i v		   = mesh.indices[hitInfo.primitiveId];
	VertexAttribute v0 = mesh.vertices[v[0]], v1 = mesh.vertices[v[1]], v2 = mesh.vertices[v[2]];

	sd.wo = normalize(hitInfo.wo);
	sd.pos = b[0] * v0.vertex + b[1] * v1.vertex + b[2] * v2.vertex;

	Vector3f face_normal = normalize(cross(v1.vertex - v0.vertex, v2.vertex - v0.vertex));

	if (any(v0.normal)) {	
		sd.frame.N = normalize(b[0] * v0.normal + b[1] * v1.normal + b[2] * v2.normal);
		sd.frame.T = normalize(b[0] * v0.tangent + b[1] * v1.tangent + b[2] * v2.tangent);
	} else {			// if the model does not contain normal...
		sd.frame.N = face_normal;
		sd.frame.T = getPerpendicular(face_normal);
	}
	
	// re-orthogonize the tangent space 
	// since tbn may become not orthogonal after the interpolation process.
	// or they are not orthogonal at the beginning (when we import the models)
	sd.frame.T = normalize(sd.frame.T - sd.frame.N * dot(sd.frame.N, sd.frame.T));
	sd.frame.B = normalize(cross(sd.frame.N, sd.frame.T));

	Vector2f uv[3] = { v0.texcoord, v1.texcoord, v2.texcoord };
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
	Texture& normalTexture = material.mTextures[(uint)Material::TextureType::Normal];

	Vector4f diff = sampleTexture(diffuseTexture, sd.uv, materialParams.diffuse);
	Vector4f spec = sampleTexture(specularTexture, sd.uv, materialParams.specular);
	Vector3f baseColor = (Vector3f)diff;

	if (normalTexture.isValid()) {	// be cautious if we have the tangent space TBN
		Vector3f normal = sampleTexture(normalTexture, sd.uv, Vector3f{0, 0, 1});
		normal = rgbToNormal(normal);

		sd.frame.N = normalize(sd.frame.T * normal[0] + sd.frame.B * normal[1] + sd.frame.N * normal[2]);
		sd.frame.T = normalize(sd.frame.T - sd.frame.N * dot(sd.frame.T, sd.frame.N));
		sd.frame.B = normalize(cross(sd.frame.N, sd.frame.T));
	}

	if (material.mShadingModel == Material::ShadingModel::MetallicRoughness) {
		// [SPECULAR] G - Roughness; B - Metallic
		sd.diffuse	 = lerp(baseColor, Vector3f::Zero(), spec[2]);
		sd.specular	 = lerp(Vector3f::Zero(), baseColor, spec[2]);
		sd.metallic	 = spec[2];
		sd.roughness = spec[1];
	}
	else if (material.mShadingModel == Material::ShadingModel::SpecularGlossiness) {
		// [SPECULAR] RGB - Specular Color; A - Glossiness
		sd.diffuse	 = baseColor;
		sd.specular	 = (Vector3f) spec; 	// specular reflectance
		sd.roughness = 1.f - spec[3];		// 
		sd.metallic	 = getMetallic(sd.diffuse, sd.specular);
	}
	else {
		assert(false);
	}
}


KRR_NAMESPACE_END