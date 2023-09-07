#pragma pack_matrix(column_major)

/* constants */
static const int LightType_None = 0;
static const int LightType_Point = 1;
static const int LightType_Directional = 2;
static const int LightType_Infinite = 3;
static const float M_PI = 3.14159265358979323846;

/* utility functions */
float square(float x) { return x * x; }
float2 square(float2 x) { return x * x; }
float3 square(float3 x) { return x * x; }
float4 square(float4 x) { return x * x; }

float3 slerp(float3 a, float3 b, float angle, float t) {
    t = saturate(t);
    float sin1 = sin(angle * t);
    float sin2 = sin(angle * (1 - t));
    float ta = sin1 / (sin1 + sin2);
    float3 result = lerp(a, b, ta);
    return normalize(result);
}

struct Vertex{
	float3 position : POSITION;
	float2 uv : TEXCOORD;
	centroid float3 normal : NORMAL;
	centroid float3 tangent : TANGENT;
};

struct ViewConstants {
	float4x4 worldToView;
	float4x4 viewToClip;
	float4x4 worldToClip;
    float3 cameraPosition;
    uint padding;
};

struct LightConstants {
	uint numLights;
    float3 ambientBottom;
    float3 ambientTop;
	uint padding;
};

struct RenderConstants {
	/* push constants */
	uint instanceID;
};

struct MaterialConstants {
	float4 baseColor;
	float4 specularColor;

	float IoR;
	float opacity;
	int metalRough;
	int flags;

	int baseTextureIndex; 
	int specularTextureIndex;
	int normalTextureIndex;
	int emissiveTextureIndex;
};

struct LightData {
    float3 direction;
    int type;

    float3 position;
    int texture; 

    float3 color;
    float scale;
};

struct MeshData {
	uint numIndices;
	uint numVertices;
	int indexBufferIndex;
	int vertexBufferIndex;

	uint positionOffset;
	uint normalOffset;
	uint texCoordOffset;
	uint tangentOffset;

	uint indexOffset;
	uint materialIndex;
	int2 padding;
};

struct InstanceData{
    uint meshIndex;
    int3 padding;
	
    float4x4 transform;
};

struct MaterialSample {
    float3 diffuse;
    float3 specular;
    float3 normal;
    float3 emissive;
	float roughness;
	float metallic;
	float transmission;
    int padding;
};

ConstantBuffer<ViewConstants> g_ViewConstants : register(b0);
ConstantBuffer<LightConstants> g_LightConstants : register(b1);
[[vk::push_constant]] ConstantBuffer<RenderConstants> g_RenderConstants : register(b2);

StructuredBuffer<MeshData> t_MeshData : register(t0);
StructuredBuffer<InstanceData> t_InstanceData : register(t1);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t2);
StructuredBuffer<LightData> t_LightData : register(t3);
SamplerState s_MaterialSampler : register(s0);
// the above bindings are implicitly assigned to register space 0.
// the bindless buffer arrays below actually bind to a register range.
[[vk::binding(0, 1)]] ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);	// register space, check it out later
[[vk::binding(1, 1)]] Texture2D t_BindlessTextures[] : register(t0, space2);

float getMetallic(float3 diffuse, float3 specular) {
	float d = dot(diffuse, float3(0.299, 0.587, 0.114));
	float s = dot(specular, float3(0.299, 0.587, 0.114));
	if (s == 0) return 0;
	float b = s + d - 0.08f;
	float c = 0.04f - s;
	float root = sqrt(b * b - 0.16f * c);
	float m = (root - b) * 12.5f;
	return max(0.f, m);
}

MaterialSample EvaluateSceneMaterial(float2 uv, float3 normal, float3 tangent, MaterialConstants material) {
	MaterialSample result;
	
	float4 baseColor = material.baseColor;
	float4 specularColor = material.specularColor;
	
	if (material.baseTextureIndex >= 0) 
		baseColor = t_BindlessTextures[material.baseTextureIndex].Sample(s_MaterialSampler, uv);
	if (material.specularTextureIndex >= 0)
		specularColor = t_BindlessTextures[material.specularTextureIndex].Sample(s_MaterialSampler, uv);
	result.normal = normal;
	if (material.normalTextureIndex >= 0){
		float3 normal = t_BindlessTextures[material.normalTextureIndex].Sample(s_MaterialSampler, uv);
		normal = 2 * normal - 1;	// rgb to normal
		result.normal = normal;
	}
	result.emissive = 0;
	if (material.emissiveTextureIndex >= 0)
		result.emissive = t_BindlessTextures[material.emissiveTextureIndex].Sample(s_MaterialSampler, uv);
	
	if (material.metalRough) {
		result.diffuse  = lerp(baseColor.rgb, 0, specularColor[2]);
		result.specular = lerp(0, baseColor, specularColor[2]);
		result.metallic = specularColor[2];
		result.roughness = specularColor[1];
	} else {
		result.diffuse = baseColor.rgb;
		result.specular = specularColor.rgb;
		result.roughness = 1 - specularColor[3];
		result.metallic = getMetallic(result.diffuse, result.specular);
	}
	return result;
}

/* shading and brdf routines. */

float Lambert(float3 normal, float3 lightIncident) {
	return max(0, dot(normal, lightIncident)) / M_PI;
}

float3 Schlick_Fresnel(float3 F0, float VdotH) {
	return F0 + (1 - F0) * pow(max(1 - VdotH, 0), 5);
}
// light incident and view incident are both within the upper hemishpere.
float3 GGX_AnalyticalLights(float3 lightIncident, float3 viewIncident, float3 normal, float roughness, float3 specularF0, float halfAngularSize) {
	float3 N = normal;
	float3 V = viewIncident;
	float3 L = lightIncident;
	float3 R = reflect(viewIncident, N);

    // Correction of light vector L for spherical / directional area lights.
    // Inspired by "Real Shading in Unreal Engine 4" by B. Karis, 
    // re-formulated to work in spherical coordinates instead of world space.
	float AngleLR = acos(clamp(dot(R, L), -1, 1));

	float3 CorrectedL = (AngleLR > 0) ? slerp(L, R, AngleLR, saturate(halfAngularSize / AngleLR)) : L;
	float3 H = normalize(CorrectedL + V);

	float NdotH = saturate(dot(N, H));
	float NdotL = saturate(dot(N, CorrectedL));
	float NdotV = saturate(dot(N, V));
	float VdotH = saturate(dot(V, H));

	float Alpha = max(0.01, square(roughness));

    // Normalization for the widening of D, see the paper referenced above.
	float CorrectedAlpha = saturate(Alpha + 0.5 * tan(halfAngularSize));
	float SphereNormalization = square(Alpha / CorrectedAlpha);

    // GGX / Trowbridge-Reitz NDF with normalization for sphere lights
	float D = square(Alpha) / (M_PI * square(square(NdotH) * (square(Alpha) - 1) + 1)) * SphereNormalization;

    // Schlick model for geometric attenuation
    // The (NdotL * NdotV) term in the numerator is cancelled out by the same term in the denominator of the final result.
	float k = square(roughness + 1) / 8.0;
	float G = 1 / ((NdotL * (1 - k) + k) * (NdotV * (1 - k) + k));

	float3 F = Schlick_Fresnel(specularF0, VdotH);

	return F * (D * G * NdotL / 4);
}

void ShadeSurface(LightData light, MaterialSample materialSample, float3 surfacePos, float3 viewIncident,
	out float3 o_diffuseRadiance, out float3 o_specularRadiance)
{
	// [TODO]: add radius for point light and angular size for directional light.
	o_diffuseRadiance = 0;
	o_specularRadiance = 0;
	
	float3 lightIncident = 0;
	float irradiance = 0;
	
	if (light.type == LightType_Point) {
		lightIncident = light.position - surfacePos;
		float distance = sqrt(dot(lightIncident, lightIncident));
		lightIncident = lightIncident / distance;
		irradiance = light.scale / (distance * distance);
	} else if (light.type == LightType_Directional) {
		lightIncident = -normalize(light.direction);
		irradiance = light.scale;
	} else return;
	
	o_diffuseRadiance = Lambert(materialSample.normal, lightIncident) * materialSample.diffuse * irradiance * light.color;
    o_specularRadiance = GGX_AnalyticalLights(lightIncident, viewIncident, materialSample.normal, materialSample.roughness, materialSample.specular, 0) * irradiance * light.color;
}

void vs_main(
	in uint i_vertexID : SV_VertexID,
	out float4 o_position : SV_Position,
	out Vertex o_vertex) {
	
    InstanceData instance = t_InstanceData[g_RenderConstants.instanceID];
	MeshData mesh = t_MeshData[instance.meshIndex];

	ByteAddressBuffer indexBuffer = t_BindlessBuffers[mesh.indexBufferIndex];
	ByteAddressBuffer vertexBuffer = t_BindlessBuffers[mesh.vertexBufferIndex];
	
	uint index = indexBuffer.Load(mesh.indexOffset + i_vertexID * 4 /*sizeof(uint)*/);
	float2 texcoord = mesh.texCoordOffset == ~0u ? 0 : 
		asfloat(vertexBuffer.Load2(mesh.texCoordOffset + index * 8 /*sizeof(float2)*/));
	float3 position = mesh.positionOffset == ~0u ? 0 : 
		asfloat(vertexBuffer.Load3(mesh.positionOffset + index * 12 /*sizeof(float3)*/));
	float3 normal = mesh.normalOffset == ~0u ? 0 : 
		asfloat(vertexBuffer.Load3(mesh.normalOffset + index * 12 /*sizeof(float3)*/));
	float3 tangent = mesh.tangentOffset == ~0u ? 0 : 
		asfloat(vertexBuffer.Load3(mesh.tangentOffset + index * 12 /*sizeof(float3)*/));

	float4 worldSpacePosition = mul(instance.transform, float4(position, 1.0));
	float4 clipSpacePosition  = mul(g_ViewConstants.worldToClip, worldSpacePosition);

	o_position = clipSpacePosition;
	// world-space shading data.
	o_vertex.uv = texcoord;
	o_vertex.position = worldSpacePosition;
	o_vertex.normal = mul(instance.transform, float4(normal, 1)).xyz;
	o_vertex.tangent = mul(instance.transform, float4(tangent, 1)).xyz;
}

void ps_main(
	in float4 i_position : SV_Position,
	in Vertex i_vertex,
	out float4 o_color: SV_Target0) {
	
	InstanceData instance = t_InstanceData[g_RenderConstants.instanceID];
	MeshData mesh = t_MeshData[instance.meshIndex];
	MaterialConstants material = t_MaterialConstants[mesh.materialIndex];
	MaterialSample materialSample = EvaluateSceneMaterial(i_vertex.uv, i_vertex.normal, i_vertex.tangent, material);

	/* forward lighting */
	float3 diffuseTerm = 0, specularTerm = 0;
	[loop] 
	for (uint nLight = 0; nLight < g_LightConstants.numLights; nLight++) {
        LightData light = t_LightData[nLight];
		float3 diffuseRadiance = 0, specularRadiance = 0;
        float3 viewIncident = normalize(g_ViewConstants.cameraPosition - i_vertex.position);
		ShadeSurface(light, materialSample, i_vertex.position, viewIncident, diffuseRadiance, specularRadiance);
		
		diffuseTerm += diffuseRadiance;
        specularTerm += specularRadiance;
    }
	
    float3 ambientColor = lerp(g_LightConstants.ambientBottom, g_LightConstants.ambientTop, materialSample.normal.y * 0.5 + 0.5);
    diffuseTerm += ambientColor * materialSample.diffuse;
    specularTerm += ambientColor * materialSample.specular;
	
	/* combine together */
	o_color.rgb = diffuseTerm + specularTerm;
	o_color.a = 1;
}