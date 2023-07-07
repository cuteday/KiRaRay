#pragma pack_matrix(column_major)

struct ViewConstants {
	float4x4 worldToView;
	float4x4 viewToClip;
	float4x4 worldToClip;
};

struct LightConstants {
	uint numLights;
	uint3 padding;
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
    float4 diffuse;
    float4 specular;
    float4 normal;
    float4 emissive;
};

MaterialSample EvaluateSceneMaterial(float3 normal, float4 tangent, MaterialConstants material) {
    MaterialSample result;
    result.diffuse = 1;
    result.specular = 0;
    result.normal = 0;
    result.emissive = 0;
	
    return result;
}

void ShadeSurface(LightConstants light, MaterialSample materialSample, float3 surfacePos, float3 viewIncident, 
	out float3 o_diffuseRadiance, out float3 o_specularRadiance) {

}

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

void vs_main(
	in uint i_vertexID: SV_VertexID,
	out float4 o_position: SV_Position,
	out float2 o_uv: TEXCOORD,
	out float3 o_normal: NORMAL,
	out float3 o_tangent: TANGENT,
	out uint o_material: MATERIAL) {
	
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

	o_uv = texcoord;
	o_position = clipSpacePosition;
	o_material = mesh.materialIndex;
	o_normal   = mul(instance.transform, float4(normal, 1)).xyz;
	o_tangent  = mul(instance.transform, float4(tangent, 1)).xyz;
}

void ps_main(
	in float4 i_position: SV_Position,
	in float2 i_uv: TEXCOORD,
	in float3 i_normal: NORMAL,
	in float3 i_tangent: TANGENT,
	nointerpolation in uint i_material: MATERIAL,
	out float4 o_color: SV_Target0) {
	
	MaterialConstants material = t_MaterialConstants[i_material];

	float4 diffuse = material.baseColor;
	if (material.baseTextureIndex >= 0) {
		diffuse *= t_BindlessTextures[material.baseTextureIndex].Sample(
			s_MaterialSampler, i_uv);
	}

	/* forward lighting */
	float3 diffuseTerm = 0, specularTerm = 0;
	[loop] 
	for (uint nLight = 0; nLight = g_LightConstants.numLights; nLight++) {

		LightData light = t_LightData[nLight];

		float3 diffuseRadiance, specularRadiance;
		diffuseTerm += diffuseRadiance * light.color;
		specularTerm += specularRadiance * light.color;

	}
	/* ambient term */


	/* combine together */
	o_color.rgb = diffuseTerm + specularTerm;
	o_color.a = 1;
}