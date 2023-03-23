#pragma pack_matrix(column_major)

struct ViewConstants {
	float4x4 worldToView;
};

struct InstantConstants {
	uint instance;
};

struct MaterialConstants {
	float3 baseColor;
	float roughness;

	float3 specularColor;
	float opacity;

	float3 emissiveColor;
	int baseTextureIndex;

	int specularTextureIndex;
	int normalTextureIndex;
	int emissiveTextureIndex;
	int flags;
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

	uint bitangentOffset;
	uint materialIndex;
	int2 padding;
};

ConstantBuffer<ViewConstants> g_ViewConstants : register(b0);
[[vk::push_constant]] ConstantBuffer<InstanceConstants> g_Instance : register(b1);

StructuredBuffer<MeshData> t_MeshData : register(t0);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t1);
SamplerState s_MaterialSampler;
// the above bindings are implicitly assigned to register space 0.
// the bindless buffer arrays below actually bind to a register range.
[[vk::binding(0, 1)]] t_BindlessBuffers[] : register(t0, space1);	// register space, check it out later
[[vk::binding(1, 1)]] t_BindlessTextures[] : register(t0, space2);

void vs_main(
	in uint i_vertexID: SV_VertexID,
	out float4 o_position: SV_Position,
	out float2 o_uv: TEXCOORD,
	out uint o_material: MATERIAL) {
	
}

void ps_main(
	in float4 i_position: SV_Position,
	in float2 i_uv: TEXCOORD,
	nointerpolation in uint i_material: MATERIAL,
	out float4 o_color: SV_Target0) {
	
}