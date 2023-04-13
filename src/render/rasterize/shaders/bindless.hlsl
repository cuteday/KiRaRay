#pragma pack_matrix(column_major)

struct ViewConstants {
	float4x4 worldToView;
	float4x4 viewToClip;
	float4x4 worldToClip;
};

struct RenderConstants {
	/* push constants */
	uint meshID;
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

ConstantBuffer<ViewConstants> g_ViewConstants : register(b0);
[[vk::push_constant]] ConstantBuffer<RenderConstants> g_RenderConstants : register(b1);

StructuredBuffer<MeshData> t_MeshData : register(t0);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t1);
SamplerState s_MaterialSampler : register(s0);
// the above bindings are implicitly assigned to register space 0.
// the bindless buffer arrays below actually bind to a register range.
[[vk::binding(0, 1)]] ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);	// register space, check it out later
[[vk::binding(1, 1)]] Texture2D t_BindlessTextures[] : register(t0, space2);

void vs_main(
	in uint i_vertexID: SV_VertexID,
	out float4 o_position: SV_Position,
	out float2 o_uv: TEXCOORD,
	//out float3 o_normal: NORMAL,
	//out float3 o_tangent: TANGENT,
	out uint o_material: MATERIAL) {
	
	MeshData mesh = t_MeshData[g_RenderConstants.meshID];

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

	float4 clipSpacePosition = mul(g_ViewConstants.worldToClip, float4(position, 1.0));

	o_uv = texcoord;
	o_position = clipSpacePosition;
	o_material = mesh.materialIndex;
}

void ps_main(
	in float4 i_position: SV_Position,
	in float2 i_uv: TEXCOORD,
	//in float3 i_normal: NORMAL,
	//in float3 i_tangent: TANGENT,
	nointerpolation in uint i_material: MATERIAL,
	out float4 o_color: SV_Target0) {
	
	MaterialConstants material = t_MaterialConstants[i_material];

	float4 diffuse = material.baseColor;
	if (material.baseTextureIndex >= 0) {
		diffuse *= t_BindlessTextures[material.baseTextureIndex].Sample(
			s_MaterialSampler, i_uv);
	}

	o_color = float4(diffuse.rgb, 1);
}