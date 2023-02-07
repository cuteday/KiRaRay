//#pragma pack_matrix(column_major)

cbuffer CB : register(b0) {
	column_major float4x4 g_modelViewProj;
};

void main_vs(
	uint i_vertexId : SV_VertexID,
	in float2 i_xyPos : POSITION, 
	in float i_height : HEIGHT,
	out float4 o_pos : SV_Position,
	out float3 o_color : COLOR
) {
	o_pos	= mul(g_modelViewProj, float4(i_xyPos.xy, 0.f, 1.f));
	o_color = float3(0.f, 0.5f + 0.5f, 0.f);
}

void main_ps(
	in float4 i_pos : SV_Position,
	in float3 i_color : COLOR,
	out float4 o_color : SV_Target0
) {
	o_color = float4(i_color, 1.f);
}