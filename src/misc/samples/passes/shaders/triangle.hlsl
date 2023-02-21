static const float2 g_positions[] = {
	float2(-0.5, -0.5),
	float2(0, 0.5),
	float2(0.5, -0.5)
};

static const float3 g_colors[] = {
	float3(1, 0, 0),
	float3(0, 1, 0),
	float3(0, 0, 1)	
};

void main_vs(
	uint i_vertexId : SV_VertexID,
	out float4 o_pos : SV_Position,
	out float3 o_color : COLOR
)
{
	o_pos = float4(g_positions[i_vertexId], 0, 1);
	o_color = g_colors[i_vertexId];
}

void main_ps(
	in float4 i_pos : SV_Position,
	in float3 i_color : COLOR,
	out float4 o_color : SV_Target0
)
{
	o_color = float4(i_color, 1);
}