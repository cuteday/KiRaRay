#pragma pack_matrix(column_major)

float3 ColorMap(float v, float vmin, float vmax) {
	float3 c = {1.0, 1.0, 1.0}; // white
	float dv;

	if (v < vmin) v = vmin;
	if (v > vmax) v = vmax;
	dv = vmax - vmin;

	if (v < (vmin + 0.25 * dv)) {
		c.r = 0;
		c.g = 4 * (v - vmin) / dv;
	} else if (v < (vmin + 0.5 * dv)) {
		c.r = 0;
		c.b = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
	} else if (v < (vmin + 0.75 * dv)) {
		c.r = 4 * (v - vmin - 0.5 * dv) / dv;
		c.b = 0;
	} else {
		c.g = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
		c.b = 0;
	}

	return (c);
}

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
	o_pos	= mul(g_modelViewProj, float4(i_xyPos.xy, i_height, 1.f));
	o_color = ColorMap(i_height, -0.5, 0.5);
}

void main_ps(
	in float4 i_pos : SV_Position,
	in float3 i_color : COLOR,
	out float4 o_color : SV_Target0
) {
	o_color = float4(i_color, 1.f);
}