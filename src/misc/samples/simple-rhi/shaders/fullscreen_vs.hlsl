void main(
	in uint iVertex : SV_VertexID,
	out float4 o_posClip : SV_Position,
	out float2 o_uv : UV)
{
	uint u = iVertex & 1;
	uint v = (iVertex >> 1) & 1;

	o_posClip = float4(float(u) * 2 - 1, 1 - float(v) * 2, QUAD_Z, 1);
	o_uv = float2(u, v);
}
