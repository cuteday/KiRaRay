#pragma pack_matrix(column_major)

cbuffer CB : register(b0)
{
    float4x4 g_Transform;
};

void main_vs(
	float3 i_pos : POSITION,
    float2 i_uv : UV,
	out float4 o_pos : SV_Position,
	out float2 o_uv : UV
)
{
	o_pos = mul(g_Transform, float4(i_pos, 1));
	o_uv = i_uv;
}


Texture2D t_Texture : register(t0);
SamplerState s_Sampler : register(s0);

void main_ps(
	in float4 i_pos : SV_Position,
	in float2 i_uv : UV,
	out float4 o_color : SV_Target0
)
{
    o_color = t_Texture.Sample(s_Sampler, i_uv);
	if (o_color.a == 0.f) o_color = float4(1, 1, 1, 1);
}