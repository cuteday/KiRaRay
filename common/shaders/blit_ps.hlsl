#if TEXTURE_ARRAY
Texture2DArray tex : register(t0);
#else
Texture2D tex : register(t0);
#endif
SamplerState samp : register(s0);

void main(
	in float4 pos : SV_Position,
	in float2 uv : UV,
	out float4 o_rgba : SV_Target)
{
#if TEXTURE_ARRAY
	o_rgba = tex.Sample(samp, float3(uv, 0));
#else
	o_rgba = tex.Sample(samp, uv);
#endif
}
