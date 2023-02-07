struct BlitConstants {
	float2 sourceOrigin;
	float2 sourceSize;

	float2 targetOrigin;
	float2 targetSize;

	float sharpenFactor;
};

#if TEXTURE_ARRAY
Texture2DArray tex : register(t0);
#else
Texture2D tex : register(t0);
#endif
SamplerState samp : register(s0);

cbuffer c_Blit : register(b0)
{
    BlitConstants g_Blit;
};

void main(
	in float4 i_pos : SV_Position,
	in float2 i_uv : UV,
	out float4 o_rgba : SV_Target)
{
#if TEXTURE_ARRAY
	float4 x = tex.SampleLevel(samp, float3(i_uv, 0), 0);
	
	float4 a = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2(-1,  0));
	float4 b = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2( 1,  0));
	float4 c = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2( 0,  1));
	float4 d = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2( 0, -1));

	float4 e = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2(-1, -1));
	float4 f = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2( 1,  1));
	float4 g = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2(-1,  1));
	float4 h = tex.SampleLevel(samp, float3(i_uv, 0), 0, int2( 1, -1));
#else
	float4 x = tex.SampleLevel(samp, i_uv, 0);
	
	float4 a = tex.SampleLevel(samp, i_uv, 0, int2(-1,  0));
	float4 b = tex.SampleLevel(samp, i_uv, 0, int2( 1,  0));
	float4 c = tex.SampleLevel(samp, i_uv, 0, int2( 0,  1));
	float4 d = tex.SampleLevel(samp, i_uv, 0, int2( 0, -1));

	float4 e = tex.SampleLevel(samp, i_uv, 0, int2(-1, -1));
	float4 f = tex.SampleLevel(samp, i_uv, 0, int2( 1,  1));
	float4 g = tex.SampleLevel(samp, i_uv, 0, int2(-1,  1));
	float4 h = tex.SampleLevel(samp, i_uv, 0, int2( 1, -1));
#endif
	
	o_rgba = x * (6.828427 * g_Blit.sharpenFactor + 1) 
		- (a + b + c + d) * g_Blit.sharpenFactor 
		- (e + g + f + h) * g_Blit.sharpenFactor * 0.7071;
}
