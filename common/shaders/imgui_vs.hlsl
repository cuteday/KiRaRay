struct Constants
{
    float2 invDisplaySize;
};

[[vk::push_constant]] ConstantBuffer<Constants> g_Const;

struct VS_INPUT
{
    float2 pos : POSITION;
    float2 uv  : TEXCOORD0;
    float4 col : COLOR0;
};

struct PS_INPUT
{
    float4 out_pos : SV_POSITION;
    float4 out_col : COLOR0;
    float2 out_uv  : TEXCOORD0;
};

PS_INPUT main(VS_INPUT input)
{
    PS_INPUT output;
    output.out_pos.xy = input.pos.xy * g_Const.invDisplaySize * float2(2.0, -2.0) + float2(-1.0, 1.0);
    output.out_pos.zw = float2(0, 1);
    output.out_col = input.col;
    output.out_uv = input.uv;
    return output;
}

