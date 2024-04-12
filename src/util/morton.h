#pragma once
#include "common.h"
#include "krrmath/math.h"

NAMESPACE_BEGIN(krr)

//https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
//https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

template <typename T, int n_bits>
KRR_CALLABLE T expandBits(T x) {
	static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>, 
		"expandBits: argument must have integral type in 32/64 bits.");
	if constexpr (n_bits == 1) {
		x &= static_cast<T>(0xffffffff);
		x = (x ^ (x << 16)) & static_cast<T>(0x0000ffff0000ffff);
		x = (x ^ (x << 8)) & static_cast<T>(0x00ff00ff00ff00ff);
		x = (x ^ (x << 4)) & static_cast<T>(0x0f0f0f0f0f0f0f0f);
		x = (x ^ (x << 2)) & static_cast<T>(0x3333333333333333);
		x = (x ^ (x << 1)) & static_cast<T>(0x5555555555555555);
	} else if constexpr (n_bits == 2) {
		x &= static_cast<T>(0x1fffff);
		x = (x | x << 32) & static_cast<T>(0x1f00000000ffff);
		x = (x | x << 16) & static_cast<T>(0x1f0000ff0000ff);
		x = (x | x << 8) & static_cast<T>(0x100f00f00f00f00f);
		x = (x | x << 4) & static_cast<T>(0x10c30c30c30c30c3);
		x = (x | x << 2) & static_cast<T>(0x1249249249249249);
	} else
		static_assert(false, "expandBits: n_bits must be 1 or 2.");
	return x;
}

template <typename T, int dim, std::enable_if_t<std::is_integral_v<T>, bool> = true>
KRR_CALLABLE T mortonEncode(Vector<T, dim> v) {
	if constexpr (dim == 2) {
		return (expandBits<T, 1>(v[1]) << 1) | expandBits<T, 1>(v[0]);
	} else if constexpr (dim == 3) {
		return (expandBits<T, 2>(v[2]) << 2) | (expandBits<T, 2>(v[1]) << 1) |
			   expandBits<T, 2>(v[0]);
	} else {
		static_assert(false, "mortonEncode<int>: dim must be 2 or 3.");
	}
}

template <typename T, int dim, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
KRR_CALLABLE decltype(auto) mortonEncode(Vector<T, dim> v) {
	using VectorType = Vector<T, dim>;
	if constexpr (std::is_same_v<T, float>) {
		v = VectorType(v * 1024.f).cwiseMin(1023.f).cwiseMax(0.f);
		return mortonEncode<uint32_t, dim>(Vector<uint32_t, dim>(v));
	} else if constexpr (std::is_same_v<T, double>) {
		v = VectorType(v * (1 << 21)).cwiseMax(0.0).cwiseMin(static_cast<double>((1 << 21) - 1));
		return mortonEncode<uint64_t, dim>(Vector<uint64_t, dim>(v));
	} else {
		static_assert(false, "mortonEncode<float>: T must be float or double.");
	}
}

NAMESPACE_END(krr)