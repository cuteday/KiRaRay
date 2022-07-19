#pragma once

#include "common.h"
#include <Eigen/Dense>

KRR_NAMESPACE_BEGIN

// template parameters are not allowed (since they contain comma) 
#define KRR_INHERIT_EIGEN(name, parent)																\
		using Eigen::parent::parent;																\
		KRR_CALLABLE name(void) : Eigen::parent() {}												\
		template <typename OtherDerived>															\
		KRR_CALLABLE name(const Eigen::MatrixBase<OtherDerived> &other) : Eigen::parent(other) {}	\
		template <typename OtherDerived>															\
		KRR_CALLABLE name &operator=(const Eigen::MatrixBase<OtherDerived> &other) {				\
			this->Eigen::parent::operator=(other);													\
			return *this;																			\
		}																					

template <typename T, int Size>
class Vector : public Eigen::Vector<T, Size> {
public:
	using Eigen::Vector<T, Size>::Vector;

	KRR_CALLABLE Vector(void) : Eigen::Vector<T, Size>() {}

	KRR_CALLABLE Vector(T v)
		: Eigen::Vector<T, Size>(Eigen::Vector<T, Size>::Constant(v)) {}

	KRR_CALLABLE Vector(Eigen::Array<T, Size, 1> arr) : Eigen::Vector<T, Size>(arr.matrix()) {}
	
	template <typename OtherDerived>
	KRR_CALLABLE Vector(const Eigen::MatrixBase<OtherDerived> &other) : Eigen::Vector<T, Size>(other) {}
	
	template <typename OtherDerived>
	KRR_CALLABLE Vector &operator=(const Eigen::MatrixBase<OtherDerived> &other) {
		this->Eigen::Vector<T, Size>::operator=(other);
		return *this;
	}		

	template <typename OtherDerived>
	KRR_CALLABLE Vector(const Eigen::Vector<OtherDerived, Size> &other) {
		*this = other.template cast<T>();
	}
};

template <typename T>
class Vector2 : public Vector<T, 2> {
public:
	using Vector<T, 2>::Vector;

	KRR_CALLABLE Vector2(Vector3<T> v) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
	}

#ifdef __CUDACC__

	KRR_CALLABLE operator float2() const {
		return make_float2(this->operator[](0), this->operator[](1));
	}

	KRR_CALLABLE Vector2(const float2 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}

	KRR_CALLABLE Vector2(const uint2 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}

	KRR_CALLABLE Vector2(const uint3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}
#endif
};

template <typename T>
class Vector3 : public Vector<T, 3> {
public:
	using Vector<T, 3>::Vector;

	KRR_CALLABLE Vector3(Vector4<T> v) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
		this->operator[](2) = v.operator[](2);
	}
	
#ifdef __CUDACC__

	KRR_CALLABLE operator float3() const {
		return make_float3(this->operator[](0), 
			this->operator[](1), 
			this->operator[](2));
	}
	
	KRR_CALLABLE Vector3(const float3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}

	KRR_CALLABLE Vector3(const uint3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}
	
	KRR_CALLABLE Vector3(const float4 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}
#endif

};

template <typename T>
class Vector4 : public Vector<T, 4> {
public:
	using Vector<T, 4>::Vector;

	KRR_CALLABLE Vector4(Vector3<T> v, T w) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
		this->operator[](2) = v.operator[](2);
		this->operator[](3) = w;
	}

#ifdef __CUDACC__
	KRR_CALLABLE operator float4() const {
		return make_float4(this->operator[](0), this->operator[](1), this->operator[](2), this->operator[](3));
	}

	KRR_CALLABLE Vector4(const float4 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
		this->operator[](3) = v.w;
	}
#endif
};

using vec2f = Vector2<float>;
using vec2i = Vector2<int>;
using vec2ui = Vector2<uint>;
using vec3f = Vector3<float>;
using vec3i = Vector3<int>;
using vec3ui = Vector3<uint>;
using vec4f = Vector4<float>;
using vec4i = Vector4<int>;
using vec4ui = Vector4<uint>;

KRR_NAMESPACE_END