#pragma once

#include "common.h"
#include <Eigen/Dense>

KRR_NAMESPACE_BEGIN

template <typename T, int Size>
class Array : public Eigen::Array<T, Size, 1> {
public:
	enum {dim = Size};

	KRR_CALLABLE Array(void) : Eigen::Array<T, Size, 1>(Eigen::Array<T, Size, 1>::Zero()) {}

	KRR_CALLABLE Array(T v) : Eigen::Array<T, Size, 1>(Eigen::Array<T, Size, 1>::Constant(v)) {}

	KRR_CALLABLE Array(Eigen::Vector<T, Size> vec): Eigen::Array<T, Size, 1>(vec.array()) {}

	template <typename OtherDerived>
	KRR_CALLABLE Array(const Array<OtherDerived, Size> &other) {
		this->Eigen::Array<T, Size, 1>::operator=(other.template cast<T>());
	}

	template <typename OtherDerived>
	KRR_CALLABLE Array(const Eigen::ArrayBase<OtherDerived> &other) : 
		Eigen::Array<T, Size, 1>(other) {}

	template <typename OtherDerived>
	KRR_CALLABLE Array &operator=(const Eigen::ArrayBase<OtherDerived> &other) {
		this->Eigen::Array<T, Size, 1>::operator=(other);
		return *this;
	}

	KRR_CALLABLE friend Array max(const Array &arr1, const Array &arr2) { return arr1.max(arr2); }
	KRR_CALLABLE friend Array min(const Array &arr1, const Array &arr2) { return arr1.min(arr2); }
	KRR_CALLABLE friend Array abs(const Array &arr) { return arr.abs(); }
	KRR_CALLABLE friend Array sqrt(const Array &arr) { return arr.sqrt(); }
	KRR_CALLABLE friend Array inverse(const Array &arr) { return arr.inverse(); }
};

template <typename T>
class Array2 : public Array<T, 2> {
public:
	using Array<T, 2>::Array;

	KRR_CALLABLE Array2(Array<T, 3>& v) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
	}

	KRR_CALLABLE Array2(const T &x, const T &y) {
		this->operator[](0) = x;
		this->operator[](1) = y;
	}

#ifdef __CUDACC__
	KRR_CALLABLE operator float2() const {
		return make_float2(this->operator[](0), this->operator[](1));
	}
	
	KRR_CALLABLE Array2(const float2 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}

	KRR_CALLABLE Array2(const uint2 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}

	KRR_CALLABLE Array2(const uint3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
	}
#endif

	friend void to_json(json &j, const Array2<T> &v) {
		for (int i = 0; i < 2; i++) {
			j.push_back(v[i]);
		}
	}

	friend void from_json(const json &j, Array2<T> &v) {
		for (int i = 0; i < 2; i++) {
			v[i] = (T) j.at(i);
		}
	}
};

template <typename T>
class Array3 : public Array<T, 3> {
public:
	using Array<T, 3>::Array;

	KRR_CALLABLE Array3(Array<T, 4>& v) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
		this->operator[](2) = v.operator[](2);
	}

	KRR_CALLABLE Array3(const T &x, const T &y, const T &z) {
		this->operator[](0) = x;
		this->operator[](1) = y;
		this->operator[](2) = z;
	}

#ifdef __CUDACC__

	KRR_CALLABLE operator float3() const {
		return make_float3(this->operator[](0), this->operator[](1), this->operator[](2));
	}

	KRR_CALLABLE Array3(const float3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}

	KRR_CALLABLE Array3(const uint3 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}

	KRR_CALLABLE Array3(const float4 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
	}
#endif

	friend void to_json(json &j, const Array3<T> &v) {
		for (int i = 0; i < 3; i++) {
			j.push_back(v[i]);
		}
	}

	friend void from_json(const json &j, Array3<T> &v) {
		for (int i = 0; i < 3; i++) {
			v[i] = (T) j.at(i);
		}
	}
};

template <typename T>
class Array4 : public Array<T, 4> {
public:
	using Array<T, 4>::Array;

	KRR_CALLABLE Array4(Array<T, 3>& v, T w) {
		this->operator[](0) = v.operator[](0);
		this->operator[](1) = v.operator[](1);
		this->operator[](2) = v.operator[](2);
		this->operator[](3) = w;
	}

	KRR_CALLABLE Array4(const T &x, const T &y, const T &z, const T &w) {
		this->operator[](0) = x;
		this->operator[](1) = y;
		this->operator[](2) = z;
		this->operator[](3) = w;
	}

#ifdef __CUDACC__
	KRR_CALLABLE operator float4() const {
		return make_float4(this->operator[](0), this->operator[](1), this->operator[](2),
						   this->operator[](3));
	}

	KRR_CALLABLE Array4(const float4 &v) {
		this->operator[](0) = v.x;
		this->operator[](1) = v.y;
		this->operator[](2) = v.z;
		this->operator[](3) = v.w;
	}
#endif

	friend void to_json(json &j, const Array4<T> &v) {
		for (int i = 0; i < 4; i++) {
			j.push_back(v[i]);
		}
	}

	friend void from_json(const json &j, Array4<T> &v) {
		for (int i = 0; i < 4; i++) {
			v[i] = (T) j.at(i);
		}
	}
};

using Array2i = Array2<int>;
using Array2f = Array2<float>;
using Array2ui = Array2<uint>;
using Array3i = Array3<int>;
using Array3f = Array3<float>;
using Array3ui = Array3<uint>;
using Array4i = Array4<int>;
using Array4f = Array4<float>;
using Array4ui = Array4<uint>;
KRR_NAMESPACE_END