#pragma once

#include "common.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "vector.h"

NAMESPACE_BEGIN(krr)

template <typename T, int Rows, int Cols, int Options = Eigen::RowMajor>
class Matrix : public Eigen::Matrix<T, Rows, Cols, Options> {
public:
	using Eigen::Matrix<T, Rows, Cols, Options>::Matrix;
	
	KRR_CALLABLE Matrix(void)
		: Eigen::Matrix<T, Rows, Cols, Options>(Eigen::Matrix<T, Rows, Cols, Options>::Zero()) {}

	template <typename U = T> /* Dummy parameter for enable_if */
	KRR_CALLABLE Matrix(typename std::enable_if_t<Rows == Cols, U> v) 
		: Eigen::Matrix<T, Rows, Cols, Options>(Vector<T, Rows>::Constant(v).asDiagonal()) {}

	template <typename OtherDerived>
	KRR_CALLABLE Matrix(const Eigen::MatrixBase<OtherDerived> &other)
		: Eigen::Matrix<T, Rows, Cols, Options>(other) {}

	template <typename OtherDerived>
	KRR_CALLABLE Matrix &operator=(const Eigen::MatrixBase<OtherDerived> &other) {
		this->Eigen::Matrix<T, Rows, Cols, Options>::operator=(other);
		return *this;
	}

	std::string string() const { 
		std::stringstream ss;
		ss << *this;
		return ss.str();
	}

#ifdef KRR_MATH_JSON
	friend void to_json(json &j, const Matrix<T, Rows, Cols, Options> &m) {
		for (int i = 0; i < Rows; i++) j.push_back(Vector<T, Cols>(m.row(i)));
	}

	friend void from_json(const json &j, Matrix<T, Rows, Cols, Options> &m) {
		assert(j.size() == Rows);
		for (int i = 0; i < Rows; i++) m.row(i) = j.at(i).get<Vector<T, Cols>>();
	}
#endif
};

/* About storage order: 
 * In most cases, changing the storage order will not affect the correctness of your application
 * (since matrix arithmetic operations, like indexing, ctor, etc., all agnostic to storage order),
 * except for cases like passing the internal data of matrices to shaders.
 * Both OpenGL/Vulkan (GLSL/HLSL) assumes matrices being in column-major, so we use this by default.
 */

template <int Rows, int Cols, int Options = Eigen::ColMajor>
using Matrixf = Matrix<float, Rows, Cols, Options>;

using Matrix2f = Matrixf<2, 2, Eigen::ColMajor>;
using Matrix3f = Matrixf<3, 3, Eigen::ColMajor>;
using Matrix4f = Matrixf<4, 4, Eigen::ColMajor>;

NAMESPACE_END(krr)