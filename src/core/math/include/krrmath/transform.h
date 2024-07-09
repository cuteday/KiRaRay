// Transform transformation and clipspace arithmetric
#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "common.h"
#include "vector.h"
#include "constants.h"

NAMESPACE_BEGIN(krr)

template <typename T, int Dim, int Mode, int Options = Eigen::RowMajor>
class Transform : public Eigen::Transform<T, Dim, Mode, Options> {
public:
	using Eigen::Transform<T, Dim, Mode, Options>::Transform;
	typedef Eigen::Transform<T, Dim, Mode, Options> NativeType;
	typedef Vector<T, Dim> ScalingVectorType;
	typedef Matrix<T, Dim, Dim, Options> LinearMatrixType;
	typedef Matrix<T, Dim, Dim, Options> RotationMatrixType;
	
	KRR_CALLABLE Transform(void)
		: Eigen::Transform<T, Dim, Mode, Options>(Eigen::Transform<T, Dim, Mode, Options>::Identity()) {}

	template <typename U = T> /* Dummy parameter for enable_if */
	KRR_CALLABLE Transform(typename std::enable_if_t<Dim == Mode && std::is_floating_point_v<U>, U> v) 
		: Eigen::Transform<T, Dim, Mode, Options>(Vector<T, Dim>::Constant(v).asDiagonal()) {}

	template <typename OtherDerived>
	KRR_CALLABLE Transform(const Eigen::MatrixBase<OtherDerived> &other)
		: Eigen::Transform<T, Dim, Mode, Options>(other) {}

	template <typename OtherDerived>
	KRR_CALLABLE Transform &operator=(const Eigen::MatrixBase<OtherDerived> &other) {
		this->Eigen::Transform<T, Dim, Mode, Options>::operator=(other);
		return *this;
	}

	KRR_CALLABLE ScalingVectorType scaling() const {
		LinearMatrixType result;
		this->computeRotationScaling((LinearMatrixType *) nullptr, &result);
		return ScalingVectorType(result.diagonal());
	}

	KRR_CALLABLE Matrix<T, NativeType::MatrixType::RowsAtCompileTime,
						NativeType::MatrixType::ColsAtCompileTime, Options> matrix() const { 
		return NativeType::matrix();
	}
};

/* About storage order: 
 * In most cases, changing the storage order will not affect the correctness of your application
 * (since Transform arithmetic operations, like indexing, ctor, etc., all agnostic to storage order),
 * except for cases like passing the internal data of matrices to shaders.
 * Both OpenGL/Vulkan (GLSL/HLSL) assumes matrices being in column-major, so we use this by default.
 */

template <int Dim, int Mode, int Options = Eigen::RowMajor>
using Transformf = Transform<float, Dim, Mode, Options>;

template <int Dim, int Options = Eigen::RowMajor>
using Affinef = Transformf<Dim, Eigen::Affine, Options>;

using Affine2f = Affinef<2, Eigen::RowMajor>;
using Affine3f = Affinef<3, Eigen::RowMajor>;

NAMESPACE_END(krr)