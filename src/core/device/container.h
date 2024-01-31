#pragma once
#include "common.h"
#include "buffer.h"

NAMESPACE_BEGIN(krr)

template <typename T, int Dim> 
class Grid {
public:
	Grid() = default;
	
	TypedBufferView<T> voxels;
	Vector<int, Dim> res;
};

NAMESPACE_END(krr)