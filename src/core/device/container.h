#pragma once
#include "common.h"
#include "buffer.h"

KRR_NAMESPACE_BEGIN

template <typename T, int Dim> 
class Grid {
public:
	Grid() = default;
	
	TypedBufferView<T> voxels;
	Vector<int, Dim> res;
};



KRR_NAMESPACE_END