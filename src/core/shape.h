#pragma once

#include "common.h"
#include "device/buffer.h"
#include "taggedptr.h"

KRR_NAMESPACE_BEGIN

class Mesh;
class MeshData;

class Triangle{
public:
	Triangle() = default;
	Triangle(uint meshId, uint triId) :
		meshId(meshId), triId(triId) {}

	__both__ inline float area() {
		return 0;
	}

private:
	friend class Mesh;

	uint meshId, triId;
	MeshData* mesh;
};


class Shape: TaggedPointer<Triangle>{
public:
	using TaggedPointer::TaggedPointer;

	__both__ inline size_t size() {
		auto f = [&](auto* ptr)-> size_t {return sizeof *ptr; };
		return dispatch(f);
	}

	__both__ inline float area() {
		auto f = [&](auto ptr) ->float {return ptr->area(); };
		return dispatch(f);
	}

	//static void toDevice(std::vector<Shape>& shapes, TypedBuffer<Shape> &buffer, CUDABuffer& storage) {
	//	uint nShapes = shapes.size();
	//	std::vector<Shape> shapeDevice(nShapes);
	//	for (int i = 0; i < nShapes; i++) {
	//		size_t shapeSize = shapes[i].size();
	//	}
	//}
};

KRR_NAMESPACE_END