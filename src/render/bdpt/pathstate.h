#pragma once
#include "common.h"
#include <atomic>

#include "device/cuda.h"
#include "device/atomic.h"
#include "logger.h"
#include "render/wavefront/workitem.h"

KRR_NAMESPACE_BEGIN

constexpr size_t MAX_BDPT_DEPTH = 10;

enum class VertexType { Camera, Light, Surface };

struct Vertex {
	VertexType type;
	RGB throughput;
	bool delta{};
	float pdfFwd{}, pdfRev{};
};

struct BDPTPathState {
	Vertex cameraSubpath[MAX_BDPT_DEPTH + 2];
	Vertex lightSubpath[MAX_BDPT_DEPTH + 1];
};

#pragma warning(push, 0)
#pragma warning(disable : ALL_CODE_ANALYSIS_WARNINGS)
#include "render/bdpt/workitem_soa.h"
#pragma warning(pop)

class BDPTPathStateBuffer : public SOA<BDPTPathState> {
public:
	BDPTPathStateBuffer() = default;
	BDPTPathStateBuffer(int n, Allocator alloc) 
		: SOA<BDPTPathState>(n, alloc) {}
};

KRR_NAMESPACE_END