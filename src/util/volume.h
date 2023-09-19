#pragma once
#include <filesystem>

#define NOMINMAX
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

#include "krrmath/math.h"

nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> loadNanoVDB(std::filesystem::path path, float* maxDensity = nullptr);