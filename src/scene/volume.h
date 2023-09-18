#pragma once
#include <filesystem>

#define NOMINMAX
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/CudaDeviceBuffer.h>

nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> loadNanoVDB(std::filesystem::path path);