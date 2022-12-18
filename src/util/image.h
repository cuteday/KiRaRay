#pragma once

#include "common.h"

#include <filesystem>

KRR_NAMESPACE_BEGIN

namespace tinyexr {
void save_exr(const float *data, int width, int height, int nChannels, int channelStride,
			  const char *outfilename, bool flip = true);
int load_exr(float **data, int *width, int *height, const char *filename, bool filp = true);
} // namespace tinyexr

namespace pfm {
/* out: data[rgb], res_x, res_y */
Color4f *ReadImagePFM(const std::string &filename, int *xres, int *yres);
} // namespace pfm


KRR_NAMESPACE_END