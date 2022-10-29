#pragma once

#include "common.h"
#include "math/math.h"
#include <filesystem>

KRR_NAMESPACE_BEGIN


namespace tinyexr {
void save_exr(const float *data, int width, int height, int nChannels, int channelStride,
			  const char *outfilename, bool flip = true);
} // namespace tinyexr

namespace pfm {

static inline int isWhitespace(char c);

static int readWord(FILE *fp, char *buffer, int bufferLength);
/* out: data[rgb], res_x, res_y */
Color4f *ReadImagePFM(const std::string &filename, int *xres, int *yres);

} // namespace pfm

namespace image {

Color4f *convertEqualAeraOctahedralMappingToSpherical(Color4f *data, int width, int height);

}

KRR_NAMESPACE_END