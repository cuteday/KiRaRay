#pragma once

namespace krr::exr {

// Returns malloc-owned RGBA floats. Throws on failure without transferring partial data.
void load(const char *filename, float **data, int *width, int *height, bool flip = false);

} // namespace krr::exr
