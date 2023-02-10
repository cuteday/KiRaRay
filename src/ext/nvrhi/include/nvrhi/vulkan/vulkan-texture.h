#pragma once
#include <nvrhi/vulkan/vulkan-backend.h>

namespace nvrhi::vulkan {
void fillTextureInfo(Texture *texture, const TextureDesc &desc);
}