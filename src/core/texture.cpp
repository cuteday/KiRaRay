#include "texture.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define TINYEXR_USE_MINIZ (0)
#define TINYEXR_IMPLEMENTATION
#include "zlib.h"			// needed by tinyexr
#include "stb_image.h"
#include "stb_image_write.h"
#include "tinyexr.h"


KRR_NAMESPACE_BEGIN


Image Image::loadImage(const string& filepath)
{
	if (sImageCache.count(filepath))
		return sImageCache[filepath];

	Image image;
	
	vec2i size;
	int channels;
	uchar* data = stbi_load(filepath.c_str(), &size.x, &size.y, &channels, STBI_rgb_alpha);
	
	if (data == nullptr) {
		logError("Failed to load image at " + filepath);
	}

	image.mSize = size;
	image.mChannels = channels;
	image.mData = std::vector<uchar>(data, data + size.x * size.y * channels);

	free(data);
	sImageCache[filepath] = image;
	return image;
}


KRR_NAMESPACE_END
