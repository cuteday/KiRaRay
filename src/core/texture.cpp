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

bool Image::loadImage(const string& filepath, bool srgb)
{
	//if (sImageCache.count(filepath))
	//	return sImageCache[filepath];
	vec2i size;
	int channels;
	uchar* data = stbi_load(filepath.c_str(), &size.x, &size.y, &channels, STBI_rgb_alpha);
	if (data == nullptr) {
		logError("Failed to load image at " + filepath);
		return false;
	}
	mSize = size;
	mChannels = channels;
	//mData = std::vector<uchar>(data, data + size.x * size.y * channels);
	//free(data);
	mData = data;
	//sImageCache[filepath] = image;
	return true;
}

Image::SharedPtr Image::createFromFile(const string& filepath, bool srgb)
{
	Image::SharedPtr pImage = Image::SharedPtr(new Image());
	pImage->loadImage(filepath);
	return pImage;
}

Texture::SharedPtr Texture::createFromFile(const string& filepath, bool srgb)
{
	Texture::SharedPtr pTexture = Texture::SharedPtr(new Texture());
	pTexture->loadImage(filepath);
	return pTexture;
}

void Texture::toDevice() {
	if (!mImage.isValid()) return;

	// we transfer our texture data to a cuda array, then make the array a cuda texture object.
	vec2i size = mImage.getSize();
	uint numComponents = mImage.getChannels();
	if (numComponents != 4)
		logError("Incorrect texture image channels (not 4)");
	// we have no padding so pitch == width
	uint pitch = size.x * numComponents * sizeof(uint8_t);
	cudaChannelFormatDesc channelDesc = {};
	channelDesc = cudaCreateChannelDesc<uchar4>();

	cudaArray_t& pixelArray = mCudaArray;
	// create internal cuda array for texture object
	CUDA_CHECK(cudaMallocArray(&pixelArray, &channelDesc, size.x, size.y));
	// transfer data to cuda array
	CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, mImage.data(), pitch, pitch, size.y, cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = pixelArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 1;
	texDesc.maxMipmapLevelClamp = 99;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.mipmapFilterMode = cudaFilterModePoint;
	texDesc.borderColor[0] = 1.0f;
	texDesc.sRGB = 0;

	CUDA_CHECK(cudaCreateTextureObject(&mCudaTexture, &resDesc, &texDesc, nullptr));
}

void Material::toDevice() {
	for (uint i = 0; i < (uint)TextureType::Count; i++) {
		mTextures->toDevice();
	}
}

KRR_NAMESPACE_END

