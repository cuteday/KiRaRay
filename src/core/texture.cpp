#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT

#include "zlib.h" // needed by tinyexr
#include "stb_image.h"
#include "stb_image_write.h"
#include "tinyexr.h"

#include <filesystem>
#include <cstdio>


#include "texture.h"
#include "window.h"
#include "logger.h"
#include "util/image.h"

KRR_NAMESPACE_BEGIN

namespace texture {
std::map<uint, TextureProp> textureProps;
std::map<uint, MaterialProp> materialProps;
} // namespace texture

using namespace texture;

Image::Image(Vector2i size, Format format, bool srgb) : mSrgb(srgb), mFormat(format), mSize(size) {
	mData = new uchar[size[0] * size[1] * 4 * getElementSize()];
}

bool Image::loadImage(const fs::path &filepath, bool flip, bool srgb) {
	Vector2i size;
	int channels;
	string filename = filepath.string();
	string format	= filepath.extension().string();
	uchar *data		= nullptr;
	stbi_set_flip_vertically_on_load(flip);
	if (IsEXR(filename.c_str()) == TINYEXR_SUCCESS) {
		char *errMsg = nullptr;
		// to do: if loadEXR always return RGBA data?
		// int res = LoadEXR((float**)&data, &size[0], &size[1], filename.c_str(), (const
		// char**)&errMsg);
		int res = tinyexr::load_exr((float **) &data, &size[0], &size[1], filename.c_str(), flip);

		if (res != TINYEXR_SUCCESS) {
			logError("Failed to load EXR image at " + filename);
			if (errMsg)
				logError(errMsg);
			return false;
		}
		mFormat = Format::RGBAfloat;
	} else if (stbi_is_hdr(filename.c_str())) {
		data =
			(uchar *) stbi_loadf(filename.c_str(), &size[0], &size[1], &channels, STBI_rgb_alpha);
		if (data == nullptr) {
			logError("Failed to load float hdr image at " + filename);
			return false;
		}
		mFormat = Format::RGBAfloat;
	} else if (format == ".pfm") {
		data = (uchar *) pfm::ReadImagePFM(filename, &size[0], &size[1]);
		if (data == nullptr) {
			logError("Failed to load PFM image at " + filename);
			return false;
		}
		mFormat = Format::RGBAfloat;
	} else { // formats other than exr...
		data = stbi_load(filename.c_str(), &size[0], &size[1], &channels, STBI_rgb_alpha);
		if (data == nullptr) {
			logError("Failed to load image at " + filename);
			return false;
		}
		mFormat = Format::RGBAuchar;
	}
	stbi_set_flip_vertically_on_load(false);
	int elementSize = getElementSize();
	mSrgb			= srgb;
	if (mData)
		delete[] mData;
	mData = data;
	mSize = size;
	logDebug("Loaded image " + to_string(size[0]) + "*" + to_string(size[1]));
	return true;
}

bool Image::saveImage(const fs::path &filepath) {
	string extension = filepath.extension().string();
	uint nElements	 = mSize[0] * mSize[1] * 4;
	if (extension == ".png") {
		stbi_flip_vertically_on_write(true);
		if (mFormat == Format::RGBAuchar) {
			stbi_write_png(filepath.string().c_str(), mSize[0], mSize[1], 4, mData, 0);
		} else if (mFormat == Format::RGBAfloat) {
			uchar *data			= new uchar[nElements];
			float *internalData = reinterpret_cast<float *>(mData);
			std::transform(internalData, internalData + nElements, data,
						   [](float v) -> uchar { return clamp((int) (v * 255), 0, 255); });
			stbi_write_png(filepath.string().c_str(), mSize[0], mSize[1], 4, data, 0);
			delete[] data;
		}
		return true;
	} else if (extension == ".exr") {
		if (mFormat != Format::RGBAfloat) {
			logError("Image::saveImage Saving non-hdr image as hdr file...");
			return false;
		}
		tinyexr::save_exr(reinterpret_cast<float *>(mData), mSize[0], mSize[1], 4, 4,
						  filepath.string().c_str(), true);
	} else {
		logError("Image::saveImage Unknown image extension: " + extension);
		return false;
	}
	return false;
}

bool Image::isHdr(const string &filepath) {
	return (IsEXR(filepath.c_str()) == TINYEXR_SUCCESS) || stbi_is_hdr(filepath.c_str());
}

Image::SharedPtr Image::createFromFile(const fs::path &filepath, bool flip, bool srgb) {
	Image::SharedPtr pImage = Image::SharedPtr(new Image());
	pImage->loadImage(filepath, flip, srgb);
	return pImage;
}

Texture::SharedPtr Texture::createFromFile(const fs::path &filepath, bool flip, bool srgb) {
	Texture::SharedPtr pTexture = Texture::SharedPtr(new Texture());
	logDebug("Attempting to load texture from " + filepath.string());
	pTexture->loadImage(filepath, flip, srgb);
	return pTexture;
}

Texture::Texture(const string &filepath, bool flip, bool srgb, uint id) : mTextureId{ id } {
	logDebug("Attempting to load texture from " + filepath);
	loadImage(filepath, flip, srgb);
	textureProps[id] = TextureProp{ filepath };
}

void Texture::toDevice() {
	if (!mImage.isValid())
		return;

	// we transfer our texture data to a cuda array, then make the array a cuda texture object.
	Vector2i size	   = mImage.getSize();
	uint numComponents = mImage.getChannels();
	if (numComponents != 4)
		logError("Incorrect texture image channels (not 4)");
	// we have no padding so pitch == width
	uint pitch;
	cudaChannelFormatDesc channelDesc = {};
	Format textureFormat			  = mImage.getFormat();

	if (textureFormat == Format::RGBAfloat) {
		pitch		= size[0] * numComponents * sizeof(float);
		channelDesc = cudaCreateChannelDesc<float4>();
	} else {
		pitch		= size[0] * numComponents * sizeof(uchar);
		channelDesc = cudaCreateChannelDesc<uchar4>();
	}

	// create internal cuda array for texture object
	CUDA_CHECK(cudaMallocArray(&mCudaArray, &channelDesc, size[0], size[1]));
	// transfer data to cuda array
	CUDA_CHECK(cudaMemcpy2DToArray(mCudaArray, 0, 0, mImage.data(), pitch, pitch, size[1],
								   cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc = {};
	resDesc.resType			 = cudaResourceTypeArray;
	resDesc.res.array.array	 = mCudaArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0]	= cudaAddressModeWrap;
	texDesc.addressMode[1]	= cudaAddressModeWrap;
	texDesc.filterMode		= cudaFilterModeLinear;
	texDesc.readMode =
		textureFormat == Format::RGBAfloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords		  = 1;
	texDesc.maxAnisotropy			  = 1;
	texDesc.maxMipmapLevelClamp		  = 99;
	texDesc.minMipmapLevelClamp		  = 0;
	texDesc.mipmapFilterMode		  = cudaFilterModePoint;
	*(Vector4f *) texDesc.borderColor = Vector4f(1.0f);
	texDesc.sRGB					  = (int) mImage.isSrgb();

	cudaTextureObject_t cudaTexture;
	CUDA_CHECK(cudaCreateTextureObject(&cudaTexture, &resDesc, &texDesc, nullptr));

	mCudaTexture = cudaTexture;
}

Material::Material(uint id, const string &name) : mMaterialId(id) {
	materialProps[id] = MaterialProp{ name };
}

void Material::setTexture(TextureType type, Texture &texture) { mTextures[(uint) type] = texture; }

void Material::setConstantTexture(TextureType type, const Color4f color) {
	mTextures[(uint) type].setConstant(color);
}

bool Material::determineSrgb(string filename, TextureType type) {
	if (Image::isHdr(filename))
		return false;
	switch (type) {
		case TextureType::Specular:
			return (mShadingModel == ShadingModel::SpecularGlossiness);
		case TextureType::Diffuse:
		case TextureType::Emissive:
		case TextureType::Transmission:
			return true;
		case TextureType::Normal:
			return false;
	}
	return false;
}
void Material::toDevice() {
	for (uint i = 0; i < (uint) TextureType::Count; i++) {
		mTextures[i].toDevice();
	}
}

void Texture::renderUI() { ui::Text(getFilemame().c_str()); }

void Material::renderUI() {
	static const char *shadingModels[] = { "MetallicRoughness", "SpecularGlossiness" };
	static const char *textureTypes[]  = { "Diffuse", "Specular", "Emissive", "Normal",
										   "Transmission" };
	static const char *bsdfTypes[]	   = { "Diffuse", "Dielectric", "Disney" };
	ui::ListBox("Shading model", (int *) &mShadingModel, shadingModels, 2);
	ui::ListBox("BSDF", (int *) &mBsdfType, bsdfTypes, (int) MaterialType::Count);
	ui::DragFloat4("Diffuse", (float *) &mMaterialParams.diffuse, 1e-3, 0, 5);
	ui::DragFloat4("Specular", (float *) &mMaterialParams.specular, 1e-3, 0, 1);
	ui::DragFloat("Specular transmission", &mMaterialParams.specularTransmission, 1e-3, 0, 1);
	ui::Checkbox("Double sided", &mDoubleSided);
	if (ui::CollapsingHeader("Texture slots")) {
		for (int i = 0; i < (int) TextureType::Count; i++) {
			if (mTextures[i].isValid() && ui::CollapsingHeader(textureTypes[i])) {
				mTextures[i].renderUI();
			}
		}
	}
}

KRR_NAMESPACE_END