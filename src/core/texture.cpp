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

Image::Image(Vector2i size, Format format, bool srgb) : 
	mSrgb(srgb), mFormat(format), mSize(size) {
	mData = new uchar[size[0] * size[1] * 4 * getElementSize()];
}

Image::~Image() { if (mData) delete[] mData; }

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

bool Image::saveImage(const fs::path &filepath, bool flip) {
	string extension = filepath.extension().string();
	uint nElements	 = mSize[0] * mSize[1] * 4;
	if (extension == ".png") {
		stbi_flip_vertically_on_write(flip);
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
		stbi_flip_vertically_on_write(false);
		return true;
	} else if (extension == ".exr") {
		if (mFormat != Format::RGBAfloat) {
			logError("Image::saveImage Saving non-hdr image as hdr file...");
			return false;
		}
		tinyexr::save_exr(reinterpret_cast<float *>(mData), mSize[0], mSize[1], 4, 4,
						  filepath.string().c_str(), flip);
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

Texture::Texture(const string &filepath, bool flip, bool srgb) : mFilename(filepath) {
	logDebug("Attempting to load texture from " + filepath);
	loadImage(filepath, flip, srgb);
}

Material::Material(const string &name) : mName(name) {}

void Material::setTexture(TextureType type, Texture::SharedPtr texture) { 
	mTextures[(uint) type] = texture; 
}

void Material::setConstantTexture(TextureType type, const Color4f color) {
	if (!mTextures[(uint) type]) mTextures[(uint) type] = std::make_shared<Texture>();
	mTextures[(uint) type]->setConstant(color);
}

bool Material::hasEmission() {
	return hasTexture(TextureType::Emissive);
}

bool Material::hasTexture(TextureType type) {
	return mTextures[(int) type].get() != nullptr; 
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

void Material::renderUI() {
	static const char *shadingModels[] = {"MetallicRoughness", "SpecularGlossiness"};
	static const char *textureTypes[]  = {"Diffuse", "Specular", "Emissive", "Normal",
										  "Transmission"};
	static const char *bsdfTypes[]	   = {"Diffuse", "Dielectric", "Disney"};
	mUpdated |= ui::ListBox("Shading model", (int *) &mShadingModel, shadingModels, 2);
	mUpdated |= ui::ListBox("BSDF", (int *) &mBsdfType, bsdfTypes, (int) MaterialType::Count);
	mUpdated |= ui::DragFloat4("Diffuse", (float *) &mMaterialParams.diffuse, 1e-3, 0, 1);
	mUpdated |= ui::DragFloat4("Specular", (float *) &mMaterialParams.specular, 1e-3, 0, 1);
	mUpdated |= ui::DragFloat("Specular transmission", &mMaterialParams.specularTransmission, 1e-3, 0, 1);
	mUpdated |= ui::InputFloat("Index of Refraction", &mMaterialParams.IoR);
}

namespace rt {

void TextureData::initializeFromHost(Texture::SharedPtr texture) {
	mValid = texture.get() != nullptr;
	if (!texture) return;

	mValue = texture->getConstant();

	if (!texture->getImage() || !texture->getImage()->isValid()) return;
	auto image = texture->getImage();

	// we transfer our texture data to a cuda array, then make the array a cuda
	// texture object.
	Vector2i size	   = image->getSize();
	uint numComponents = image->getChannels();
	if (numComponents != 4)
		logError("Incorrect texture image channels (not 4)");
	// we have no padding so pitch == width
	uint pitch;
	cudaChannelFormatDesc channelDesc = {};
	Image::Format textureFormat		  = image->getFormat();

	if (textureFormat == Image::Format::RGBAfloat) {
		pitch		= size[0] * numComponents * sizeof(float);
		channelDesc = cudaCreateChannelDesc<float4>();
	} else {
		pitch		= size[0] * numComponents * sizeof(uchar);
		channelDesc = cudaCreateChannelDesc<uchar4>();
	}

	cudaArray_t cudaArray;
	// create internal cuda array for texture object
	CUDA_CHECK(cudaMallocArray(&cudaArray, &channelDesc, size[0], size[1]));
	// transfer data to cuda array
	CUDA_CHECK(cudaMemcpy2DToArray(cudaArray, 0, 0, (void*)image->data(), pitch,
								   pitch, size[1], cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc = {};
	resDesc.resType			 = cudaResourceTypeArray;
	resDesc.res.array.array	 = cudaArray;

	cudaTextureDesc texDesc			  = {};
	texDesc.addressMode[0]			  = cudaAddressModeWrap;
	texDesc.addressMode[1]			  = cudaAddressModeWrap;
	texDesc.filterMode				  = cudaFilterModeLinear;
	texDesc.readMode				  = textureFormat == Image::Format::RGBAfloat
											? cudaReadModeElementType
											: cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords		  = 1;
	texDesc.maxAnisotropy			  = 1;
	texDesc.maxMipmapLevelClamp		  = 99;
	texDesc.minMipmapLevelClamp		  = 0;
	texDesc.mipmapFilterMode		  = cudaFilterModePoint;
	*(Vector4f *) texDesc.borderColor = Vector4f(1.0f);
	texDesc.sRGB					  = (int) image->isSrgb();

	CUDA_CHECK(cudaCreateTextureObject(&mCudaTexture, &resDesc, &texDesc, nullptr));
}

void MaterialData::initializeFromHost(Material::SharedPtr material) {
	mBsdfType		= material->mBsdfType;
	mMaterialParams = material->mMaterialParams;
	mShadingModel	= material->mShadingModel;
	for (size_t tex_idx = 0; tex_idx < (size_t)Material::TextureType::Count;
		 tex_idx++) {
		mTextures[tex_idx].initializeFromHost(material->mTextures[tex_idx]);
	}
}

} // namespace rt

KRR_NAMESPACE_END