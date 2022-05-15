#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define TINYEXR_USE_MINIZ (0)
#define TINYEXR_IMPLEMENTATION
#include "zlib.h"			// needed by tinyexr
#include "stb_image.h"
#include "stb_image_write.h"
#include "tinyexr.h"

#include <filesystem>

#include "texture.h"
#include "window.h"
#include "logger.h"
#include "device/optix.h"

namespace tinyexr {
	void save_exr(const float* data, int width, int height, int nChannels, int channelStride, const char* outfilename, bool flip = true) {
		EXRHeader header;
		InitEXRHeader(&header);

		EXRImage image;
		InitEXRImage(&image);

		image.num_channels = nChannels;

		std::vector<std::vector<float>> images(nChannels);
		std::vector<float*> image_ptr(nChannels);
		for (int i = 0; i < nChannels; ++i) {
			images[i].resize((size_t)width * (size_t)height);
		}

		for (int i = 0; i < nChannels; ++i) {
			image_ptr[i] = images[nChannels - i - 1].data();
		}

		for (int c = 0; c < nChannels; c++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					// whether flip vertically? 
					int ry = flip ? height - 1 - y : y;
					int idx = y * width + x;
					int ridx = ry * width + x;
					images[c][ridx] = data[channelStride * idx + c];
				}
			}
		}

		image.images = (unsigned char**)image_ptr.data();
		image.width = width;
		image.height = height;

		header.line_order = 1;
		header.num_channels = nChannels;
		header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
		// Must be (A)BGR order, since most of EXR viewers expect this channel order.
		strncpy(header.channels[0].name, "B", 255); header.channels[0].name[strlen("B")] = '\0';
		if (nChannels > 1) {
			strncpy(header.channels[1].name, "G", 255); header.channels[1].name[strlen("G")] = '\0';
		}
		if (nChannels > 2) {
			strncpy(header.channels[2].name, "R", 255); header.channels[2].name[strlen("R")] = '\0';
		}
		if (nChannels > 3) {
			strncpy(header.channels[3].name, "A", 255); header.channels[3].name[strlen("A")] = '\0';
		}

		header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
		for (int i = 0; i < header.num_channels; i++) {
			header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
			header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF; // pixel type of output image to be stored in .EXR
		}

		const char* err = NULL; // or nullptr in C++11 or later.
		int ret = SaveEXRImageToFile(&image, &header, outfilename, &err);
		if (ret != TINYEXR_SUCCESS) {
			std::string error_message = std::string("Failed to save EXR image: ") + err;
			FreeEXRErrorMessage(err); // free's buffer for an error message
			throw std::runtime_error(error_message);
		}
		printf("Saved exr file. [ %s ] \n", outfilename);

		free(header.channels);
		free(header.pixel_types);
		free(header.requested_pixel_types);
	}
}

KRR_NAMESPACE_BEGIN

namespace texture {
	std::map<uint, TextureProp> textureProps;
	std::map<uint, MaterialProp> materialProps;
}

using namespace texture;

Image::Image(vec2i size, Format format, bool srgb):
	mSrgb(srgb), mFormat(format), mSize(size){
	mData = new uchar[size.x * size.y * 4 * getElementSize()];
}

bool Image::loadImage(const fs::path& filepath, bool srgb){
	vec2i size;
	int channels;
	string filename = filepath.string();
	uchar* data = nullptr;
	if (IsEXR(filename.c_str()) == TINYEXR_SUCCESS) {
		char* errMsg = nullptr;
		// to do: if loadEXR always return RGBA data?
		int res = LoadEXR((float**)&data, &size.x, &size.y, filename.c_str(), (const char**)&errMsg);
		if (res != TINYEXR_SUCCESS) {
			logError("Failed to load EXR image at " + filename);
			if (errMsg) logError(errMsg);
			return false;
		}
		mFormat = Format::RGBAfloat;
	}
	else if (stbi_is_hdr(filename.c_str())){
		data = (uchar*)stbi_loadf(filename.c_str(), &size.x, &size.y, &channels, STBI_rgb_alpha);
		if (data == nullptr) {
			logError("Failed to load float hdr image at " + filename);
			return false;
		}
		mFormat = Format::RGBAfloat;
	}
	else {	// formats other than exr...
		data = stbi_load(filename.c_str(), &size.x, &size.y, &channels, STBI_rgb_alpha);
		if (data == nullptr) {
			logError("Failed to load image at " + filename);
			return false;
		}
		mFormat = Format::RGBAuchar;
	}
	int elementSize = getElementSize();
	mSrgb = srgb;
	if (mData) delete[] mData;
	mData = data;

	mSize = size;
	logDebug("Loaded image " + to_string(size.x) + "*" + to_string(size.y));
	return true;
}

bool Image::saveImage(const fs::path& filepath){
	string extension = filepath.extension().string();
	uint nElements = mSize.x * mSize.y * 4;
	if (extension == ".png") {
		stbi_flip_vertically_on_write(true);
		if (mFormat == Format::RGBAuchar) {
			stbi_write_png(filepath.string().c_str(), mSize.x, mSize.y, 4, mData, 0);
		}
		else if (mFormat == Format::RGBAfloat) {
			uchar* data = new uchar[nElements];
			float* internalData = reinterpret_cast<float*>(mData);
			std::transform(internalData, internalData + nElements, data,
				[](float v) -> uchar { return math::clamp((int)(v * 255), 0, 255); });
			stbi_write_png(filepath.string().c_str(), mSize.x, mSize.y, 4, data, 0);
			delete[] data;
		}
		return true;
	}
	else if (extension == ".exr") {
		if (mFormat != Format::RGBAfloat) {
			logError("Image::saveImage Saving non-hdr image as hdr file...");
			return false;
		}
		tinyexr::save_exr(reinterpret_cast<float*>(mData), mSize.x, mSize.y, 4, 4, filepath.string().c_str());
	}
	else {
		logError("Image::saveImage Unknown image extension: " + extension);
		return false;
	}
	return false;
}

bool Image::isHdr(const string& filepath){
	return (IsEXR(filepath.c_str()) == TINYEXR_SUCCESS) ||
		stbi_is_hdr(filepath.c_str());
}

Image::SharedPtr Image::createFromFile(const string& filepath, bool srgb){
	Image::SharedPtr pImage = Image::SharedPtr(new Image());
	pImage->loadImage(filepath);
	return pImage;
}

Texture::SharedPtr Texture::createFromFile(const string& filepath, bool srgb){
	Texture::SharedPtr pTexture = Texture::SharedPtr(new Texture());
	logDebug("Attempting to load texture from " + filepath);
	pTexture->loadImage(filepath);
	return pTexture;
}

Texture::Texture(const string& filepath, bool srgb, uint id)
	:mTextureId{id} {
	logDebug("Attempting to load texture from " + filepath);
	loadImage(filepath, srgb);
	textureProps[id] = TextureProp{ filepath };
}

void Texture::toDevice() {
	if (!mImage.isValid()) return;

	// we transfer our texture data to a cuda array, then make the array a cuda texture object.
	vec2i size = mImage.getSize();
	uint numComponents = mImage.getChannels();
	if (numComponents != 4)
		logError("Incorrect texture image channels (not 4)");
	// we have no padding so pitch == width
	uint pitch;
	cudaChannelFormatDesc channelDesc = {};
	Format textureFormat = mImage.getFormat();

	if (textureFormat == Format::RGBAfloat) {
		pitch = size.x * numComponents * sizeof(float);
		channelDesc = cudaCreateChannelDesc<float4>();
	}
	else {
		pitch = size.x * numComponents * sizeof(uchar);
		channelDesc = cudaCreateChannelDesc<uchar4>();
	}
	
	// create internal cuda array for texture object
	CUDA_CHECK(cudaMallocArray(&mCudaArray, &channelDesc, size.x, size.y));
	// transfer data to cuda array
	CUDA_CHECK(cudaMemcpy2DToArray(mCudaArray, 0, 0, mImage.data(), pitch, pitch, size.y, cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = mCudaArray;

	cudaTextureDesc texDesc = {};
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = textureFormat == Format::RGBAfloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;
	texDesc.maxAnisotropy = 1;
	texDesc.maxMipmapLevelClamp = 99;
	texDesc.minMipmapLevelClamp = 0;
	texDesc.mipmapFilterMode = cudaFilterModePoint;
	*(vec4f*)texDesc.borderColor = vec4f(1.0f);
	texDesc.sRGB = (int)mImage.isSrgb();

	cudaTextureObject_t cudaTexture;
	CUDA_CHECK(cudaCreateTextureObject(&cudaTexture, &resDesc, &texDesc, nullptr));

	mCudaTexture = cudaTexture;
}

Material::Material(uint id, const string& name) :mMaterialId(id) {
	materialProps[id] = MaterialProp{ name };
}

void Material::setTexture(TextureType type, Texture& texture) {
	mTextures[(uint)type] = texture;
}

bool Material::determineSrgb(string filename, TextureType type) {
	if (Image::isHdr(filename)) return false;
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
	for (uint i = 0; i < (uint)TextureType::Count; i++) {
		mTextures[i].toDevice();
	}
}

void Texture::renderUI() {
	ui::Text(getFilemame().c_str());
}

void Material::renderUI() {
	static const char* shadingModels[] = { "MetallicRoughness", "SpecularGlossiness"};
	static const char* textureTypes[] = { "Diffuse", "Specular", "Emissive", "Normal", "Transmission" };
	static const char* bsdfTypes[] = {"Diffuse", "FresnelBlend", "Disney"};
	ui::ListBox("Shading model", (int*)&mShadingModel, shadingModels, 2);
	ui::ListBox("BSDF", (int*)&mBsdfType, bsdfTypes, (int)BsdfType::Count);
	ui::InputFloat4("Diffuse", (float*)&mMaterialParams.diffuse);
	ui::InputFloat4("Specular", (float*)&mMaterialParams.specular);
	ui::InputFloat3("Emissive", (float*)&mMaterialParams.emissive);
	ui::InputFloat("Diffuse transmission", &mMaterialParams.diffuseTransmission);
	ui::InputFloat("Specular transmission", &mMaterialParams.specularTransmission);
	ui::Checkbox("Double sided", &mDoubleSided);
	if (ui::CollapsingHeader("Texture slots")) {
		for (int i = 0; i < (int)TextureType::Count; i++) {
			if (mTextures[i].isValid() && ui::CollapsingHeader(textureTypes[i])) {
				mTextures[i].renderUI();
			}
		}
	}
}

KRR_NAMESPACE_END