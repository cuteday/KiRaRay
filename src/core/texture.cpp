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
#include "graphics/ui.h"
#include "logger.h"
#include "util/image.h"
#include "util/exr.h"
#include "device/buffer.h"

NAMESPACE_BEGIN(krr)

Image::Image(Vector2i size, Format format, bool srgb) : 
	mSrgb(srgb), mFormat(format), mSize(size) {
	mData = new uchar[size[0] * size[1] * 4 * getElementSize()];
}

Image::~Image() { if (mData) delete[] mData; }

bool Image::loadImage(const fs::path &filepath, bool flip, bool srgb) {
	Vector2i size;
	int channels;
	string filename = File::resolve(filepath).string();
	string format	= filepath.extension().string();
	uchar *data		= nullptr;
	bool pfmData = false;
	
	if (filename.find("$") != string::npos) {
		/* special built-in textures */
		auto pos = filename.find("$") + 1;
		filename = (File::textureDir() / filename.substr(pos, filename.size() - pos)).string();
	}

	stbi_set_flip_vertically_on_load(flip);
	struct ResetFlip { ~ResetFlip() { stbi_set_flip_vertically_on_load(false); } } resetFlip;
	if (IsEXR(filename.c_str()) == TINYEXR_SUCCESS) {
		try {
			exr::load(filename.c_str(), (float **) &data, &size[0], &size[1], flip);
		} catch (const std::exception &error) {
			logError("Failed to load EXR image at " + filename + ": " + error.what());
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
		pfmData = true;
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
	int elementSize = getElementSize();
	mSrgb			= srgb;
	std::unique_ptr<uchar, void (*)(uchar *)> source(data, pfmData ?
		+[](uchar *p) { delete[] reinterpret_cast<RGBA *>(p); } : +[](uchar *p) { free(p); });
	auto owned = std::make_unique<uchar[]>(size_t(size[0]) * size[1] * 4 * elementSize);
	memcpy(owned.get(), data, size_t(size[0]) * size[1] * 4 * elementSize);
	if (mData)
		delete[] mData;
	mData = owned.release();
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

namespace {
MaterialType authoredMaterialType(MaterialModel model) {
	if (model == MaterialModel::PreviewSurface) return MaterialType::PreviewSurface;
	if (model == MaterialModel::OpenPBR || model == MaterialModel::Error) return MaterialType::OpenPBR;
	return MaterialType::Composite;
}
}

void Material::setDescription(std::shared_ptr<MaterialDescription> description) {
	if (description) {
		auto compiled = compileMaterial(*description);
		mHasProgramEmission = compiled.hasEmission;
		mBsdfType = authoredMaterialType(compiled.model);
	} else {
		mHasProgramEmission = false;
		if (mDescription) mBsdfType = MaterialType::Disney;
	}
	mDescription = std::move(description);
	if (getNode()) setUpdated();
	else mUpdated = true;
}

void Material::updateFrom(const Material &material) {
	mMaterialParams = material.mMaterialParams;
	for (int index = 0; index < int(TextureType::Count); ++index) mTextures[index] = material.mTextures[index];
	mBsdfType = material.mBsdfType;
	mShadingModel = material.mShadingModel;
	mColorSpace = material.mColorSpace;
	mName = material.mName;
	mDescription = material.mDescription;
	mHasProgramEmission = material.mHasProgramEmission;
	if (getNode()) setUpdated();
	else mUpdated = true;
}

void Material::setTexture(TextureType type, Texture::SharedPtr texture) { 
	mTextures[(uint) type] = texture; 
}

void Material::setConstantTexture(TextureType type, const RGBA color) {
	if (!mTextures[(uint) type]) mTextures[(uint) type] = std::make_shared<Texture>();
	mTextures[(uint) type]->setConstant(color);
}

bool Material::hasEmission() {
	return mDescription ? mHasProgramEmission : hasTexture(TextureType::Emissive);
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
	if (mDescription) {
		ui::Text(materialModelName(mDescription->model));
		return;
	}
	static const char *shadingModels[] = {"MetallicRoughness", "SpecularGlossiness"};
	static const char *textureTypes[]  = {"Diffuse", "Specular", "Emissive", "Normal",
										  "Transmission"};
	static const char *bsdfTypes[]	   = {"Null", "Diffuse", "Dielectric", "Conductor", "Disney", "OpenPBR", "Preview Surface"};
	bool updated					   = false;
	updated |= ui::ListBox("Shading model", (int *) &mShadingModel, shadingModels, 2);
	updated |= ui::ListBox("BSDF", (int *) &mBsdfType, bsdfTypes, (int) MaterialType::Disney + 1);
	updated |= ui::DragFloat4("Diffuse", (float *) &mMaterialParams.diffuse, 1e-3, 0, 1);
	updated |= ui::DragFloat4("Specular", (float *) &mMaterialParams.specular, 1e-3, 0, 1);
	updated |= ui::DragFloat("Transmission", &mMaterialParams.specularTransmission, 1e-3, 0, 1);
	updated |= ui::DragFloat("Anisotropy", &mMaterialParams.anisotropic, 1e-3, 0, 1);
	if (mMaterialParams.spectralEta) ui::Text("Has Spectral Eta");
	else updated |= ui::InputFloat("IoR", &mMaterialParams.IoR);
	if (mMaterialParams.spectralK) ui::Text("Has Spectral K");
	setUpdated(updated);
}

namespace rt {

void TextureData::release() noexcept {
	if (mCudaTexture) cudaDestroyTextureObject(mCudaTexture);
	if (mCudaArray) cudaFreeArray(mCudaArray);
	mCudaTexture = 0;
	mCudaArray = nullptr;
	mValid = false;
	mSize = {0, 0};
}

void TextureData::initializeFromHost(Texture::SharedPtr texture, const MaterialTexture *sampling) {
	mValid = texture.get() != nullptr;
	if (!texture) return;

	mValue = texture->getConstant();
	mFilter = sampling ? sampling->filter : MaterialFilter::Linear;

	if (!texture->getImage() || !texture->getImage()->isValid()) return;
	auto image = texture->getImage();

	// we transfer our texture data to a cuda array, then make the array a cuda
	// texture object.
	Vector2i size	   = image->getSize();
	mSize = size;
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

	// create internal cuda array for texture object
	CUDA_CHECK(cudaMallocArray(&mCudaArray, &channelDesc, size[0], size[1]));
	std::vector<RGBA> linearPixels;
	const void *pixels = image->data();
	bool decodeFloatSrgb = sampling && textureFormat == Image::Format::RGBAfloat && image->isSrgb();
	if (decodeFloatSrgb) {
		const auto *source = reinterpret_cast<const RGBA *>(image->data());
		linearPixels.assign(source, source + size_t(size[0]) * size[1]);
		for (RGBA &pixel : linearPixels) for (int channel = 0; channel < 3; ++channel) {
			float value = pixel[channel];
			pixel[channel] = value <= .04045f ? value / 12.92f : powf((value + .055f) / 1.055f, 2.4f);
		}
		pixels = linearPixels.data();
	}
	// transfer data to cuda array
	CUDA_CHECK(cudaMemcpy2DToArray(mCudaArray, 0, 0, pixels, pitch,
								   pitch, size[1], cudaMemcpyHostToDevice));

	cudaResourceDesc resDesc = {};
	resDesc.resType			 = cudaResourceTypeArray;
	resDesc.res.array.array	 = mCudaArray;

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
	texDesc.sRGB					  = image->isSrgb() && !decodeFloatSrgb;
	if (sampling) {
		auto address = [](MaterialWrap wrap) {
			switch (wrap) {
				case MaterialWrap::Clamp: return cudaAddressModeClamp;
				case MaterialWrap::Mirror: return cudaAddressModeMirror;
				case MaterialWrap::Border: return cudaAddressModeBorder;
				default: return cudaAddressModeWrap;
			}
		};
		texDesc.addressMode[0] = address(sampling->wrapU);
		texDesc.addressMode[1] = address(sampling->wrapV);
		texDesc.filterMode = sampling->filter == MaterialFilter::Linear ? cudaFilterModeLinear : cudaFilterModePoint;
		for (int index = 0; index < 4; ++index) texDesc.borderColor[index] = sampling->fallback[index];
	}

	CUDA_CHECK(cudaCreateTextureObject(&mCudaTexture, &resDesc, &texDesc, nullptr));
}

class MaterialTextureStorage {
public:
	~MaterialTextureStorage() {
		for (auto &texture : textures) texture.release();
	}

	TextureData add(const MaterialTexture &binding) {
		for (size_t index = 0; index < bindings.size(); ++index) {
			const auto &other = bindings[index];
			bool same = binding.texture == other.texture && binding.wrapU == other.wrapU &&
				binding.wrapV == other.wrapV && binding.filter == other.filter;
			for (int channel = 0; channel < 4; ++channel)
				same &= binding.fallback[channel] == other.fallback[channel];
			if (same) return textures[index];
		}
		if (!binding.texture->hasImage()) throw std::invalid_argument("Material image failed to load");
		bindings.push_back(binding);
		textures.emplace_back();
		textures.back().initializeFromHost(binding.texture, &binding);
		return textures.back();
	}

private:
	std::vector<MaterialTexture> bindings;
	std::vector<TextureData> textures;
};

class MaterialProgramStorage {
public:
	~MaterialProgramStorage() {
		clear(components);
		componentStorage.clear();
		clear(surface); clear(opacity); clear(emission); clear(classification); clear(weight);
		clear(uniforms); clear(textures); clear(simple);
	}

	MaterialProgramData initialize(const CompiledMaterial &compiled,
		std::shared_ptr<MaterialTextureStorage> sharedTextures = {}) {
		textureStorage = sharedTextures ? std::move(sharedTextures) : std::make_shared<MaterialTextureStorage>();
		MaterialProgramData program;
		program.enabled = true;
		program.model = compiled.model;
		program.kind = compiled.kind;
		program.defaults = compiled.defaults;
		program.authoredMask = compiled.authoredMask;
		std::vector<TextureData> hostTextures;
		for (const auto &texture : compiled.textures) hostTextures.push_back(textureStorage->add(texture));
		textures.alloc_and_copy_from_host(hostTextures);
		surface.alloc_and_copy_from_host(compiled.surface);
		opacity.alloc_and_copy_from_host(compiled.opacity);
		emission.alloc_and_copy_from_host(compiled.emission);
		classification.alloc_and_copy_from_host(compiled.classification);
		weight.alloc_and_copy_from_host(compiled.weight);
		uniforms.alloc_and_copy_from_host(compiled.uniforms);
		simple.alloc_and_copy_from_host(compiled.simple);
		program.surface = {surface.data(), uint32_t(surface.size())};
		program.opacity = {opacity.data(), uint32_t(opacity.size())};
		program.emission = {emission.data(), uint32_t(emission.size())};
		program.classification = {classification.data(), uint32_t(classification.size())};
		program.weight = {weight.data(), uint32_t(weight.size())};
		program.uniforms = uniforms.data();
		program.textures = textures.data();
		program.simple = simple.data();
		program.simpleCount = uint32_t(simple.size());
		std::vector<MaterialProgramData> hostComponents;
		for (const auto &component : compiled.components) {
			auto storage = std::make_unique<MaterialProgramStorage>();
			hostComponents.push_back(storage->initialize(component, textureStorage));
			componentStorage.push_back(std::move(storage));
		}
		components.alloc_and_copy_from_host(hostComponents);
		program.components = components.data();
		program.componentCount = uint32_t(components.size());
		return program;
	}

private:
	template <typename T> static void clear(TypedBuffer<T> &buffer) noexcept {
		try { buffer.clear(); } catch (...) {}
	}
	std::shared_ptr<MaterialTextureStorage> textureStorage;
	std::vector<std::unique_ptr<MaterialProgramStorage>> componentStorage;
	TypedBuffer<MaterialProgramData> components;
	TypedBuffer<MaterialInstruction> surface, opacity, emission, classification, weight;
	TypedBuffer<MaterialValue> uniforms;
	TypedBuffer<TextureData> textures;
	TypedBuffer<MaterialSimpleBinding> simple;
};

void MaterialData::releaseProgram() noexcept {
	delete mProgramStorage;
	mProgramStorage = nullptr;
	mProgram = {};
}

void MaterialData::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, 
	bool initialize) const {
	auto material = std::dynamic_pointer_cast<Material>(object);
	auto gdata = reinterpret_cast<MaterialData *>(data->data());
	if (!material->mDescription && material->mBsdfType >= MaterialType::OpenPBR)
		throw std::invalid_argument("Material '" + material->getName() + "' requires an authored material description");
	if (!initialize) CUDA_CHECK(cudaDeviceSynchronize());
	for (size_t tex_idx = 0; tex_idx < (size_t) Material::TextureType::Count; tex_idx++) {
		if (!initialize) gdata->mTextures[tex_idx].release();
		gdata->mTextures[tex_idx].initializeFromHost(material->mTextures[tex_idx]);
	}
	if (material->mDescription) {
		auto compiled = compileMaterial(*material->mDescription);
		auto storage = std::make_unique<MaterialProgramStorage>();
		auto program = storage->initialize(compiled);
		if (!initialize) CUDA_CHECK(cudaDeviceSynchronize());
		gdata->releaseProgram();
		gdata->mProgram = program;
		gdata->mProgramStorage = storage.release();
	} else if (gdata->mProgramStorage) {
		if (!initialize) CUDA_CHECK(cudaDeviceSynchronize());
		gdata->releaseProgram();
	}
	gdata->mBsdfType = material->mDescription ? authoredMaterialType(gdata->mProgram.model) : material->mBsdfType;
	gdata->mMaterialParams = material->mMaterialParams;
	gdata->mShadingModel   = material->mShadingModel;
	gdata->mColorSpace	   = material->getColorSpace();
}

} // namespace rt

NAMESPACE_END(krr)
