// TODO: reorganize and clean up constructors (and "createFromFile" methods)
#pragma once
#include <map>
#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "common.h"

#include "file.h"
#include "raytracing.h"
#include "render/materials/bxdf.h"

KRR_NAMESPACE_BEGIN

class SceneGraph;
 
class Image {
public:
	using SharedPtr = std::shared_ptr<Image>;
	
	enum class Format {
		NONE = 0,
		RGBAuchar,
		RGBAfloat,
	};
	
	Image() = default;
	Image(Vector2i size, Format format = Format::RGBAuchar, bool srgb = false);
	~Image();

	bool loadImage(const fs::path& filepath, bool flip = false, bool srgb = false);
	bool saveImage(const fs::path &filepath, bool flip = false);

	static bool isHdr(const string& filepath);
	static Image::SharedPtr createFromFile(const fs::path& filepath, bool flip = false, bool srgb = false);
	bool isValid() const { return mFormat != Format::NONE && mSize[0] * mSize[1]; }
	bool isSrgb() const { return mSrgb; }
	Vector2i getSize() const { return mSize; }
	Format getFormat() const { return mFormat; }
	inline size_t getElementSize() const { return mFormat == Format::RGBAfloat ? sizeof(float) : sizeof(uchar); }
	int getChannels() const { return mChannels; }
	template <int DIM>
	inline void permuteChannels(const Vector<int, DIM> permutation);
	template <typename F> inline void process(F func);
	size_t getSizeInBytes() const { return mChannels * mSize[0] * mSize[1] * getElementSize(); }
	uchar* data() const { return mData; }
	void reset(uchar *data) { mData = data; }
	
private:
	bool mSrgb{ };
	Vector2i mSize = Vector2i::Zero();
	int mChannels{ 4 };
	Format mFormat{ };
	uchar* mData{ };
};

template <typename F> void Image::process(F func) {
	if (!isValid()) Log(Error, "Load the image before do permutations");
	CHECK_LOG(4 == mChannels, "Only support channel == 4 currently!");
	size_t data_size = getElementSize();
	size_t n_pixels	 = mSize[0] * mSize[1];
	if (data_size == sizeof(float)) {
		using PixelType = Array<float, 4>;
		auto *pixels	= reinterpret_cast<PixelType *>(mData);
		thrust::transform(thrust::host, pixels, pixels + n_pixels, pixels,
						  [=](auto pixel) { return func(pixel); });
	}
	else if (data_size == sizeof(char)) {
		using PixelType = Array<char, 4>;
		auto *pixels	= reinterpret_cast<PixelType *>(mData);
		thrust::transform(thrust::host, pixels, pixels + n_pixels, pixels,
						  [=](auto pixel) { return func(pixel); });
	}
	else Log(Error, "Permute channels not implemented yet :-(");
}

class Texture {
public:
	using SharedPtr = std::shared_ptr<Texture>;
	using Format = Image::Format;

	Texture() = default; 
	Texture(Color4f value) { setConstant(value); }
	Texture(const string& filepath, bool flip = false, bool srgb = false);

	void setConstant(const Color4f value) { 
		mValue = value;
	};

	void loadImage(const fs::path &filepath, bool flip = false, bool srgb = false) {
		mImage = std::make_shared<Image>();
		if(!mImage->loadImage(filepath, flip, srgb) || !mImage->isValid())
			mImage.reset();
	}
	Image::SharedPtr getImage() const { return mImage; }
	Color4f getConstant() const { return mValue; }
	std::string getFilemame() const { return mFilename; }
	bool hasImage() const { return mImage && mImage->isValid(); }

	static Texture::SharedPtr createFromFile(const fs::path &filepath, bool flip = false,
											 bool srgb = false);

	Color4f mValue{};	 /* If this is a constant texture, the value should be set. */
	Image::SharedPtr mImage;
	string mFilename;
};

class Material {
	friend class SceneGraph;
public:
	using SharedPtr = std::shared_ptr<Material>;

	enum class TextureType {
		Diffuse	= 0	,
		Specular	,
		Emissive	,
		Normal		,
		Transmission,
		Count
	};

	enum class ShadingModel {
		MetallicRoughness = 0,
		SpecularGlossiness,
	};

	struct MaterialParams {
		Color4f diffuse{ 1 };			// RGB for base color and A (optional) for opacity 
		Color4f specular{ 0 };			// G-roughness B-metallic A-shininess in MetalRough model
										// RGB - specular color (F0); A - shininess in SpecGloss model
		float IoR{ 1.5f };
		float specularTransmission{ 0 };
	};

	Material() {};
	Material(const string& name);

	void setName(const std::string& name) { mName = name; }
	void setTexture(TextureType type, Texture::SharedPtr texture);
	void setConstantTexture(TextureType type, const Color4f color);
	bool determineSrgb(string filename, TextureType type);

	bool hasEmission();
	bool hasTexture(TextureType type);
	Texture::SharedPtr getTexture(TextureType type) { return mTextures[(uint)type]; }
	
	const string& getName() const { return mName; }
	int getMaterialId() const { return mMaterialId; }
	bool isUpdated() const { return mUpdated; }
	void setUpdated(bool updated = true) { mUpdated = updated; }
	void renderUI();

	MaterialParams mMaterialParams;
	Texture::SharedPtr mTextures[(uint32_t)TextureType::Count];
	MaterialType mBsdfType{ MaterialType::Disney };
	ShadingModel mShadingModel{ ShadingModel::SpecularGlossiness };
	string mName;
	bool mUpdated{false};
	int mMaterialId{-1};
};

namespace rt {
class TextureData {
public:
	Color4f mValue{};
	cudaTextureObject_t mCudaTexture{};
	bool mValid{};

	void initializeFromHost(Texture::SharedPtr texture);

	KRR_CALLABLE bool isValid() const { return mValid; }
	KRR_CALLABLE cudaTextureObject_t getCudaTexture() const {
		return mCudaTexture;
	}
	KRR_CALLABLE Color4f getConstant() const { return mValue; }
	KRR_DEVICE Color4f evaluate(Vector2f uv) const {
#ifdef __NVCC__
		if (mCudaTexture) return tex2D<float4>(mCudaTexture, uv[0], uv[1]);
#endif
		return mValue;
	}
};

class MaterialData {
public:
	Material::MaterialParams mMaterialParams;
	TextureData mTextures[(uint32_t) Material::TextureType::Count];
	MaterialType mBsdfType{MaterialType::Disney};
	Material::ShadingModel mShadingModel{Material::ShadingModel::MetallicRoughness};

	void initializeFromHost(Material::SharedPtr material);

	KRR_CALLABLE TextureData getTexture(Material::TextureType type) const {
		return mTextures[(uint32_t) type];
	}
	void renderUI();
};
}

KRR_NAMESPACE_END