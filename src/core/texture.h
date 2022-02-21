// we want a image->texture->material class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "math/math.h"
#include "file.h"

KRR_NAMESPACE_BEGIN
 
// Image class is opaque to gpu.
class Image {
public:
	using SharedPtr = std::shared_ptr<Image>;
	
	enum class Format {
		NONE = 0,
		RGBAuchar,
		RGBAfloat,
	};

	
	Image() {};
	Image(vec2i size, Format format = Format::RGBAuchar, bool srgb = false);
	~Image() {}

	bool loadImage(const fs::path& filepath, bool srgb = false);
	bool saveImage(const fs::path& filepath);

	static Image::SharedPtr createFromFile(const string& filepath, bool srgb = false);
	bool isValid() const { return mFormat != Format::NONE && mSize.x * mSize.y; }
	bool isSrgb() const { return mSrgb; }
	vec2i getSize() const { return mSize; }
	Format getFormat() const { return mFormat; }
	inline size_t getElementSize() const { return mFormat == Format::RGBAfloat ? sizeof(float) : sizeof(uchar); }
	int getChannels() const { return mChannels; }
	uchar* data() { return mData.data(); }

	
private:
	bool mSrgb = false;
	vec2i mSize = {0, 0};
	int mChannels = 4;
	Format mFormat = Format::NONE;
	std::vector<uchar> mData;
};

class Texture {
public:
	using SharedPtr = std::shared_ptr<Texture>;
	using Format = Image::Format;

	Texture() {
		mImage = Image::SharedPtr(new Image());
	};

	void loadImage(const string& filepath, bool srgb = false) {
		mImage->loadImage(filepath);
	}
	__both__ bool isValid() { return mCudaTexture != 0; }
	__both__ cudaTextureObject_t getCudaTexture() { return mCudaTexture; }
	void toDevice();

	static Texture::SharedPtr createFromFile(const string& filepath, bool srgb = false);

//private:
	Image::SharedPtr mImage;
	cudaTextureObject_t mCudaTexture = 0;
	cudaArray_t mCudaArray = 0;
};

class Material {
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

	enum class BsdfType {
		Diffuse		= 0,
		Microfacet	,
		FresnelBlended
	};

	enum class ShadingModel {
		MetallicRoughness = 0,
		SpecularGlossiness,
	};

	struct MaterialParams {
		vec4f diffuse = vec4f(1);		// 
		vec4f specular = vec4f(0);		// G-roughness B-metallic in MetalRough model
		vec3f emissive = vec3f(0);
		float IoR = 1.5f;
		vec3f transmission = vec3f(1);
	};

	Material() {};
	Material(const string& name) { mName = name; }

	void setTexture(TextureType type, Texture& texture);

	__both__ cudaTextureObject_t getCudaTexture(TextureType type) {
		return mTextures[(uint)type].getCudaTexture();
	}

	void toDevice();

//private:
	friend class AssimpImporter;

	MaterialParams mMaterialParams;
	Texture mTextures[5];
	BsdfType mBsdfType = BsdfType::FresnelBlended;
	ShadingModel mShadingModel = ShadingModel::MetallicRoughness;
	bool mDoubleSided = false;
	string mName;
};



KRR_NAMESPACE_END