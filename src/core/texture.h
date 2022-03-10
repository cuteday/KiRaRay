// we want a image->texture->material class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "math/math.h"
#include "file.h"
#include "render/materials/bxdf.h"

KRR_NAMESPACE_BEGIN
 
class Image {	// Not available on GPU
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
		mValid = mImage->loadImage(filepath);

	}
	__both__ bool isValid() const { return mValid; }
	__both__ bool isOnDevice() const { return mCudaTexture != 0; }
	
	__both__ cudaTextureObject_t getCudaTexture()const { return mCudaTexture; }
	
	__device__ vec3f tex(vec2f uv)const {
#ifdef __NVCC__ 
		DCHECK(mCudaTexture);
		vec3f color = (vec3f)tex2D<float4>(mCudaTexture, uv.x, uv.y);
		return color;
#endif 
		return 0;
	}

	void toDevice();

	static Texture::SharedPtr createFromFile(const string& filepath, bool srgb = false);

//private:
	bool mValid = false;
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
		Microfacet,
		FresnelBlend,
		Uniform,
		Disney,
	};

	enum class ShadingModel {
		MetallicRoughness = 0,
		SpecularGlossiness,
	};

	struct MaterialParams {
		vec4f diffuse{ 0 };			 
		vec4f specular{ 0 };			// G-roughness B-metallic in MetalRough model
		vec3f emissive{ 0 };
		float IoR{ 1.5f };
		float diffuseTransmission{ 0 };
		float specularTransmission{ 0 };
	};

	Material() {};
	Material(const string& name) { mName = name; }

	void setTexture(TextureType type, Texture& texture);

	bool hasEmission() { 
		return any(mMaterialParams.emissive) || 
			mTextures[(int)TextureType::Emissive].isValid(); 
	}

	__both__ Texture getTexture(TextureType type) {
		return mTextures[(uint)type];
	}

	__both__ cudaTextureObject_t getCudaTexture(TextureType type) {
		return mTextures[(uint)type].getCudaTexture();
	}

	void toDevice();

	friend class AssimpImporter;
	MaterialParams mMaterialParams;
	Texture mTextures[5];
#if KRR_USE_DISNEY
	BsdfType mBsdfType = BsdfType::Disney;
#elif KRR_USE_FRESNEL_BLEND
	BsdfType mBsdfType = BsdfType::FresnelBlend;
#else
	BsdfType mBsdfType = BsdfType::Diffuse;
#endif
	ShadingModel mShadingModel = ShadingModel::MetallicRoughness;
	bool mDoubleSided = false;
	string mName;
};

KRR_NAMESPACE_END