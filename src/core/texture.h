// we want a image->texture->material class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

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

namespace texture {
	typedef struct{
		string path;
	} TextureProp;
	typedef struct {
		string name;
	} MaterialProp;
	extern std::map<uint, TextureProp> textureProps;
	extern std::map<uint, MaterialProp> materialProps;
}
 
class Image {	// Not available on GPU
public:
	using SharedPtr = std::shared_ptr<Image>;
	
	enum class Format {
		NONE = 0,
		RGBAuchar,
		RGBAfloat,
	};
	
	Image() {};
	Image(Vector2i size, Format format = Format::RGBAuchar, bool srgb = false);
	~Image() {}

	bool loadImage(const fs::path& filepath, bool flip = false, bool srgb = false);
	bool saveImage(const fs::path& filepath);

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
	size_t getSizeInBytes() const { return mChannels * mSize[0] * mSize[1] * getElementSize(); }
	uchar* data() { return mData; }
	void reset(uchar *data) { mData = data; }
	
private:
	bool mSrgb{ };
	Vector2i mSize = Vector2i::Zero();
	int mChannels{ 4 };
	Format mFormat{ };
	uchar* mData{ };
};

template <int DIM> void Image::permuteChannels(const Vector<int, DIM> permutation) {
	if (!isValid())
		Log(Error, "Load the image before do permutations");
	CHECK_LOG(4 == mChannels, "Only support channel == 4 currently!");
	CHECK_LOG(DIM <= mChannels, "Permutation do not match channel count!");
	size_t data_size = getElementSize();
	size_t n_pixels	 = mSize[0] * mSize[1];
	if (data_size == sizeof(float)) {
		using PixelType = Array<float, 4>;
		auto *pixels	= reinterpret_cast<PixelType *>(mData);
		thrust::transform(thrust::host, pixels, pixels + n_pixels, pixels, [=](PixelType pixel) {
			PixelType res = pixel;
			for (int c = 0; c < DIM; c++) {
				res[c] = pixel[permutation[c]];
			}
			return res;
		});
	} else if (data_size == sizeof(char)) {
		using PixelType = Array<char, 4>;
		auto *pixels	= reinterpret_cast<PixelType *>(mData);
		thrust::transform(thrust::host, pixels, pixels + n_pixels, pixels, [=](PixelType pixel) {
			PixelType res = pixel;
			for (int c = 0; c < DIM; c++) {
				res[c] = pixel[permutation[c]];
			}
			return res;
		});
	}
	else {
		Log(Error, "Permute channels not implemented yet :-(");
	}
}


class Texture {
public:
	using SharedPtr = std::shared_ptr<Texture>;
	using Format = Image::Format;

	Texture() = default; 
	Texture(Color4f value) { setConstant(value); }
	Texture(const string& filepath, bool flip = false, bool srgb = false, uint id = 0);

	void setConstant(const Color4f value){ 
		mValid = true;
		mValue = value;
	};
	void loadImage(const fs::path &filepath, bool flip = false, bool srgb = false) {
		mValid = mImage.loadImage(filepath, flip, srgb);
	}
	Image &getImage() { return mImage; }
	KRR_CALLABLE Color4f getConstant() const { return mValue; }

	KRR_CALLABLE bool isValid() const { return mValid; }
	KRR_CALLABLE bool isOnDevice() const { return mCudaTexture != 0; }
	
	KRR_CALLABLE cudaTextureObject_t getCudaTexture()const { return mCudaTexture; }
	
	__device__ Color tex(Vector2f uv) const {
#ifdef __NVCC__ 
		Color color = (Vector3f) tex2D<float4>(mCudaTexture, uv[0], uv[1]);
		return color;
#endif 
		return {};
	}

	void toDevice();

	void renderUI();
	string getFilemame() {
		if (mTextureId)
			return texture::textureProps[mTextureId].path;
		return "unknown filepath";
	}
	static Texture::SharedPtr createFromFile(const fs::path &filepath, bool flip = false,
											 bool srgb = false);

	bool mValid{ false };
	Color4f mValue{};	 /* If this is a constant texture, the value should be set. */
	Image mImage;
	cudaTextureObject_t mCudaTexture = 0;
	cudaArray_t mCudaArray = 0;
	uint mTextureId;
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
	Material(uint id, const string& name);

	void setTexture(TextureType type, Texture& texture);
	void setConstantTexture(TextureType type, const Color4f color);
	bool determineSrgb(string filename, TextureType type);

	bool hasEmission() { 
		return mTextures[(int)TextureType::Emissive].isValid(); 
	}

	KRR_CALLABLE Texture getTexture(TextureType type) {
		return mTextures[(uint)type];
	}

	KRR_CALLABLE cudaTextureObject_t getCudaTexture(TextureType type) {
		return mTextures[(uint)type].getCudaTexture();
	}

	void toDevice();
	void renderUI();
	string getName() {
		return texture::materialProps.count(mMaterialId) ? 
			texture::materialProps[mMaterialId].name : "unknown";
	}

	MaterialParams mMaterialParams;
	Texture mTextures[5];
	MaterialType mBsdfType{ MaterialType::Disney };
	ShadingModel mShadingModel{ ShadingModel::MetallicRoughness };
	bool mDoubleSided = false;
	uint mMaterialId;
};

KRR_NAMESPACE_END