// we want a image->texture->material class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

// TODO: reorganize and clean up constructors (and "createFromFile" methods)
#pragma once
#include <map>
#include <cuda_runtime.h>

#include "common.h"
#include "math/math.h"
#include "file.h"
#include "window.h"
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

	bool loadImage(const fs::path& filepath, bool srgb = false);
	bool saveImage(const fs::path& filepath);

	static bool isHdr(const string& filepath);
	static Image::SharedPtr createFromFile(const string& filepath, bool srgb = false);
	bool isValid() const { return mFormat != Format::NONE && mSize[0] * mSize[1]; }
	bool isSrgb() const { return mSrgb; }
	Vector2i getSize() const { return mSize; }
	Format getFormat() const { return mFormat; }
	inline size_t getElementSize() const { return mFormat == Format::RGBAfloat ? sizeof(float) : sizeof(uchar); }
	int getChannels() const { return mChannels; }
	uchar* data() { return mData; }
	void reset(uchar *data) { mData = data; }
	
private:
	bool mSrgb{ };
	Vector2i mSize = Vector2i::Zero();
	int mChannels{ 4 };
	Format mFormat{ };
	uchar* mData{ };
};

class Texture {
public:
	using SharedPtr = std::shared_ptr<Texture>;
	using Format = Image::Format;

	Texture() = default; 
	Texture(const string& filepath, bool srgb = false, uint id = 0);

	void loadImage(const string& filepath, bool srgb = false) {
		mValid = mImage.loadImage(filepath, srgb);
	}
	Image &getImage() { return mImage; }

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
	static Texture::SharedPtr createFromFile(const string& filepath, bool srgb = false);

	bool mValid = false;
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
		Vector4f diffuse{ 1 };			// RGB for base color and A (optional) for opacity 
		Vector4f specular{ 0 };			// G-roughness B-metallic A-shininess in MetalRough model
										// RGB - specular color (F0); A - shininess in SpecGloss model
		Vector3f emissive{ 0 };
		float IoR{ 1.5f };
		float diffuseTransmission{ 0 };
		float specularTransmission{ 0 };
	};

	Material() {};
	Material(uint id, const string& name);

	void setTexture(TextureType type, Texture& texture);
	bool determineSrgb(string filename, TextureType type);

	bool hasEmission() { 
		return mMaterialParams.emissive.any() || mTextures[(int)TextureType::Emissive].isValid(); 
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
	BsdfType mBsdfType{ BsdfType::Disney };
	ShadingModel mShadingModel{ ShadingModel::MetallicRoughness };
	bool mDoubleSided = false;
	uint mMaterialId;
};

KRR_NAMESPACE_END