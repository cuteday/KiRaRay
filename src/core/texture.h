// we want a image->texture->material class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN
 
// Image class is opaque to gpu.
class Image {
public:
	using SharedPtr = std::shared_ptr<Image>;
	
	enum class Format {
		NONE = 0,
		R,
		RG,
		RGB,
		RGBA,
	};
	enum class DataType {
		Uint8,
		Float16,
		Float32
	};
	
	Image() {};
	~Image() {}

	bool loadImage(const string& filepath, bool srgb = false);
	bool saveImage(const string& filepath) { return false; }

	static Image::SharedPtr createFromFile(const string& filepath, bool srgb = false);
	bool isValid() const { return mSize.x * mSize.y; }
	vec2i getSize() const { return mSize; }
	int getChannels() const { return mChannels; }
	uchar* data() { return mData.data(); }

	bool mSrgb = false;
private:
	vec2i mSize = {0, 0};
	int mChannels = 0;
	std::vector<uchar> mData;
	//uchar* mData = nullptr;
};

class Texture {
public:
	using SharedPtr = std::shared_ptr<Texture>;

	Texture() {
		mImage = Image::SharedPtr(new Image());
	};

	void loadImage(const string& filepath, bool srgb = false) {
		mImage->loadImage(filepath);
	}
	__both__ bool isValid() { return mCudaTexture; }
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
		Diffuse = 0,
		Specular,
		Emissive,
		Normal,
		Transmission,
		Count
	};

	enum class ShadingModel {
		Diffuse = 0,
		MetallicRoughness,
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
	ShadingModel mShadingModel = ShadingModel::Diffuse;
	bool mDoubleSided = false;
	//cudaTextureObject_t mDiffuseTexture;
	//cudaTextureObject_t mCudaTextures[5] = {};
	string mName;
};



KRR_NAMESPACE_END