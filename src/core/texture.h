// we want a image->texture->material class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

#pragma once

#include <future>

#include "common.h"
#include "logger.h"
#include "gpu/buffer.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

class Image {
public:
	KRR_CLASS_DEFINE(Image);
	
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
	
	Image() = default;
	~Image() { free(mData); }

	bool loadImage(const string& filepath, bool srgb = false);
	bool saveImage(const string& filepath) { return false; }

	static Image::SharedPtr createFromFile(const string& filepath, bool srgb = false);
	__both__ bool isValid() const { return mSize.x * mSize.y != 0; }
	__both__ vec2i getSize() const { return mSize; }
	__both__ int getChannels() const { return mChannels; }
	__both__ uchar* data() const { return mData; }

private:
	vec2i mSize = {0, 0};
	int mChannels = 0;
	//std::vector<uchar> mData;
	uchar* mData = nullptr;
};

class Texture {
public:
	KRR_CLASS_DEFINE(Texture);
	Texture() = default;
	~Texture() {}

	void loadImage(const string& filepath, bool srgb = false) {
		mImage.loadImage(filepath);
	}
	__both__ bool isValid() { return mImage.isValid() && mCudaTexture; }
	void toDevice();

	static Texture::SharedPtr createFromFile(const string& filepath, bool srgb = false);

private:
	Image mImage;
	CUDABuffer mDeviceMemory;
	cudaTextureObject_t mCudaTexture = 0;
	cudaArray_t mCudaArray = 0;
};

class Material {
public:
	enum class TextureType {
		Diffuse = 0,
		Specular,
		Emissive,
		Normal,
		Transmission,
		Count
	};

	KRR_CLASS_DEFINE(Material);
	Material() = default;
	Material(const string& name) { mName = name; }
	~Material() {}

	void setTexture(TextureType type, Texture& texture) 
	{ 
		//mTextures[(uint)type] = texture; 
	}
	void toDevice();

	struct MaterialParams {
		vec4f diffuse = vec4f(1);
		vec4f specular = vec4f(0);
		vec3f emissive = vec3f(0);
		float emissiveFactor = 1.f;
		float IoR = 1.5f;
		vec3f transmission = vec3f(1);
	} mMaterialParams;

private:
	bool mDoubleSided = false;
	Texture mTextures[(uint)TextureType::Count];
	string mName;
};



KRR_NAMESPACE_END