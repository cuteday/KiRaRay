// we want a image->texture class 
// a gpu buffer should be a member of texture class, a texture cache may also needed

#pragma once

#include "common.h"
#include "logger.h"
#include "gpu/buffer.h"
#include "math/math.h"

KRR_NAMESPACE_BEGIN

class Image {
public:
	enum class Format {
		NONE = 0,
		R,
		RG,
		RGB,
		RGBA,
	};

	KRR_CLASS_DEFINE(Image);

	Image() = default;
	~Image() {}

	static Image loadImage(const string& filepath);
	bool saveImage(const string& filepath) { return false; }

private:

	vec2i mSize = {0, 0};
	int mChannels = 0;
	std::vector<uchar> mData;
	
	static std::map<string, Image> sImageCache;
};

class Texture {
public:
	KRR_CLASS_DEFINE(Texture);
	Texture() = default;
	~Texture() {}

	void loadFromImage(const string& filepath) {
		mImage = Image::loadImage(filepath);
	}
//private:

	Image mImage;
	CUDABuffer mDeviceMemory;
};

class Material {
public:
	enum class TextureType {
		Diffuse,
		Specular,
		Emissive,
		Normal
	};

	KRR_CLASS_DEFINE(Material);
	Material() = default;
	~Material() {}

//private:
	Texture mDiffuse;
};

KRR_NAMESPACE_END