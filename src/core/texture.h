// TODO: reorganize and clean up constructors (and "createFromFile" methods)
#pragma once
#include <map>
#include <cuda_runtime.h>
#ifdef __NVCC__
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#endif
#include "common.h"

#include "file.h"
#include "scenenode.h"
#include "raytracing.h"
#include "material/description.h"
#include "render/materials/bxdf.h"

NAMESPACE_BEGIN(krr)

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
#ifdef __NVCC__
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
#endif
}

class Texture {
public:
	using SharedPtr = std::shared_ptr<Texture>;
	using Format = Image::Format;

	Texture() = default; 
	Texture(RGBA value) { setConstant(value); }
	Texture(const string& filepath, bool flip = false, bool srgb = false);

	void setConstant(const RGBA value) { 
		mValue = value;
	};

	void loadImage(const fs::path &filepath, bool flip = false, bool srgb = false) {
		mImage = std::make_shared<Image>();
		if(!mImage->loadImage(filepath, flip, srgb) || !mImage->isValid())
			mImage.reset();
	}
	Image::SharedPtr getImage() const { return mImage; }
	RGBA getConstant() const { return mValue; }
	std::string getFilename() const { return mFilename; }
	bool hasImage() const { return mImage && mImage->isValid(); }

	static Texture::SharedPtr createFromFile(const fs::path &filepath, bool flip = false,
											 bool srgb = false);

	RGBA mValue{};	 /* If this is a constant texture, the value should be set. */
	Image::SharedPtr mImage;
	string mFilename;
};

class Material : public SceneGraphLeaf {
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
		RGBA diffuse{ 1 };			// RGB for base color and A (optional) for opacity 
		RGBA specular{ 0 };			// G-roughness B-metallic A-shininess in MetalRough model
									// RGB - specular color (F0); A - shininess in SpecGloss model
		float specularTransmission{ 0 };
		float anisotropic{ 0 };
		float IoR{ 1.5f };
		Spectra spectralEta{}, spectralK{};
	};

	Material() {};
	Material(const string& name);

	void setName(const std::string& name) { mName = name; }
	void setTexture(TextureType type, Texture::SharedPtr texture);
	void setConstantTexture(TextureType type, const RGBA color);
	bool determineSrgb(string filename, TextureType type);
	void setColorSpace(const ColorSpaceType colorSpace) { mColorSpace = colorSpace; }
	void setDescription(std::shared_ptr<MaterialDescription> description);
	std::shared_ptr<MaterialDescription> getDescription() const { return mDescription; }
	void updateFrom(const Material &material);

	bool hasEmission();
	bool hasTexture(TextureType type);
	Texture::SharedPtr getTexture(TextureType type) { return mTextures[(uint)type]; }
	
	const string& getName() const { return mName; }
	int getMaterialId() const { return mMaterialId; }
	const RGBColorSpace *getColorSpace() const { return spec::getColorSpace(mColorSpace); }
	AABB getLocalBoundingBox() const override { return AABB::Zero(); }
	std::shared_ptr<SceneGraphLeaf> clone() override;

	void renderUI();

	MaterialParams mMaterialParams;
	Texture::SharedPtr mTextures[(uint32_t)TextureType::Count];
	MaterialType mBsdfType{ MaterialType::Disney };
	ShadingModel mShadingModel{ ShadingModel::SpecularGlossiness };
	ColorSpaceType mColorSpace{ ColorSpaceType::sRGB };
	string mName;
	int mMaterialId{-1};
	std::shared_ptr<MaterialDescription> mDescription;
	bool mHasProgramEmission{false};
};

KRR_ENUM_DEFINE(Material::TextureType, {
	{Material::TextureType::Diffuse, "diffuse"},
	{Material::TextureType::Specular, "specular"},
	{Material::TextureType::Normal, "normal"},
	{Material::TextureType::Emissive, "emissive"},
	{Material::TextureType::Transmission, "transmission"},
})

KRR_ENUM_DEFINE(Material::ShadingModel, {
	{Material::ShadingModel::MetallicRoughness, "metallic_roughness"},
	{Material::ShadingModel::SpecularGlossiness, "specular_glossiness"},
})

namespace rt {
class TextureData {
public:
	RGBA mValue{};
	cudaTextureObject_t mCudaTexture{};
	cudaArray_t mCudaArray{};
	bool mValid{};
	MaterialFilter mFilter{MaterialFilter::Linear};
	Vector2i mSize{0, 0};

	void initializeFromHost(Texture::SharedPtr texture, const MaterialTexture *sampling = nullptr);
	void release() noexcept;

	KRR_CALLABLE bool isValid() const { return mValid; }
	KRR_CALLABLE cudaTextureObject_t getCudaTexture() const {
		return mCudaTexture;
	}
	KRR_CALLABLE RGBA getConstant() const { return mValue; }
	KRR_CALLABLE RGBA evaluate(Vector2f uv) const {
#ifdef __CUDA_ARCH__
		if (mCudaTexture) {
			if (mFilter != MaterialFilter::Cubic) return tex2D<float4>(mCudaTexture, uv[0], uv[1]);
			float x = uv[0] * mSize[0] - .5f, y = uv[1] * mSize[1] - .5f;
			float ix = floorf(x), iy = floorf(y);
			MaterialValue wx = materialCubicWeights(x - ix), wy = materialCubicWeights(y - iy);
			RGBA result(0);
			for (int j = 0; j < 4; ++j)
				for (int i = 0; i < 4; ++i) {
					RGBA value = tex2D<float4>(mCudaTexture,
						(ix + i - .5f) / mSize[0], (iy + j - .5f) / mSize[1]);
					result += value * (wx[i] * wy[j]);
				}
			return result;
		}
#endif
		return mValue;
	}
};

enum class MaterialEvaluation : uint8_t { Surface, Opacity, Emission, Classification, Weight };

struct MaterialProgramData {
	bool enabled{false};
	MaterialModel model{MaterialModel::OpenPBR};
	MaterialProgramKind kind{MaterialProgramKind::Constant};
	MaterialValues defaults;
	uint32_t authoredMask{0};
	MaterialProgramView surface, opacity, emission, classification, weight;
	const MaterialValue *uniforms{nullptr};
	const TextureData *textures{nullptr};
	const MaterialSimpleBinding *simple{nullptr};
	uint32_t simpleCount{0};
	const MaterialProgramData *components{nullptr};
	uint32_t componentCount{0};

	KRR_CALLABLE MaterialValues evaluate(const MaterialContext &context,
		MaterialEvaluation evaluation = MaterialEvaluation::Surface) const {
		MaterialValues result = defaults;
		if (!(authoredMask & (1u << int(MaterialParameter::Normal)))) result[MaterialParameter::Normal] = context.normal;
		if (!(authoredMask & (1u << int(MaterialParameter::CoatNormal)))) result[MaterialParameter::CoatNormal] = context.normal;
		if (!(authoredMask & (1u << int(MaterialParameter::Tangent)))) result[MaterialParameter::Tangent] = context.tangent;
		if (kind == MaterialProgramKind::Constant) return result;
		if (kind == MaterialProgramKind::Simple) {
			for (uint32_t index = 0; index < simpleCount; ++index) {
				const MaterialSimpleBinding &binding = simple[index];
				if (evaluation == MaterialEvaluation::Opacity && binding.parameter != MaterialParameter::Opacity) continue;
				if (evaluation == MaterialEvaluation::Emission && !binding.emission) continue;
				if (evaluation == MaterialEvaluation::Weight && binding.parameter != MaterialParameter::Weight) continue;
				if (evaluation == MaterialEvaluation::Classification &&
					!materialClassificationParameter(model, binding.parameter)) continue;
				RGBA value = textures[binding.texture].evaluate({context.uv[0], context.uv[1]});
				result[binding.parameter] = binding.channel >= 0 ? MaterialValue(value[binding.channel]) :
					MaterialValue(value[0], value[1], value[2], value[3]);
			}
			return result;
		}
		struct Sampler {
			const TextureData *textures;
			KRR_CALLABLE MaterialValue operator()(int index, MaterialValue uv) const {
#ifdef __CUDA_ARCH__
				RGBA value = textures[index].evaluate({uv[0], uv[1]});
				return {value[0], value[1], value[2], value[3]};
#else
				RGBA value = textures[index].getConstant();
				return {value[0], value[1], value[2], value[3]};
#endif
			}
		};
		MaterialProgramView program = surface;
		if (evaluation == MaterialEvaluation::Opacity) program = opacity;
		else if (evaluation == MaterialEvaluation::Emission) program = emission;
		else if (evaluation == MaterialEvaluation::Classification) program = classification;
		else if (evaluation == MaterialEvaluation::Weight) program = weight;
		evaluateMaterialProgram(program, uniforms, context, Sampler{textures}, result);
		return result;
	}
};

class MaterialProgramStorage;

class MaterialData {
public:
	Material::MaterialParams mMaterialParams;
	TextureData mTextures[(uint32_t) Material::TextureType::Count];
	MaterialType mBsdfType{MaterialType::Disney};
	Material::ShadingModel mShadingModel{Material::ShadingModel::MetallicRoughness};
	const RGBColorSpace *mColorSpace{nullptr};
	MaterialProgramData mProgram;
	MaterialProgramStorage *mProgramStorage{nullptr};
	void releaseProgram() noexcept;

	void getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data,
					   bool initialize) const;

	KRR_CALLABLE TextureData getTexture(Material::TextureType type) const {
		return mTextures[(uint32_t) type];
	}

	KRR_CALLABLE const RGBColorSpace *getColorSpace() const { return mColorSpace; }
};
}

NAMESPACE_END(krr)
