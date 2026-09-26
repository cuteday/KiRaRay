#include <chrono>
#include <cstring>
#include <stdexcept>

#include <stb_image_write.h>

#include "graphics/textureloader.h"

using namespace krr;

namespace {

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

class TestTextureCache : public TextureCache {
public:
	TestTextureCache() : TextureCache(nullptr, nullptr) {}
	using TextureCache::FillTextureData;
	using TextureCache::FindTextureInCache;
};

struct TemporaryDirectory {
	fs::path path = fs::current_path() / ("krr-textures-" + std::to_string(
		std::chrono::steady_clock::now().time_since_epoch().count()));
	TemporaryDirectory() { require(fs::create_directory(path), "Could not create test directory"); }
	~TemporaryDirectory() {
		std::error_code error;
		fs::remove_all(path, error);
	}
};

void checkPixels(const fs::path &directory) {
	TestTextureCache cache;
	for (int channels : {1, 2}) {
		fs::path path = directory / (std::to_string(channels) + ".png");
		const unsigned char gray[] = {32, 192};
		const unsigned char grayAlpha[] = {32, 64, 192, 128};
		const unsigned char *source = channels == 1 ? gray : grayAlpha;
		require(stbi_write_png(path.string().c_str(), 2, 1, channels, source, 0) != 0,
			"Could not write test image");
		auto image = Image::createFromFile(path);
		require(image->getChannels() == 4, "Image decoder did not expand grayscale to RGBA");
		auto texture = std::make_shared<TextureData>();
		require(cache.FillTextureData(image, texture), "Could not prepare grayscale texture");
		const unsigned char expected[] = {32, 32, 32, static_cast<unsigned char>(channels == 1 ? 255 : 64),
			192, 192, 192, static_cast<unsigned char>(channels == 1 ? 255 : 128)};
		require(texture->format == nvrhi::Format::RGBA8_UNORM && texture->data->size() == 8 &&
			texture->dataLayout[0][0].rowPitch == 8, "Incorrect grayscale texture layout");
		require(std::memcmp(texture->data->data(), expected, sizeof(expected)) == 0,
			"Grayscale texture pixels or alpha changed");
		image.reset();
		require(std::memcmp(texture->data->data(), expected, sizeof(expected)) == 0,
			"Texture upload data did not own its pixels");
	}

	auto image = std::make_shared<Image>(Vector2i(2, 1), Image::Format::RGBAfloat);
	const float expected[] = {-1.f, 0.5f, 4.f, 1.f, 16.f, 0.f, 0.25f, 0.5f};
	std::memcpy(image->data(), expected, sizeof(expected));
	auto texture = std::make_shared<TextureData>();
	require(cache.FillTextureData(image, texture), "Could not prepare HDR texture");
	require(texture->format == nvrhi::Format::RGBA32_FLOAT &&
		texture->data->size() == sizeof(expected) &&
		std::memcmp(texture->data->data(), expected, sizeof(expected)) == 0,
		"HDR upload data changed");
}

void checkCache(const fs::path &directory) {
	TestTextureCache cache;
	fs::path path = directory / "cached.png";
	const unsigned char pixels[] = {32, 64, 128, 192};
	require(stbi_write_png(path.string().c_str(), 1, 1, 4, pixels, 0) != 0,
		"Could not write cached image");
	auto linear = cache.LoadTextureFromFileDeferred(path, false);
	auto srgb = cache.LoadTextureFromFileDeferred(path, true);
	require(linear && srgb && linear != srgb, "Color spaces shared a cached texture");
	require(cache.GetLoadedTexture(path, false)->format == nvrhi::Format::RGBA8_UNORM &&
		cache.GetLoadedTexture(path, true)->format == nvrhi::Format::SRGBA8_UNORM,
		"Cached texture has the wrong color space");
	require(cache.LoadTextureFromFileDeferred(path, false) == linear &&
		cache.LoadTextureFromFileDeferred(path, true) == srgb,
		"Repeated loads did not reuse the matching color space");
	fs::path relative = fs::relative(path, File::cwd());
	require(cache.LoadTextureFromFileDeferred(relative, true) == srgb,
		"Relative and absolute requests did not share the cache");
	require(cache.GetNumberOfRequestedTextures() == 2 && cache.GetNumberOfLoadedTextures() == 2,
		"Cache hits changed loading counters");
}

void checkUnload(const fs::path &directory) {
	TestTextureCache cache;
	const auto path = directory / "unload.png";
	std::shared_ptr<TextureData> linear, srgb, replacement;
	cache.FindTextureInCache(path, false, linear);
	cache.FindTextureInCache(path, true, srgb);
	require(cache.UnloadTexture(srgb) && !cache.GetLoadedTexture(path, true) &&
		cache.GetLoadedTexture(path, false) == linear, "Unload removed the wrong color space");
	require(!cache.UnloadTexture(srgb) && !cache.UnloadTexture(nullptr),
		"Unloading a missing texture succeeded");
	cache.FindTextureInCache(path, true, replacement);
	require(!cache.UnloadTexture(srgb) && cache.GetLoadedTexture(path, true) == replacement,
		"Stale texture removed a newer cache entry");
	auto unrelated = std::make_shared<LoadedTexture>();
	unrelated->path = path.generic_string();
	require(!cache.UnloadTexture(unrelated), "Unowned texture removed a cache entry");
	require(cache.UnloadTexture(linear) && cache.UnloadTexture(replacement),
		"Could not unload remaining texture variants");
}

void checkFailures(const fs::path &directory) {
	TestTextureCache cache;
	fs::path path = directory / "retry.png";
	require(!cache.LoadTextureFromFile(path, false, nullptr), "Missing file load succeeded");
	require(!cache.GetLoadedTexture(path), "Failed immediate load stayed in the cache");
	require(!cache.LoadTextureFromFileDeferred(path, false), "Missing deferred load succeeded");
	require(!cache.GetLoadedTexture(path), "Failed deferred load stayed in the cache");
	require(!cache.LoadTextureFromImage(nullptr, nullptr) &&
		!cache.LoadTextureFromImage(std::make_shared<Image>(), nullptr),
		"Invalid images were accepted");
	require(!cache.IsTextureFinalized(nullptr), "Null texture was considered finalized");
	require(cache.GetNumberOfLoadedTextures() == 0 && cache.GetNumberOfFinalizedTextures() == 0,
		"Failed loads were counted as successes");
	const unsigned char pixels[] = {1, 2, 3, 255};
	require(stbi_write_png(path.string().c_str(), 1, 1, 4, pixels, 0) != 0,
		"Could not write replacement image");
	auto texture = cache.LoadTextureFromFileDeferred(path, false);
	require(texture && cache.IsTextureLoaded(texture), "Failed file could not be retried");
	require(cache.GetNumberOfRequestedTextures() == 3 && cache.GetNumberOfLoadedTextures() == 1,
		"Retried load has incorrect counters");
}

}

void checkTextureCache() {
	TemporaryDirectory directory;
	checkPixels(directory.path);
	checkCache(directory.path);
	checkUnload(directory.path);
	checkFailures(directory.path);
}
