#pragma once
#include <common.h>
#include <logger.h>

#include <nvrhi/nvrhi.h>
#include <atomic>
#include <filesystem>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <queue>

#include "descriptor.h"
#include <file.h>
#include <texture.h>

KRR_NAMESPACE_BEGIN

class CommonRenderPasses;

enum class TextureAlphaMode {
	UNKNOWN		  = 0,
	STRAIGHT	  = 1,
	PREMULTIPLIED = 2,
	OPAQUE_		  = 3,
	CUSTOM		  = 4,
};

struct LoadedTexture {
	nvrhi::TextureHandle texture;
	TextureAlphaMode alphaMode	  = TextureAlphaMode::UNKNOWN;
	uint32_t originalBitsPerPixel = 0;
	DescriptorHandle bindlessDescriptor;
	std::string path;
	std::string mimeType;
};

struct TextureSubresourceData {
	size_t rowPitch		 = 0;
	size_t depthPitch	 = 0;
	ptrdiff_t dataOffset = 0;
	size_t dataSize		 = 0;
};

struct TextureData : public LoadedTexture {
	std::shared_ptr<Blob> data;

	nvrhi::Format format			  = nvrhi::Format::UNKNOWN;
	uint32_t width					  = 1;
	uint32_t height					  = 1;
	uint32_t depth					  = 1;
	uint32_t arraySize				  = 1;
	uint32_t mipLevels				  = 1;
	nvrhi::TextureDimension dimension = nvrhi::TextureDimension::Unknown;
	bool isRenderTarget				  = false;
	bool forceSRGB					  = false;

	// ArraySlice -> MipLevel -> TextureSubresourceData
	std::vector<std::vector<TextureSubresourceData>> dataLayout;
};

class TextureCache {
protected:
	nvrhi::DeviceHandle m_Device;
	nvrhi::CommandListHandle m_CommandList;
	std::unordered_map<std::string, std::shared_ptr<TextureData>> m_LoadedTextures;
	mutable std::shared_mutex m_LoadedTexturesMutex;

	std::queue<std::shared_ptr<TextureData>> m_TexturesToFinalize;
	std::shared_ptr<DescriptorTableManager> m_DescriptorTable;
	std::mutex m_TexturesToFinalizeMutex;

	uint32_t m_MaxTextureSize = 0;

	bool m_GenerateMipmaps = true;

	Log::Level m_InfoLogSeverity	 = Log::Level::Debug;
	Log::Level m_ErrorLogSeverity = Log::Level::Warning;

	std::atomic<uint32_t> m_TexturesRequested = 0;
	std::atomic<uint32_t> m_TexturesLoaded	  = 0;
	uint32_t m_TexturesFinalized			  = 0;

	bool FindTextureInCache(const std::filesystem::path &path,
							std::shared_ptr<TextureData> &texture);
	std::shared_ptr<Blob> ReadTextureFile(const std::filesystem::path &path) const;
	bool FillTextureData(const Image::SharedPtr image,
						 const std::shared_ptr<TextureData> &texture) const;
	void FinalizeTexture(std::shared_ptr<TextureData> texture, CommonRenderPasses *passes,
						 nvrhi::ICommandList *commandList);
	virtual void TextureLoaded(std::shared_ptr<TextureData> texture);
	virtual std::shared_ptr<TextureData> CreateTextureData();

public:
	TextureCache(nvrhi::IDevice *device, std::shared_ptr<DescriptorTableManager> descriptorTable);
	virtual ~TextureCache();

	// Release all cached textures
	void Reset();

	// Synchronous read and decode, synchronous upload and mip generation on a given command list
	// (must be open). The `passes` argument is optional, and mip generation is disabled if it's
	// NULL.
	std::shared_ptr<LoadedTexture> LoadTextureFromFile(const std::filesystem::path &path, bool sRGB,
													   nvrhi::ICommandList *commandList,
													   CommonRenderPasses *passes = nullptr);

	std::shared_ptr<LoadedTexture> LoadTextureFromImage(const Image::SharedPtr image,
													   nvrhi::ICommandList *commandList,
													   CommonRenderPasses *passes = nullptr);

	// Synchronous read and decode, deferred upload and mip generation (in the
	// ProcessRenderingThreadCommands queue).
	std::shared_ptr<LoadedTexture> LoadTextureFromFileDeferred(const std::filesystem::path &path,
															   bool sRGB);

	// Tells if the texture has been loaded from file successfully and its data is available in the
	// texture object. After the texture is finalized and uploaded to the GPU, the data is no longer
	// available on the CPU, and this function returns false.
	bool IsTextureLoaded(const std::shared_ptr<LoadedTexture> &texture);

	// Tells if the texture has been uploaded to the GPU
	bool IsTextureFinalized(const std::shared_ptr<LoadedTexture> &texture);

	// Removes the texture from cache. The texture must *not* be in the deferred finalization queue
	// when it's unloaded. Returns true if the texture has been found and removed from the cache,
	// false otherwise. Note: Any existing handles for the texture remain valid after the texture is
	// unloaded.
	//       Texture lifetimes are tracked by NVRHI and the texture object is only destroyed when no
	//       references exist.
	bool UnloadTexture(const std::shared_ptr<LoadedTexture> &texture);

	// Process a portion of the upload queue, taking up to `timeLimitMilliseconds` CPU time.
	// If `timeLimitMilliseconds` is 0, processes the entire queue.
	// Returns true if any textures have been processed.
	bool ProcessRenderingThreadCommands(CommonRenderPasses &passes, float timeLimitMilliseconds);

	// Destroys the internal command list in order to release the upload buffers used in it.
	void LoadingFinished();

	// Set the maximum texture size allowed after load. Larger textures are resized to fit this
	// constraint. Currently does not affect DDS textures.
	void SetMaxTextureSize(uint32_t size);

	// Enables or disables automatic mip generation for loaded textures.
	void SetGenerateMipmaps(bool generateMipmaps);

	// Sets the Severity of log messages about textures being loaded.
	void SetInfoLogSeverity(Log::Level value) { m_InfoLogSeverity = value; }

	// Sets the Severity of log messages about textures that couldn't be loaded.
	void SetErrorLogSeverity(Log::Level value) { m_ErrorLogSeverity = value; }

	uint32_t GetNumberOfLoadedTextures() { return m_TexturesLoaded.load(); }
	uint32_t GetNumberOfRequestedTextures() { return m_TexturesRequested.load(); }
	uint32_t GetNumberOfFinalizedTextures() { return m_TexturesFinalized; }

	std::shared_ptr<TextureData> GetLoadedTexture(std::filesystem::path const &path);

	// Texture cache traversal
	// Note: the iterator locks all cache write-accesses for the duration its lifespan !
	class Iterator {
	public:
		typedef std::unordered_map<std::string, std::shared_ptr<TextureData>>::iterator CacheIter;

		Iterator &operator++() {
			++m_Iterator;
			return *this;
		}

		friend bool operator==(Iterator const &a, Iterator const &b) {
			return a.m_Iterator == b.m_Iterator;
		}
		friend bool operator!=(Iterator const &a, Iterator const &b) { return !(a == b); }

		CacheIter const &operator->() const { return m_Iterator; }

	private:
		friend class TextureCache;

		Iterator(std::shared_mutex &mutex, CacheIter it) : m_Lock(mutex), m_Iterator(it) {}

		std::shared_lock<std::shared_mutex> m_Lock;
		CacheIter m_Iterator;
	};

	Iterator begin() { return Iterator(m_LoadedTexturesMutex, m_LoadedTextures.begin()); }

	Iterator end() { return Iterator(m_LoadedTexturesMutex, m_LoadedTextures.end()); }
};

KRR_NAMESPACE_END