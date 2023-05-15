#include <common.h>
#include <logger.h>
#include <texture.h>

#include "textureloader.h"
#include "helperpass.h"

#include <algorithm>
#include <chrono>
#include <regex>

KRR_NAMESPACE_BEGIN

TextureCache::TextureCache(nvrhi::IDevice *device,
						   std::shared_ptr<DescriptorTableManager> descriptorTable) :
	m_Device(device), m_DescriptorTable(std::move(descriptorTable)) {}

TextureCache::~TextureCache() { Reset(); }

void TextureCache::Reset() {
	std::lock_guard<std::shared_mutex> guard(m_LoadedTexturesMutex);

	m_LoadedTextures.clear();

	m_TexturesRequested = 0;
	m_TexturesLoaded	= 0;
}

void TextureCache::SetGenerateMipmaps(bool generateMipmaps) { m_GenerateMipmaps = generateMipmaps; }

bool TextureCache::FindTextureInCache(const std::filesystem::path &path,
									  std::shared_ptr<TextureData> &texture) {
	std::lock_guard<std::shared_mutex> guard(m_LoadedTexturesMutex);

	// First see if this texture is already loaded (or being loaded).

	texture = m_LoadedTextures[path.generic_string()];
	if (texture) {
		return true;
	}

	// Allocate a new texture slot for this file name and return it. Load the file later in a thread
	// pool. LoadTextureFromFileAsync function for a given scene is only called from one thread, so
	// there is no chance of loading the same texture twice.

	texture									= CreateTextureData();
	m_LoadedTextures[path.generic_string()] = texture;

	++m_TexturesRequested;

	return false;
}

std::shared_ptr<Blob> TextureCache::ReadTextureFile(const std::filesystem::path &path) const {
	auto fileData = File::readFile(path);

	if (!fileData)
		logMessage(m_ErrorLogSeverity, "Couldn't read texture file '%s'",
				   path.generic_string().c_str());

	return fileData;
}

std::shared_ptr<TextureData> TextureCache::CreateTextureData() {
	return std::make_shared<TextureData>();
}

bool TextureCache::FillTextureData(const Image::SharedPtr image,
								   const std::shared_ptr<TextureData> &texture) const {
	int width = 0, height = 0, originalChannels = 0, channels = 0;

	bool is_hdr = image->getFormat() == Image::Format::RGBAfloat;
	originalChannels = image->getChannels();

	if (originalChannels == 3) {
		channels = 4;
	} else {
		channels = originalChannels;
	}
	width = image->getSize()[0], height = image->getSize()[1];
	
	int bytesPerPixel = channels * (is_hdr ? 4 : 1);
	size_t sizeInBytes	  = width * height * bytesPerPixel;
	unsigned char *bitmap = new unsigned char[sizeInBytes];
	memcpy(bitmap, image->data(), sizeInBytes);

	if (!bitmap) {
		logMessage(m_ErrorLogSeverity, "Couldn't load generic texture '%s'", texture->path.c_str());
		return false;
	}

	texture->originalBitsPerPixel = static_cast<uint32_t>(originalChannels) * (is_hdr ? 32 : 8);
	texture->width				  = static_cast<uint32_t>(width);
	texture->height				  = static_cast<uint32_t>(height);
	texture->isRenderTarget		  = true;
	texture->mipLevels			  = 1;
	texture->dimension			  = nvrhi::TextureDimension::Texture2D;

	texture->dataLayout.resize(1);
	texture->dataLayout[0].resize(1);
	texture->dataLayout[0][0].dataOffset = 0;
	texture->dataLayout[0][0].rowPitch	 = static_cast<size_t>(width * bytesPerPixel);
	texture->dataLayout[0][0].dataSize = static_cast<size_t>(sizeInBytes);

	texture->data = std::make_shared<Blob>(bitmap, sizeInBytes);
	bitmap		  = nullptr; // ownership transferred to the blob

	switch (channels) {
		case 1:
			texture->format = is_hdr ? nvrhi::Format::R32_FLOAT : nvrhi::Format::R8_UNORM;
			break;
		case 2:
			texture->format = is_hdr ? nvrhi::Format::RG32_FLOAT : nvrhi::Format::RG8_UNORM;
			break;
		case 4:
			texture->format = is_hdr ? nvrhi::Format::RGBA32_FLOAT
									 : (texture->forceSRGB ? nvrhi::Format::SRGBA8_UNORM
														   : nvrhi::Format::RGBA8_UNORM);
			break;
		default:
			texture->data.reset(); // release the bitmap data
			logMessage(m_ErrorLogSeverity, "Unsupported number of components (%d) for texture '%s'",
					   channels, texture->path.c_str());
			return false;
	}

	return true;
}

uint GetMipLevelsNum(uint width, uint height) {
	uint size	   = std::min(width, height);
	uint levelsNum = (uint) (logf((float) size) / logf(2.0f)) + 1;

	return levelsNum;
}

void TextureCache::FinalizeTexture(std::shared_ptr<TextureData> texture, CommonRenderPasses *passes,
								   nvrhi::ICommandList *commandList) {
	assert(texture->data);
	assert(commandList);

	uint originalWidth	= texture->width;
	uint originalHeight = texture->height;

	bool isBlockCompressed = (texture->format == nvrhi::Format::BC1_UNORM) ||
							 (texture->format == nvrhi::Format::BC1_UNORM_SRGB) ||
							 (texture->format == nvrhi::Format::BC2_UNORM) ||
							 (texture->format == nvrhi::Format::BC2_UNORM_SRGB) ||
							 (texture->format == nvrhi::Format::BC3_UNORM) ||
							 (texture->format == nvrhi::Format::BC3_UNORM_SRGB) ||
							 (texture->format == nvrhi::Format::BC4_SNORM) ||
							 (texture->format == nvrhi::Format::BC4_UNORM) ||
							 (texture->format == nvrhi::Format::BC5_SNORM) ||
							 (texture->format == nvrhi::Format::BC5_UNORM) ||
							 (texture->format == nvrhi::Format::BC6H_SFLOAT) ||
							 (texture->format == nvrhi::Format::BC6H_UFLOAT) ||
							 (texture->format == nvrhi::Format::BC7_UNORM) ||
							 (texture->format == nvrhi::Format::BC7_UNORM_SRGB);

	if (isBlockCompressed) {
		originalWidth  = (originalWidth + 3) & ~3;
		originalHeight = (originalHeight + 3) & ~3;
	}

	uint scaledWidth  = originalWidth;
	uint scaledHeight = originalHeight;

	if (m_MaxTextureSize > 0 && int(std::max(originalWidth, originalHeight)) > m_MaxTextureSize &&
		texture->isRenderTarget && texture->dimension == nvrhi::TextureDimension::Texture2D) {
		if (originalWidth >= originalHeight) {
			scaledHeight = originalHeight * m_MaxTextureSize / originalWidth;
			scaledWidth	 = m_MaxTextureSize;
		} else {
			scaledWidth	 = originalWidth * m_MaxTextureSize / originalHeight;
			scaledHeight = m_MaxTextureSize;
		}
	}

	const char *dataPointer = static_cast<const char *>(texture->data->data());

	nvrhi::TextureDesc textureDesc;
	textureDesc.format		   = texture->format;
	textureDesc.width		   = scaledWidth;
	textureDesc.height		   = scaledHeight;
	textureDesc.depth		   = texture->depth;
	textureDesc.arraySize	   = texture->arraySize;
	textureDesc.dimension	   = texture->dimension;
	textureDesc.mipLevels	   = m_GenerateMipmaps && texture->isRenderTarget && passes
									 ? GetMipLevelsNum(textureDesc.width, textureDesc.height)
									 : texture->mipLevels;
	textureDesc.debugName	   = texture->path;
	textureDesc.isRenderTarget = texture->isRenderTarget;
	texture->texture		   = m_Device->createTexture(textureDesc);

	commandList->beginTrackingTextureState(texture->texture, nvrhi::AllSubresources,
										   nvrhi::ResourceStates::Common);

	if (m_DescriptorTable)
		texture->bindlessDescriptor = m_DescriptorTable->CreateDescriptorHandle(
			nvrhi::BindingSetItem::Texture_SRV(0, texture->texture));

	if (scaledWidth != originalWidth || scaledHeight != originalHeight) {
		nvrhi::TextureDesc tempTextureDesc;
		tempTextureDesc.format	  = texture->format;
		tempTextureDesc.width	  = originalWidth;
		tempTextureDesc.height	  = originalHeight;
		tempTextureDesc.depth	  = textureDesc.depth;
		tempTextureDesc.arraySize = textureDesc.arraySize;
		tempTextureDesc.mipLevels = 1;
		tempTextureDesc.dimension = textureDesc.dimension;

		nvrhi::TextureHandle tempTexture = m_Device->createTexture(tempTextureDesc);
		assert(tempTexture);
		commandList->beginTrackingTextureState(tempTexture, nvrhi::AllSubresources,
											   nvrhi::ResourceStates::Common);

		for (uint32_t arraySlice = 0; arraySlice < texture->arraySize; arraySlice++) {
			const TextureSubresourceData &layout = texture->dataLayout[arraySlice][0];

			commandList->writeTexture(tempTexture, arraySlice, 0, dataPointer + layout.dataOffset,
									  layout.rowPitch, layout.depthPitch);
		}

		nvrhi::FramebufferHandle framebuffer = m_Device->createFramebuffer(
			nvrhi::FramebufferDesc().addColorAttachment(texture->texture));

		passes->BlitTexture(commandList, framebuffer, tempTexture);
	} else {
		for (uint32_t arraySlice = 0; arraySlice < texture->arraySize; arraySlice++) {
			for (uint32_t mipLevel = 0; mipLevel < texture->mipLevels; mipLevel++) {
				const TextureSubresourceData &layout = texture->dataLayout[arraySlice][mipLevel];

				commandList->writeTexture(texture->texture, arraySlice, mipLevel,
										  dataPointer + layout.dataOffset, layout.rowPitch,
										  layout.depthPitch);
			}
		}
	}

	texture->data.reset();

	uint width	= scaledWidth;
	uint height = scaledHeight;
	for (uint mipLevel = texture->mipLevels; mipLevel < textureDesc.mipLevels; mipLevel++) {
		width /= 2;
		height /= 2;

		nvrhi::FramebufferHandle framebuffer = m_Device->createFramebuffer(
			nvrhi::FramebufferDesc().addColorAttachment(nvrhi::FramebufferAttachment()
															.setTexture(texture->texture)
															.setArraySlice(0)
															.setMipLevel(mipLevel)));

		BlitParameters blitParams;
		blitParams.sourceTexture	 = texture->texture;
		blitParams.sourceMip		 = mipLevel - 1;
		blitParams.targetFramebuffer = framebuffer;
		passes->BlitTexture(commandList, blitParams);
	}

	commandList->setPermanentTextureState(texture->texture, nvrhi::ResourceStates::ShaderResource);
	commandList->commitBarriers();

	++m_TexturesFinalized;
}

void TextureCache::TextureLoaded(std::shared_ptr<TextureData> texture) {
	std::lock_guard<std::mutex> guard(m_TexturesToFinalizeMutex);

	if (texture->mimeType.empty())
		logMessage(m_InfoLogSeverity, "Loaded %d x %d, %d bpp: %s", texture->width, texture->height,
				   texture->originalBitsPerPixel, texture->path.c_str());
	else
		logMessage(m_InfoLogSeverity, "Loaded %d x %d, %d bpp: %s (%s)", texture->width,
				   texture->height, texture->originalBitsPerPixel, texture->path.c_str(),
				   texture->mimeType.c_str());
}

std::shared_ptr<LoadedTexture> TextureCache::LoadTextureFromFile(const std::filesystem::path &path,
																 bool sRGB,
																 nvrhi::ICommandList *commandList,
																 CommonRenderPasses *passes) {
	std::shared_ptr<TextureData> texture;
	fs::path absolutePath = File::resolve(path);

	if (FindTextureInCache(absolutePath, texture)) return texture;

	texture->forceSRGB = sRGB;
	texture->path	   = absolutePath.generic_string();

	auto image = Image::createFromFile(absolutePath, false, sRGB);

	if (image->isValid()) {
		if (FillTextureData(image, texture)) {
			TextureLoaded(texture);
			FinalizeTexture(texture, passes, commandList);
		}
	} else {
		Log(Error, "Failed to load texture from an invalid image!");
		return nullptr;
	}
	++m_TexturesLoaded;
	return texture;
}

std::shared_ptr<LoadedTexture> TextureCache::LoadTextureFromImage(
								   const Image::SharedPtr image,
								   nvrhi::ICommandList *commandList,
								   CommonRenderPasses *passes) {
	std::shared_ptr<TextureData> texture = std::make_shared<TextureData>();
	texture->forceSRGB = image->isSrgb();
	
	if (image->isValid()) {
		if (FillTextureData(image, texture)) {
			TextureLoaded(texture);
			FinalizeTexture(texture, passes, commandList);
		}
	} else {
		Log(Error, "Failed to load texture from an invalid image!");
		return nullptr;
	}
	
	++m_TexturesLoaded;
	return texture;
}

std::shared_ptr<LoadedTexture>
TextureCache::LoadTextureFromFileDeferred(const std::filesystem::path &path, bool sRGB) {
	std::shared_ptr<TextureData> texture;

	if (FindTextureInCache(path, texture)) return texture;

	texture->forceSRGB = sRGB;
	texture->path	   = path.generic_string();

	auto image = Image::createFromFile(path, false, sRGB);
	if (image->isValid()) {
		if (FillTextureData(image, texture)) {
			TextureLoaded(texture);

			std::lock_guard<std::mutex> guard(m_TexturesToFinalizeMutex);

			m_TexturesToFinalize.push(texture);
		}
	}

	++m_TexturesLoaded;

	return texture;
}

std::shared_ptr<TextureData> TextureCache::GetLoadedTexture(std::filesystem::path const &path) {
	std::lock_guard<std::shared_mutex> guard(m_LoadedTexturesMutex);
	return m_LoadedTextures[path.generic_string()];
}

bool TextureCache::ProcessRenderingThreadCommands(CommonRenderPasses &passes,
												  float timeLimitMilliseconds) {
	using namespace std::chrono;

	time_point<high_resolution_clock> startTime = high_resolution_clock::now();

	uint commandsExecuted = 0;
	while (true) {
		std::shared_ptr<TextureData> pTexture;

		if (timeLimitMilliseconds > 0 && commandsExecuted > 0) {
			time_point<high_resolution_clock> now = high_resolution_clock::now();

			if (float(duration_cast<microseconds>(now - startTime).count()) >
				timeLimitMilliseconds * 1e3f)
				break;
		}

		{
			std::lock_guard<std::mutex> guard(m_TexturesToFinalizeMutex);

			if (m_TexturesToFinalize.empty()) break;

			pTexture = m_TexturesToFinalize.front();
			m_TexturesToFinalize.pop();
		}

		if (pTexture->data) {
			// LOG("Finalizing texture %s", pTexture->fileName.c_str());
			commandsExecuted += 1;

			if (!m_CommandList) {
				m_CommandList = m_Device->createCommandList();
			}

			m_CommandList->open();

			FinalizeTexture(pTexture, &passes, m_CommandList);

			m_CommandList->close();
			m_Device->executeCommandList(m_CommandList);
			m_Device->runGarbageCollection();
		}
	}

	return (commandsExecuted > 0);
}

void TextureCache::LoadingFinished() { m_CommandList = nullptr; }

void TextureCache::SetMaxTextureSize(uint32_t size) { m_MaxTextureSize = size; }

bool TextureCache::IsTextureLoaded(const std::shared_ptr<LoadedTexture> &_texture) {
	TextureData *texture = static_cast<TextureData *>(_texture.get());

	return texture && texture->data;
}

bool TextureCache::IsTextureFinalized(const std::shared_ptr<LoadedTexture> &texture) {
	return texture->texture != nullptr;
}

bool TextureCache::UnloadTexture(const std::shared_ptr<LoadedTexture> &texture) {
	const auto &it = m_LoadedTextures.find(texture->path);

	if (it == m_LoadedTextures.end()) return false;

	m_LoadedTextures.erase(it);

	return true;
}

KRR_NAMESPACE_END