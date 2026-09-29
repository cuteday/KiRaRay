#include "exr.h"

#include <openexr.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace krr::exr {
namespace {

struct Context {
	exr_context_t handle = nullptr;
	char error[1024] = {};
	~Context() { if (handle) exr_finish(&handle); }

	void check(exr_result_t result) const {
		if (result != EXR_ERR_SUCCESS)
			throw std::runtime_error(error[0] ? error : exr_get_default_error_message(result));
	}

	static void report(exr_const_context_t context, exr_result_t, const char *message) {
		void *user = nullptr;
		if (context && exr_get_user_data(context, &user) == EXR_ERR_SUCCESS && user)
			std::snprintf(static_cast<Context *>(user)->error, sizeof(Context::error), "%s", message);
	}
};

struct Decoder {
	exr_context_t context;
	exr_decode_pipeline_t pipeline = EXR_DECODE_PIPELINE_INITIALIZER;
	bool initialized = false;
	~Decoder() { exr_decoding_destroy(context, &pipeline); }
};

int component(const char *name) {
	if (!std::strcmp(name, "R")) return 0;
	if (!std::strcmp(name, "G")) return 1;
	if (!std::strcmp(name, "B")) return 2;
	if (!std::strcmp(name, "A")) return 3;
	return -1;
}

} // namespace

void load(const char *filename, float **data, int *width, int *height, bool flip) {
	*data = nullptr;
	*width = *height = 0;
	Context context;
	exr_context_initializer_t options = EXR_DEFAULT_CONTEXT_INITIALIZER;
	options.user_data = &context;
	options.error_handler_fn = Context::report;
	context.check(exr_start_read(&context.handle, filename, &options));

	int parts = 0;
	exr_storage_t storage;
	context.check(exr_get_count(context.handle, &parts));
	context.check(exr_get_storage(context.handle, 0, &storage));
	if (parts != 1 || (storage != EXR_STORAGE_SCANLINE && storage != EXR_STORAGE_TILED))
		throw std::runtime_error("EXR textures require one flat image; multipart and deep images are unsupported");

	exr_attr_box2i_t window;
	context.check(exr_get_data_window(context.handle, 0, &window));
	int64_t w = int64_t(window.max.x) - window.min.x + 1;
	int64_t h = int64_t(window.max.y) - window.min.y + 1;
	if (w <= 0 || h <= 0 || w > std::numeric_limits<int32_t>::max() / (4 * sizeof(float)) ||
		h > std::numeric_limits<int>::max() || uint64_t(w) * uint64_t(h) > SIZE_MAX / (4 * sizeof(float)))
		throw std::runtime_error("EXR image dimensions exceed supported limits");
	const int imageWidth = int(w), imageHeight = int(h);

	const exr_attr_chlist_t *channels = nullptr;
	context.check(exr_get_channels(context.handle, 0, &channels));
	const char *gray = nullptr;
	int topLevelChannels = 0;
	bool rgb[3] = {};
	for (int i = 0; i < channels->num_channels; ++i) {
		const auto &channel = channels->entries[i];
		const char *name = channel.name.str;
		if (std::strchr(name, '.')) continue;
		++topLevelChannels;
		gray = name;
		int index = component(name);
		if (index >= 0 && index < 3) rgb[index] = true;
		if (channel.x_sampling != 1 || channel.y_sampling != 1)
			throw std::runtime_error("Subsampled EXR texture channels are unsupported");
	}
	if (topLevelChannels != 1) {
		gray = nullptr;
		if (!rgb[0] || !rgb[1] || !rgb[2])
			throw std::runtime_error("EXR texture requires top-level R, G, B channels or one grayscale channel");
	}

	size_t pixels = size_t(imageWidth) * imageHeight;
	std::unique_ptr<float, decltype(&std::free)> output(
		static_cast<float *>(std::malloc(pixels * 4 * sizeof(float))), &std::free);
	if (!output) throw std::bad_alloc();
	std::fill_n(output.get(), pixels * 4, 0.f);
	for (size_t i = 0; i < pixels; ++i) output.get()[4 * i + 3] = 1.f;
	Decoder decoder{context.handle};
	auto decode = [&](const exr_chunk_info_t &chunk, int64_t x, int64_t y) {
		if (x < 0 || y < 0 || chunk.width <= 0 || chunk.height <= 0 ||
			x + chunk.width > imageWidth || y + chunk.height > imageHeight)
			throw std::runtime_error("EXR chunk lies outside its data window");
		if (decoder.initialized)
			context.check(exr_decoding_update(context.handle, 0, &chunk, &decoder.pipeline));
		else {
			context.check(exr_decoding_initialize(context.handle, 0, &chunk, &decoder.pipeline));
			decoder.initialized = true;
		}
		for (int i = 0; i < decoder.pipeline.channel_count; ++i) {
			auto &channel = decoder.pipeline.channels[i];
			int index = gray ? (std::strcmp(channel.channel_name, gray) ? -1 : 0) : component(channel.channel_name);
			channel.decode_to_ptr = nullptr;
			if (index < 0) continue;
			channel.user_data_type = EXR_PIXEL_FLOAT;
			channel.user_bytes_per_element = sizeof(float);
			channel.user_pixel_stride = 4 * sizeof(float);
			channel.user_line_stride = imageWidth * 4 * sizeof(float);
			channel.decode_to_ptr = reinterpret_cast<uint8_t *>(output.get() + 4 * (size_t(y) * imageWidth + size_t(x)) + index);
		}
		context.check(exr_decoding_choose_default_routines(context.handle, 0, &decoder.pipeline));
		context.check(exr_decoding_run(context.handle, 0, &decoder.pipeline));
	};

	if (storage == EXR_STORAGE_SCANLINE) {
		for (int64_t y = window.min.y; y <= window.max.y;) {
			exr_chunk_info_t chunk;
			context.check(exr_read_scanline_chunk_info(context.handle, 0, int(y), &chunk));
			decode(chunk, int64_t(chunk.start_x) - window.min.x, int64_t(chunk.start_y) - window.min.y);
			y += chunk.height;
		}
	} else {
		int32_t tileWidth, tileHeight;
		context.check(exr_get_tile_sizes(context.handle, 0, 0, 0, &tileWidth, &tileHeight));
		if (tileWidth <= 0 || tileHeight <= 0) throw std::runtime_error("Invalid EXR tile dimensions");
		for (int64_t y = 0; y < imageHeight; y += tileHeight) {
			for (int64_t x = 0; x < imageWidth; x += tileWidth) {
				exr_chunk_info_t chunk;
				context.check(exr_read_tile_chunk_info(context.handle, 0, int(x / tileWidth), int(y / tileHeight), 0, 0, &chunk));
				decode(chunk, x, y);
			}
		}
	}
	if (gray) {
		for (size_t i = 0; i < pixels; ++i)
			output.get()[4 * i + 1] = output.get()[4 * i + 2] = output.get()[4 * i + 3] = output.get()[4 * i];
	}
	if (flip) {
		size_t stride = size_t(imageWidth) * 4;
		for (int y = 0; y < imageHeight / 2; ++y)
			std::swap_ranges(output.get() + y * stride, output.get() + (y + 1) * stride,
				output.get() + (imageHeight - 1 - y) * stride);
	}
	*data = output.release();
	*width = imageWidth;
	*height = imageHeight;
}

} // namespace krr::exr
