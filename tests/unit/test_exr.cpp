#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <openexr.h>
#include "util/exr.h"

namespace fs = std::filesystem;
using Pixels = std::unique_ptr<float, decltype(&std::free)>;

void require(bool condition, const std::string &message) {
	if (!condition) throw std::runtime_error(message);
}

void check(exr_result_t result) {
	require(result == EXR_ERR_SUCCESS, exr_get_default_error_message(result));
}

struct Writer {
	exr_context_t context = nullptr;
	exr_encode_pipeline_t encoder = EXR_ENCODE_PIPELINE_INITIALIZER;
	~Writer() {
		if (context) {
			exr_encoding_destroy(context, &encoder);
			exr_finish(&context);
		}
	}
};

void write(const fs::path &path, int width, int height, const std::vector<float> &pixels,
	const std::vector<std::string> &channels, exr_compression_t compression,
	exr_pixel_type_t type = EXR_PIXEL_FLOAT, bool tiled = false) {
	Writer writer;
	check(exr_start_write(&writer.context, path.string().c_str(), EXR_WRITE_FILE_DIRECTLY, nullptr));
	int part = 0;
	check(exr_add_part(writer.context, nullptr,
		tiled ? EXR_STORAGE_TILED : EXR_STORAGE_SCANLINE, &part));
	check(exr_initialize_required_attr_simple(writer.context, part, width, height, compression));
	exr_attr_box2i_t window{};
	window.min.x = -7;
	window.min.y = 11;
	window.max.x = window.min.x + width - 1;
	window.max.y = window.min.y + height - 1;
	check(exr_set_data_window(writer.context, part, &window));
	for (const auto &channel : channels)
		check(exr_add_channel(writer.context, part, channel.c_str(), type,
			EXR_PERCEPTUALLY_LOGARITHMIC, 1, 1));
	if (tiled)
		check(exr_set_tile_descriptor(writer.context, part, 8, 16, EXR_TILE_ONE_LEVEL, EXR_TILE_ROUND_DOWN));
	check(exr_write_header(writer.context));
	int32_t rows = 16;
	if (!tiled) check(exr_get_scanlines_per_chunk(writer.context, part, &rows));
	bool initialized = false;
	for (int y = 0; y < height; y += rows) {
		for (int x = 0; x < width; x += tiled ? 8 : width) {
			exr_chunk_info_t chunk{};
			if (tiled)
				check(exr_write_tile_chunk_info(writer.context, part, x / 8, y / 16, 0, 0, &chunk));
			else
				check(exr_write_scanline_chunk_info(writer.context, part, window.min.y + y, &chunk));
			if (initialized)
				check(exr_encoding_update(writer.context, part, &chunk, &writer.encoder));
			else
				check(exr_encoding_initialize(writer.context, part, &chunk, &writer.encoder));
			for (int c = 0; c < writer.encoder.channel_count; ++c) {
				auto &channel = writer.encoder.channels[c];
				const char components[] = "RGBA";
				const char *component = std::strchr(components, channel.channel_name[0]);
				int index = component ? int(component - components) : 0;
				channel.encode_from_ptr = reinterpret_cast<const uint8_t *>(
					pixels.data() + (y * width + x) * 4 + index);
				channel.user_data_type = EXR_PIXEL_FLOAT;
				channel.user_bytes_per_element = sizeof(float);
				channel.user_pixel_stride = 4 * sizeof(float);
				channel.user_line_stride = width * 4 * sizeof(float);
			}
			if (!initialized)
				check(exr_encoding_choose_default_routines(writer.context, part, &writer.encoder));
			check(exr_encoding_run(writer.context, part, &writer.encoder));
			initialized = true;
		}
	}
	check(exr_encoding_destroy(writer.context, &writer.encoder));
	check(exr_finish(&writer.context));
}

Pixels load(const fs::path &path, int &width, int &height, bool flip = false) {
	float *data = nullptr;
	krr::exr::load(path.string().c_str(), &data, &width, &height, flip);
	return Pixels(data, &std::free);
}

void compare(const fs::path &path, const std::vector<float> &expected, int width, int height,
	float tolerance = 0) {
	for (bool flip : {false, true}) {
		int loadedWidth = 0, loadedHeight = 0;
		auto pixels = load(path, loadedWidth, loadedHeight, flip);
		require(pixels && loadedWidth == width && loadedHeight == height, "EXR dimensions differ");
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width * 4; ++x) {
				float value = expected[((flip ? height - y - 1 : y) * width * 4) + x];
				float actual = pixels.get()[y * width * 4 + x];
				require(std::isfinite(actual) && std::abs(actual - value) <=
					tolerance * std::max(1.f, std::abs(value)),
					path.filename().string() + ": EXR channel, row or value differs");
			}
		}
	}
}

void rejects(const fs::path &path) {
	float *data = nullptr;
	int width = 123, height = 456;
	try {
		krr::exr::load(path.string().c_str(), &data, &width, &height);
	} catch (const std::runtime_error &error) {
		require(!data && !width && !height, "Failed EXR load returned partial output");
		require(std::strlen(error.what()) > 0, "EXR failure has no diagnostic");
		return;
	}
	std::free(data);
	throw std::runtime_error("Invalid EXR was accepted: " + path.string());
}

int main(int argc, char **argv) {
	try {
		require(argc >= 2, "Usage: krr_test_exr <artifact-directory> [external.exr]");
		fs::path artifacts = argv[1];
		fs::create_directories(artifacts);
		constexpr int width = 17, height = 259;
		std::vector<float> pixels(width * height * 4);
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				int offset = (y * width + x) * 4;
				pixels[offset] = -.5f + y * .0078125f;
				pixels[offset + 1] = .25f + x * .125f;
				pixels[offset + 2] = 1.f + (x + y) * .03125f;
				pixels[offset + 3] = .25f * (1 + (x + y) % 4);
			}
		}
		for (auto compression : {EXR_COMPRESSION_ZIP, EXR_COMPRESSION_PIZ,
				EXR_COMPRESSION_DWAA, EXR_COMPRESSION_DWAB}) {
			for (auto type : {EXR_PIXEL_HALF, EXR_PIXEL_FLOAT}) {
				auto path = artifacts / ("rgba_" + std::to_string(compression) + "_" + std::to_string(type) + ".exr");
				write(path, width, height, pixels, {"R", "G", "B", "A"}, compression, type);
				bool lossy = compression == EXR_COMPRESSION_DWAA || compression == EXR_COMPRESSION_DWAB;
				compare(path, pixels, width, height, lossy ? .02f : 0);
			}
		}
		auto tiled = artifacts / "tiled.exr";
		write(tiled, width, height, pixels, {"R", "G", "B", "A"}, EXR_COMPRESSION_ZIP, EXR_PIXEL_HALF, true);
		compare(tiled, pixels, width, height);
		auto rgb = artifacts / "rgb.exr";
		write(rgb, width, height, pixels, {"R", "G", "B"}, EXR_COMPRESSION_ZIP);
		auto expected = pixels;
		for (size_t i = 3; i < expected.size(); i += 4) expected[i] = 1;
		compare(rgb, expected, width, height);
		auto gray = artifacts / "gray.exr";
		write(gray, width, height, pixels, {"Y"}, EXR_COMPRESSION_PIZ);
		for (size_t i = 0; i < expected.size(); ++i) expected[i] = pixels[(i / 4) * 4];
		compare(gray, expected, width, height);
		auto incomplete = artifacts / "missing_blue.exr";
		write(incomplete, width, height, pixels, {"R", "G"}, EXR_COMPRESSION_ZIP);
		rejects(incomplete);
		auto invalid = artifacts / "invalid.exr";
		std::ofstream(invalid, std::ios::binary) << "Invalid EXR data";
		rejects(invalid);
		rejects(artifacts / "missing" / "missing.exr");
		auto truncated = artifacts / "truncated.exr";
		fs::copy_file(rgb, truncated, fs::copy_options::overwrite_existing);
		fs::resize_file(truncated, fs::file_size(truncated) / 2);
		rejects(truncated);
		auto special = artifacts / "nonfinite.exr";
		write(special, 1, 1, {-2.f, std::numeric_limits<float>::quiet_NaN(),
			std::numeric_limits<float>::infinity(), 1.f}, {"R", "G", "B", "A"}, EXR_COMPRESSION_ZIP);
		int loadedWidth = 0, loadedHeight = 0;
		auto result = load(special, loadedWidth, loadedHeight);
		require(result.get()[0] == -2 && std::isnan(result.get()[1]) &&
			std::isinf(result.get()[2]) && result.get()[3] == 1, "EXR float values were altered");
		if (argc > 2) {
			auto external = load(argv[2], loadedWidth, loadedHeight);
			double sum = 0;
			for (size_t i = 0; i < size_t(loadedWidth) * loadedHeight * 4; ++i) {
				require(std::isfinite(external.get()[i]), "External EXR contains nonfinite pixels");
				if (i % 4 != 3) sum += std::abs(external.get()[i]);
			}
			require(sum > 0, "External EXR is black");
			std::cout << "External EXR decoded: " << loadedWidth << " x " << loadedHeight << '\n';
		}
		std::cout << "EXR decoding tests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
