#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <json.hpp>

namespace krr {

struct RenderOptions {
	static void validate(int64_t frames, uint64_t seed) {
		if (frames <= 0 || frames > std::numeric_limits<uint32_t>::max())
			throw std::invalid_argument("frames must be a positive 32-bit integer");
	}

	static void validateBenchmark(int64_t frames, int64_t warmup, uint64_t seed) {
		validate(frames, seed);
		if (warmup < 0 || warmup > std::numeric_limits<uint32_t>::max() - frames)
			throw std::invalid_argument("warmup must be nonnegative and warmup + frames must fit in 32 bits");
	}

	static void validateConfig(const nlohmann::json &config) {
		if (!config.is_object()) throw std::invalid_argument("config must be an object");
		if (config.contains("resolution")) {
			const auto &size = config.at("resolution");
			if (!size.is_array() || size.size() != 2)
				throw std::invalid_argument("resolution must contain width and height");
			uint64_t pixels = 1;
			for (const auto &dimension : size) {
				if (!dimension.is_number_integer() || dimension <= 0 ||
					dimension > std::numeric_limits<int>::max())
					throw std::invalid_argument("resolution dimensions must be positive integers");
				pixels *= dimension.get<uint64_t>();
			}
			if (pixels > std::numeric_limits<int>::max())
				throw std::invalid_argument("resolution contains too many pixels");
		}
		if (!config.contains("passes") || !config.at("passes").is_array() ||
			config.at("passes").empty())
			throw std::invalid_argument("config must specify a nonempty passes array");
		for (const auto &pass : config.at("passes")) {
			if (!pass.is_object() || !pass.contains("name") || !pass.at("name").is_string() ||
				pass.at("name").get<std::string>().empty())
				throw std::invalid_argument("each pass must have a name");
			if (pass.contains("params") && !pass.at("params").is_object())
				throw std::invalid_argument("pass params must be an object");
			if (pass.contains("enable") && !pass.at("enable").is_boolean())
				throw std::invalid_argument("pass enable must be a boolean");
		}
		if (!config.contains("model") && !config.contains("scene"))
			throw std::invalid_argument("config must specify a model or scene");
		if (config.contains("model") && !config.at("model").is_string())
			throw std::invalid_argument("model must be a path string");
		if (config.contains("scene") && !config.at("scene").is_object())
			throw std::invalid_argument("scene must be an object");
	}
};

}
