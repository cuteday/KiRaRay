#pragma once

#include <json.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace krr::hydra {

struct WavefrontSettings {
	int maxDepth{10};
	bool nee{true};
	float rr{0.8f};

	bool set(const std::string &name, const nlohmann::json &value) {
		if (name == "krr:maxDepth") {
			if (!value.is_number_integer() || value < 0 || value > std::numeric_limits<int>::max())
				throw std::invalid_argument("KiRaRay maximum path depth must be a nonnegative integer");
			maxDepth = value.get<int>();
		} else if (name == "krr:nee") {
			if (!value.is_boolean())
				throw std::invalid_argument("KiRaRay next event estimation must be a boolean");
			nee = value.get<bool>();
		} else if (name == "krr:rr") {
			if (!value.is_number())
				throw std::invalid_argument("KiRaRay Russian roulette survival must be a number in (0, 1]");
			const double probability = value.get<double>();
			if (!std::isfinite(probability) || probability <= 0.0 || probability > 1.0 ||
				float(probability) == 0.f)
				throw std::invalid_argument("KiRaRay Russian roulette survival must be a number in (0, 1]");
			rr = float(probability);
		} else return false;
		return true;
	}
	nlohmann::json params() const { return {{"max_depth", maxDepth}, {"nee", nee}, {"rr", rr}}; }
};

} // namespace krr::hydra
