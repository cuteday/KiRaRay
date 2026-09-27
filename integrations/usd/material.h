#pragma once

#include "texture.h"

NAMESPACE_BEGIN(krr)
namespace interop {

struct MaterialInput {
	std::string node;
	std::string output;
	std::string type;
	json value;
	std::string colorSpace;
	std::string error;
};

struct MaterialNodeInput {
	std::string identifier;
	std::map<std::string, MaterialInput> inputs;
};

struct MaterialNetwork {
	std::string name;
	std::string terminal;
	std::map<std::string, MaterialNodeInput> nodes;
	std::vector<std::string> diagnostics;
	double emissionLuminanceScale{1.0};
};

Material::SharedPtr translateMaterial(const MaterialNetwork &network,
									  std::vector<std::string> *diagnostics = nullptr);

} // namespace interop
NAMESPACE_END(krr)
