#pragma once

#include "material/program.h"
#include <array>

NAMESPACE_BEGIN(krr)

class Texture;
enum class MaterialWrap : uint8_t {
	Repeat,
	Clamp,
	Mirror,
	Border
};
enum class MaterialFilter : uint8_t {
	Linear,
	Closest
};
enum class MaterialProgramKind : uint8_t {
	Constant,
	Simple,
	Program
};

struct MaterialTexture {
	std::shared_ptr<Texture> texture;
	MaterialWrap wrapU{MaterialWrap::Repeat}, wrapV{MaterialWrap::Repeat};
	MaterialFilter filter{MaterialFilter::Linear};
	MaterialValue fallback;
	std::string uvSet{"st"};
};

struct MaterialNode {
	MaterialOp op{MaterialOp::Constant};
	MaterialValueType type{MaterialValueType::Float};
	std::array<int, 5> inputs{{-1, -1, -1, -1, -1}};
	MaterialValue value;
	int auxiliary{0};
	std::string source;
};

struct MaterialDescription {
	MaterialModel model{MaterialModel::OpenPBR};
	std::vector<MaterialNode> nodes;
	std::vector<MaterialTexture> textures;
	std::array<int, MaterialParameterCount> outputs;
	MaterialDescription() { outputs.fill(-1); }
	int add(MaterialNode node) {
		nodes.push_back(std::move(node));
		return int(nodes.size() - 1);
	}
	int constant(MaterialValue value, MaterialValueType type = MaterialValueType::Float) {
		MaterialNode node;
		node.type  = type;
		node.value = value;
		return add(std::move(node));
	}
	void set(MaterialParameter parameter, int node) { outputs[int(parameter)] = node; }
};

struct CompiledMaterial {
	MaterialModel model{MaterialModel::OpenPBR};
	MaterialProgramKind kind{MaterialProgramKind::Constant};
	MaterialValues defaults;
	std::vector<MaterialInstruction> surface, opacity, emission;
	std::vector<MaterialSimpleBinding> simple;
	std::vector<MaterialValue> uniforms;
	std::vector<MaterialTexture> textures;
	bool hasEmission{false};
	uint32_t authoredMask{0};
};

MaterialValues defaultMaterialValues(MaterialModel model);
CompiledMaterial compileMaterial(const MaterialDescription &description);

NAMESPACE_END(krr)
