#include "material/description.h"

#include <cstring>
#include <functional>
#include <map>

NAMESPACE_BEGIN(krr)

MaterialValues defaultMaterialValues(MaterialModel model) {
	MaterialValues values;
	for (MaterialParameter parameter :
		 {MaterialParameter::BaseColor, MaterialParameter::SpecularColor,
		  MaterialParameter::TransmissionColor, MaterialParameter::CoatColor,
		  MaterialParameter::FuzzColor, MaterialParameter::EmissionColor})
		values[parameter] = MaterialValue(1);
	values[MaterialParameter::BaseColor] =
		MaterialValue(model == MaterialModel::PreviewSurface ? 0.18f : 0.8f);
	for (MaterialParameter parameter :
		 {MaterialParameter::BaseWeight, MaterialParameter::SpecularWeight,
		  MaterialParameter::Opacity})
		values[parameter] = MaterialValue(1);
	values[MaterialParameter::SpecularIor] = values[MaterialParameter::CoatIor] =
		MaterialValue(1.5f);
	values[MaterialParameter::SpecularRoughness] = values[MaterialParameter::FuzzRoughness] =
		MaterialValue(0.5f);
	values[MaterialParameter::CoatRoughness] =
		MaterialValue(model == MaterialModel::PreviewSurface ? 0.01f : 0.03f);
	values[MaterialParameter::Normal] = values[MaterialParameter::CoatNormal] =
		MaterialValue(0, 0, 1);
	values[MaterialParameter::Tangent] = MaterialValue(1, 0, 0);
	if (model == MaterialModel::Error) {
		values[MaterialParameter::BaseColor]	  = MaterialValue(1, 0, 1);
		values[MaterialParameter::SpecularWeight] = MaterialValue(0);
	}
	if (model == MaterialModel::PreviewSurface)
		values[MaterialParameter::EmissionColor] = MaterialValue(0);
	return values;
}

namespace {
int arity(MaterialOp op, MaterialValueType type) {
	switch (op) {
		case MaterialOp::Constant:
		case MaterialOp::Uniform:
		case MaterialOp::UV:
		case MaterialOp::Normal:
		case MaterialOp::Tangent:
		case MaterialOp::Bitangent:
			return 0;
		case MaterialOp::Image:
		case MaterialOp::Convert:
		case MaterialOp::Extract:
		case MaterialOp::Normalize:
			return 1;
		case MaterialOp::Add:
		case MaterialOp::Subtract:
		case MaterialOp::Multiply:
		case MaterialOp::Divide:
		case MaterialOp::Minimum:
		case MaterialOp::Maximum:
		case MaterialOp::Power:
		case MaterialOp::Less:
		case MaterialOp::Greater:
		case MaterialOp::Equal:
		case MaterialOp::Dot:
		case MaterialOp::Cross:
			return 2;
		case MaterialOp::MultiplyAdd:
		case MaterialOp::Mix:
		case MaterialOp::Clamp:
		case MaterialOp::Rotate3D:
			return 3;
		case MaterialOp::Combine:
			return materialComponents(type);
		case MaterialOp::IfGreater:
		case MaterialOp::IfGreaterEqual:
		case MaterialOp::IfEqual:
			return 4;
		case MaterialOp::NormalMap:
		case MaterialOp::Remap:
			return 5;
		default:
			throw std::invalid_argument("Invalid authored material operation");
	}
}

bool finite(MaterialValue value) {
	for (float component : value.data)
		if (!std::isfinite(component)) return false;
	return true;
}

int parameterComponents(MaterialParameter parameter) {
	switch (parameter) {
		case MaterialParameter::BaseColor:
		case MaterialParameter::SpecularColor:
		case MaterialParameter::TransmissionColor:
		case MaterialParameter::CoatColor:
		case MaterialParameter::FuzzColor:
		case MaterialParameter::EmissionColor:
		case MaterialParameter::Normal:
		case MaterialParameter::CoatNormal:
		case MaterialParameter::Tangent:
			return 3;
		default:
			return 1;
	}
}

bool identical(const MaterialNode &a, const MaterialNode &b) {
	return a.op == b.op && a.type == b.type && a.inputs == b.inputs && a.auxiliary == b.auxiliary &&
		   std::memcmp(a.value.data, b.value.data, sizeof(a.value.data)) == 0;
}

class Compiler {
public:
	Compiler(const MaterialDescription &description) :
		description(description),
		state(description.nodes.size(), 0),
		resolved(description.nodes.size(), -1) {
		result.model	= description.model;
		result.defaults = defaultMaterialValues(description.model);
	}

	CompiledMaterial compile() {
		std::array<int, MaterialParameterCount> roots;
		roots.fill(-1);
		for (int parameter = 0; parameter < MaterialParameterCount; ++parameter) {
			int source = description.outputs[parameter];
			if (source == -1) continue;
			result.authoredMask |= 1u << parameter;
			int node = visit(source);
			if (materialComponents(nodes[node].type) !=
				parameterComponents(MaterialParameter(parameter)))
				fail(nodes[node], "Material parameter has the wrong value type");
			if (nodes[node].op == MaterialOp::Constant)
				result.defaults.values[parameter] = nodes[node].value;
			else
				roots[parameter] = node;
		}
		result.surface = emit(roots);
		auto opacity = roots, emission = roots;
		bool emissionLayers = roots[int(MaterialParameter::CoatWeight)] >= 0 ||
							  roots[int(MaterialParameter::FuzzWeight)] >= 0 ||
							  result.defaults[MaterialParameter::CoatWeight][0] > 0 ||
							  result.defaults[MaterialParameter::FuzzWeight][0] > 0;
		for (int parameter = 0; parameter < MaterialParameterCount; ++parameter) {
			if (parameter != int(MaterialParameter::Opacity)) opacity[parameter] = -1;
			if (parameter == int(MaterialParameter::Opacity) ||
				(!emissionLayers && parameter != int(MaterialParameter::EmissionColor) &&
				 parameter != int(MaterialParameter::EmissionLuminance) &&
				 parameter != int(MaterialParameter::ThinWalled)))
				emission[parameter] = -1;
		}
		result.opacity	   = emit(opacity);
		result.emission	   = emit(emission);
		result.hasEmission = roots[int(MaterialParameter::EmissionLuminance)] >= 0 ||
							 result.defaults[MaterialParameter::EmissionLuminance][0] > 0;
		if (roots[int(MaterialParameter::EmissionColor)] < 0) {
			const auto &color = result.defaults[MaterialParameter::EmissionColor];
			result.hasEmission &= color[0] > 0 || color[1] > 0 || color[2] > 0;
		}
		result.kind =
			result.surface.empty() ? MaterialProgramKind::Constant : MaterialProgramKind::Program;
		bool simple = !result.surface.empty();
		for (int parameter = 0; parameter < MaterialParameterCount && simple; ++parameter)
			if (roots[parameter] >= 0) {
				int index = roots[parameter], channel = -1;
				while (nodes[index].op == MaterialOp::Convert ||
					   nodes[index].op == MaterialOp::Extract) {
					if (nodes[index].op == MaterialOp::Extract)
						channel = nodes[index].auxiliary;
					else {
						int from = materialComponents(nodes[nodes[index].inputs[0]].type);
						int to	 = materialComponents(nodes[index].type);
						if (from > 1 && to > from) {
							simple = false;
							break;
						}
						if (to == 1) channel = 0;
					}
					index = nodes[index].inputs[0];
				}
				if (!simple) break;
				const MaterialNode &node = nodes[index];
				if (node.op != MaterialOp::Image || nodes[node.inputs[0]].op != MaterialOp::UV)
					simple = false;
				else
					result.simple.push_back({MaterialParameter(parameter), uint16_t(node.auxiliary),
											 int8_t(channel), emission[parameter] >= 0});
			}
		if (simple)
			result.kind = MaterialProgramKind::Simple;
		else
			result.simple.clear();
		return std::move(result);
	}

private:
	[[noreturn]] void fail(const MaterialNode &node, const std::string &message) const {
		throw std::invalid_argument(node.source.empty() ? message : node.source + ": " + message);
	}

	int visit(int index) {
		if (index < 0 || index >= int(description.nodes.size()))
			throw std::invalid_argument("Material graph has an invalid node reference");
		if (state[index] == 1) fail(description.nodes[index], "Material graph contains a cycle");
		if (state[index] == 2) return resolved[index];
		if (++depth > MaterialNodeLimit)
			fail(description.nodes[index], "Material exceeds the expression depth limit");
		state[index]	  = 1;
		MaterialNode node = description.nodes[index];
		if (int(node.type) > int(MaterialValueType::Color4))
			fail(node, "Invalid material value type");
		int count	  = arity(node.op, node.type);
		bool constant = count > 0 && node.op != MaterialOp::Image;
		for (int input = 0; input < count; ++input) {
			node.inputs[input] = visit(node.inputs[input]);
			constant &= nodes[node.inputs[input]].op == MaterialOp::Constant;
		}
		for (int input = count; input < 5; ++input) node.inputs[input] = -1;
		validateTypes(node, count);
		if (node.op == MaterialOp::Constant || node.op == MaterialOp::Uniform) {
			if (!finite(node.value)) fail(node, "Material values must be finite");
			if (materialComponents(node.type) == 1) node.value = MaterialValue(node.value[0]);
		}
		if (node.op == MaterialOp::Uniform) {
			node.auxiliary = int(result.uniforms.size());
			result.uniforms.push_back(node.value);
		}
		if (node.op == MaterialOp::Image) {
			int binding = node.auxiliary;
			if (binding < 0 || binding >= int(description.textures.size()) ||
				!description.textures[binding].texture)
				fail(node, "Image has no valid texture binding");
			auto found = textureIndices.find(binding);
			if (found == textureIndices.end()) {
				node.auxiliary			= int(result.textures.size());
				textureIndices[binding] = node.auxiliary;
				result.textures.push_back(description.textures[binding]);
			} else
				node.auxiliary = found->second;
			if (materialComponents(nodes[node.inputs[0]].type) < 2)
				fail(node, "Image coordinates require a vector");
		}
		if (node.op == MaterialOp::Extract) {
			if (node.auxiliary < 0 ||
				node.auxiliary >= materialComponents(nodes[node.inputs[0]].type))
				fail(node, "Extract channel is outside the input value");
		}
		if (node.op == MaterialOp::Convert)
			node.auxiliary = materialComponents(nodes[node.inputs[0]].type);
		if (constant) {
			MaterialInstruction instruction;
			instruction.op		  = node.op;
			instruction.type	  = node.type;
			instruction.auxiliary = uint16_t(node.auxiliary);
			MaterialValue inputs[5];
			for (int input = 0; input < count; ++input)
				inputs[input] = nodes[node.inputs[input]].value;
			node.value = evaluateMaterialOperation(instruction, inputs[0], inputs[1], inputs[2],
												   inputs[3], inputs[4]);
			if (!finite(node.value)) fail(node, "Constant material expression is not finite");
			node.op = MaterialOp::Constant;
			node.inputs.fill(-1);
			node.auxiliary = 0;
		}
		int canonical = 0;
		for (; canonical < int(nodes.size()); ++canonical)
			if (identical(nodes[canonical], node)) break;
		if (canonical == int(nodes.size())) {
			if (nodes.size() >= MaterialNodeLimit) fail(node, "Material exceeds the node limit");
			nodes.push_back(std::move(node));
		}
		state[index] = 2;
		--depth;
		resolved[index] = canonical;
		return canonical;
	}

	void validateTypes(const MaterialNode &node, int count) const {
		int components = materialComponents(node.type);
		auto size = [&](int input) { return materialComponents(nodes[node.inputs[input]].type); };
		auto require = [&](bool condition) {
			if (!condition) fail(node, "Material operation has incompatible value types");
		};
		switch (node.op) {
			case MaterialOp::Constant:
			case MaterialOp::Uniform:
				break;
			case MaterialOp::UV:
				require(components == 2);
				break;
			case MaterialOp::Normal:
			case MaterialOp::Tangent:
			case MaterialOp::Bitangent:
				require(components == 3);
				break;
			case MaterialOp::Image:
				require(size(0) >= 2 && components >= 3);
				break;
			case MaterialOp::Convert:
				break;
			case MaterialOp::Extract:
				require(components == 1);
				break;
			case MaterialOp::Combine:
				for (int input = 0; input < count; ++input) require(size(input) == 1);
				break;
			case MaterialOp::Normalize:
				require(components == 3 && size(0) == 3);
				break;
			case MaterialOp::Dot:
				require(components == 1 && size(0) == 3 && size(1) == 3);
				break;
			case MaterialOp::Cross:
				require(components == 3 && size(0) == 3 && size(1) == 3);
				break;
			case MaterialOp::Rotate3D:
				require(components == 3 && size(0) == 3 && size(1) == 1 && size(2) == 3);
				break;
			case MaterialOp::NormalMap:
				require(components == 3 && size(0) >= 3 && size(1) == 1 && size(2) == 3 &&
						size(3) == 3 && size(4) == 3);
				break;
			case MaterialOp::IfGreater:
			case MaterialOp::IfGreaterEqual:
			case MaterialOp::IfEqual:
				require(size(0) == 1 && size(1) == 1 && size(2) == components &&
						size(3) == components);
				break;
			default:
				for (int input = 0; input < count; ++input)
					require(size(input) == 1 || size(input) == components);
		}
	}

	std::vector<MaterialInstruction> emit(const std::array<int, MaterialParameterCount> &roots) {
		std::vector<int> order, uses(nodes.size(), 0), registers(nodes.size(), -1), free;
		std::vector<bool> visited(nodes.size(), false);
		std::vector<std::vector<int>> stores(nodes.size());
		std::function<void(int)> traverse = [&](int index) {
			if (visited[index]) return;
			visited[index] = true;
			for (int input : nodes[index].inputs)
				if (input >= 0) {
					++uses[input];
					traverse(input);
				}
			order.push_back(index);
		};
		for (int parameter = 0; parameter < MaterialParameterCount; ++parameter)
			if (roots[parameter] >= 0) {
				int root = roots[parameter];
				++uses[root];
				stores[root].push_back(parameter);
				traverse(root);
			}
		for (int reg = MaterialRegisterLimit - 1; reg >= 0; --reg) free.push_back(reg);
		std::vector<MaterialInstruction> instructions;
		for (int index : order) {
			const MaterialNode &node = nodes[index];
			MaterialInstruction instruction;
			instruction.op		  = node.op;
			instruction.type	  = node.type;
			instruction.auxiliary = uint16_t(node.auxiliary);
			instruction.value	  = node.value;
			for (int input = 0; input < 5; ++input)
				if (node.inputs[input] >= 0) {
					int source				  = node.inputs[input];
					instruction.inputs[input] = uint8_t(registers[source]);
					if (--uses[source] == 0) free.push_back(registers[source]);
				}
			if (free.empty()) fail(node, "Material exceeds the register limit");
			registers[index] = free.back();
			free.pop_back();
			instruction.destination = uint8_t(registers[index]);
			instructions.push_back(instruction);
			for (int parameter : stores[index]) {
				MaterialInstruction store;
				store.op		= MaterialOp::Store;
				store.inputs[0] = instruction.destination;
				store.auxiliary = uint16_t(parameter);
				instructions.push_back(store);
				if (--uses[index] == 0) free.push_back(registers[index]);
			}
		}
		return instructions;
	}

	const MaterialDescription &description;
	CompiledMaterial result;
	std::vector<int> state, resolved;
	std::vector<MaterialNode> nodes;
	std::map<int, int> textureIndices;
	int depth{0};
};
} // namespace

CompiledMaterial compileMaterial(const MaterialDescription &description) {
	return Compiler(description).compile();
}

NAMESPACE_END(krr)
