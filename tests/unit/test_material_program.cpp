#include "material/description.h"

#include <limits>
#include <stdexcept>

using namespace krr;

namespace {
void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

void near(float actual, float expected) {
	require(std::abs(actual - expected) < 1e-5f, "Unexpected material expression result");
}

int operation(MaterialDescription &description, MaterialOp op, MaterialValueType type,
			  std::initializer_list<int> inputs) {
	MaterialNode node;
	node.op	  = op;
	node.type = type;
	std::copy(inputs.begin(), inputs.end(), node.inputs.begin());
	return description.add(node);
}

MaterialValues evaluate(const CompiledMaterial &material, MaterialContext context = {}) {
	MaterialValues values = material.defaults;
	auto image			  = [](int, MaterialValue) {
		   throw std::runtime_error("Unexpected image sample");
		   return MaterialValue();
	};
	evaluateMaterialProgram({material.surface.data(), uint32_t(material.surface.size())},
							material.uniforms.data(), context, image, values);
	return values;
}

template <typename F> void invalid(F function) {
	bool rejected = false;
	try {
		function();
	} catch (const std::invalid_argument &) {
		rejected = true;
	}
	require(rejected, "Invalid material graph was accepted");
}

void testConstantsAndReachability() {
	MaterialDescription description;
	int a	= description.constant(MaterialValue(0.2f));
	int b	= description.constant(MaterialValue(0.3f));
	int sum = operation(description, MaterialOp::Add, MaterialValueType::Float, {a, b});
	description.set(MaterialParameter::SpecularRoughness, sum);
	MaterialNode unreachable;
	unreachable.op	   = MaterialOp::Image;
	unreachable.source = "unreachable invalid image";
	description.add(unreachable);
	auto material = compileMaterial(description);
	require(material.kind == MaterialProgramKind::Constant && material.surface.empty(),
			"Constant material was not folded");
	near(material.defaults[MaterialParameter::SpecularRoughness][0], 0.5f);
	require(!material.hasEmission && material.textures.empty(),
			"Unreachable resources were retained");
}

void testUniformsCseAndSlices() {
	MaterialDescription description;
	MaterialNode uniform;
	uniform.op	  = MaterialOp::Uniform;
	uniform.value = MaterialValue(0.25f);
	int weight	  = description.add(uniform);
	int one		  = description.constant(MaterialValue(1));
	int a = operation(description, MaterialOp::Subtract, MaterialValueType::Float, {one, weight});
	int b = operation(description, MaterialOp::Subtract, MaterialValueType::Float, {one, weight});
	description.set(MaterialParameter::Opacity, a);
	description.set(MaterialParameter::SpecularRoughness, b);
	description.set(MaterialParameter::EmissionLuminance, weight);
	auto material = compileMaterial(description);
	require(material.uniforms.size() == 1, "Uniform identity was lost");
	int subtracts = 0;
	for (auto &instruction : material.surface) subtracts += instruction.op == MaterialOp::Subtract;
	require(subtracts == 1, "Common subexpressions were not merged");
	require(material.opacity.size() < material.surface.size(),
			"Opacity includes unrelated instructions");
	require(material.emission.size() < material.surface.size(),
			"Emission includes unrelated instructions");
	near(evaluate(material)[MaterialParameter::Opacity][0], 0.75f);
	material.uniforms[0] = MaterialValue(0.5f);
	near(evaluate(material)[MaterialParameter::SpecularRoughness][0], 0.5f);
}

void testProducerGeometryOperations() {
	MaterialDescription description;
	int normal	  = operation(description, MaterialOp::Normal, MaterialValueType::Vector3, {});
	int tangent	  = operation(description, MaterialOp::Tangent, MaterialValueType::Vector3, {});
	int bitangent = operation(description, MaterialOp::Bitangent, MaterialValueType::Vector3, {});
	int color	  = description.constant(MaterialValue(0.75f, 0.5f, 1), MaterialValueType::Color3);
	int strength  = description.constant(MaterialValue(1));
	int mapped	  = operation(description, MaterialOp::NormalMap, MaterialValueType::Vector3,
							  {color, strength, normal, tangent, bitangent});
	description.set(MaterialParameter::Normal, mapped);
	int angle	= description.constant(MaterialValue(90));
	int rotated = operation(description, MaterialOp::Rotate3D, MaterialValueType::Vector3,
							{tangent, angle, normal});
	description.set(MaterialParameter::Tangent, rotated);
	auto values = evaluate(compileMaterial(description));
	near(values[MaterialParameter::Normal][0], 1 / std::sqrt(5.f));
	near(values[MaterialParameter::Normal][2], 2 / std::sqrt(5.f));
	near(values[MaterialParameter::Tangent][0], 0);
	near(values[MaterialParameter::Tangent][1], 1);
}

void testClassificationDependencies() {
	MaterialDescription description;
	MaterialNode uniform;
	uniform.op = MaterialOp::Uniform;
	uniform.value = MaterialValue(.25f);
	int metal = description.add(uniform);
	int one = description.constant(MaterialValue(1));
	int transmission = operation(description, MaterialOp::Subtract,
		MaterialValueType::Float, {one, metal});
	int normal = operation(description, MaterialOp::Normal, MaterialValueType::Vector3, {});
	description.set(MaterialParameter::Metalness, metal);
	description.set(MaterialParameter::TransmissionWeight, transmission);
	description.set(MaterialParameter::Normal, normal);
	description.set(MaterialParameter::SpecularRoughness, transmission);
	auto material = compileMaterial(description);
	require(material.classification.size() < material.surface.size(),
		"Classification includes unrelated instructions");
	for (const auto &instruction : material.classification) {
		require(instruction.op != MaterialOp::Normal, "Classification evaluates a surface normal");
		if (instruction.op == MaterialOp::Store)
			require(instruction.auxiliary == int(MaterialParameter::Metalness) ||
				instruction.auxiliary == int(MaterialParameter::TransmissionWeight),
				"Classification writes an unrelated parameter");
	}
	for (float weight : {0.f, .25f, 1.f}) {
		material.uniforms[0] = MaterialValue(weight);
		MaterialValues actual = material.defaults;
		auto image = [](int, MaterialValue) {
			throw std::runtime_error("Classification sampled an unrelated image");
			return MaterialValue();
		};
		evaluateMaterialProgram({material.classification.data(), uint32_t(material.classification.size())},
			material.uniforms.data(), MaterialContext{}, image, actual);
		auto expected = evaluate(material);
		near(actual[MaterialParameter::Metalness][0], expected[MaterialParameter::Metalness][0]);
		near(actual[MaterialParameter::TransmissionWeight][0],
			expected[MaterialParameter::TransmissionWeight][0]);
		near(actual[MaterialParameter::SpecularRoughness][0],
			material.defaults[MaterialParameter::SpecularRoughness][0]);
	}
	description.set(MaterialParameter::Metalness, one);
	description.set(MaterialParameter::TransmissionWeight, description.constant(MaterialValue(0)));
	material = compileMaterial(description);
	require(!material.surface.empty() && material.classification.empty(),
		"Constant classification retained surface evaluation");
}

void testEmissionDependencies() {
	MaterialDescription description;
	MaterialNode weight;
	weight.op	 = MaterialOp::Uniform;
	weight.value = MaterialValue(.4f);
	int coat	 = description.add(weight);
	description.set(MaterialParameter::CoatWeight, coat);
	weight.value = MaterialValue(1);
	int thin	 = description.add(weight);
	description.set(MaterialParameter::ThinWalled, thin);
	description.set(MaterialParameter::EmissionLuminance,
					description.constant(MaterialValue(1000)));
	int normal = operation(description, MaterialOp::Normal, MaterialValueType::Vector3, {});
	description.set(MaterialParameter::CoatNormal, normal);
	auto material		  = compileMaterial(description);
	MaterialValues values = material.defaults;
	MaterialContext context;
	context.normal = MaterialValue(.2f, .3f, .9f);
	auto image	   = [](int, MaterialValue) { return MaterialValue(); };
	evaluateMaterialProgram({material.emission.data(), uint32_t(material.emission.size())},
							material.uniforms.data(), context, image, values);
	near(values[MaterialParameter::CoatWeight][0], .4f);
	near(values[MaterialParameter::ThinWalled][0], 1);
	near(values[MaterialParameter::CoatNormal][0], .2f);
	require(material.hasEmission, "Authored emission was not recognized");
}

void testLongProgramReusesRegisters() {
	MaterialDescription description;
	MaterialNode uniform;
	uniform.op	  = MaterialOp::Uniform;
	uniform.value = MaterialValue(0);
	int previous  = description.add(uniform);
	int increment = description.constant(MaterialValue(0.01f));
	for (int i = 0; i < 100; ++i)
		previous = operation(description, MaterialOp::Add, MaterialValueType::Float,
							 {previous, increment});
	description.set(MaterialParameter::Opacity, previous);
	near(evaluate(compileMaterial(description))[MaterialParameter::Opacity][0], 1);
}

void testInvalidGraphs() {
	MaterialDescription description;
	MaterialNode cycle;
	cycle.op	 = MaterialOp::Add;
	cycle.inputs = {{0, 0, -1, -1, -1}};
	description.add(cycle);
	description.set(MaterialParameter::Opacity, 0);
	invalid([&] { compileMaterial(description); });
	description.nodes[0].op	   = MaterialOp::Constant;
	description.nodes[0].value = MaterialValue(std::numeric_limits<float>::infinity());
	invalid([&] { compileMaterial(description); });
	description.nodes[0].value = MaterialValue(1);
	description.nodes[0].type  = MaterialValueType::Color3;
	invalid([&] { compileMaterial(description); });
	description.outputs[int(MaterialParameter::Opacity)] = 90;
	invalid([&] { compileMaterial(description); });
}
} // namespace

int main() {
	try {
		testConstantsAndReachability();
		testUniformsCseAndSlices();
		testProducerGeometryOperations();
		testClassificationDependencies();
		testEmissionDependencies();
		testLongProgramReusesRegisters();
		testInvalidGraphs();
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
