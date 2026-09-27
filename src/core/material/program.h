#pragma once

#include "common.h"
#include <cstdint>

NAMESPACE_BEGIN(krr)

enum class MaterialModel : uint8_t {
	OpenPBR,
	PreviewSurface,
	Error
};

KRR_CALLABLE float materialEmissionIntensity(float luminance, MaterialModel model) {
	// One renderer radiance unit corresponds to 1000 nits.
	constexpr float nitsPerRadianceUnit = 1000.f;
	return fmaxf(luminance, 0.f) / (model == MaterialModel::OpenPBR ? nitsPerRadianceUnit : 1.f);
}

enum class MaterialValueType : uint8_t {
	Float,
	Boolean,
	Vector2,
	Vector3,
	Color3,
	Vector4,
	Color4
};
enum class MaterialOp : uint8_t {
	Constant,
	Uniform,
	UV,
	Normal,
	Tangent,
	Bitangent,
	Image,
	Add,
	Subtract,
	Multiply,
	Divide,
	MultiplyAdd,
	Minimum,
	Maximum,
	Power,
	Less,
	Greater,
	Equal,
	Mix,
	Clamp,
	Remap,
	Convert,
	Extract,
	Combine,
	Normalize,
	Dot,
	Cross,
	Rotate3D,
	NormalMap,
	IfGreater,
	IfGreaterEqual,
	IfEqual,
	Store
};
enum class MaterialParameter : uint8_t {
	BaseColor,
	BaseWeight,
	Metalness,
	DiffuseRoughness,
	SpecularColor,
	SpecularWeight,
	SpecularIor,
	SpecularRoughness,
	SpecularAnisotropy,
	TransmissionColor,
	TransmissionWeight,
	CoatColor,
	CoatWeight,
	CoatIor,
	CoatRoughness,
	CoatNormal,
	FuzzColor,
	FuzzWeight,
	FuzzRoughness,
	Normal,
	Tangent,
	Opacity,
	ThinWalled,
	EmissionColor,
	EmissionLuminance,
	Count
};

constexpr int MaterialParameterCount = int(MaterialParameter::Count);
constexpr int MaterialRegisterLimit	 = 32;
constexpr int MaterialNodeLimit		 = 256;

struct MaterialValue {
	float data[4]{};
	KRR_CALLABLE MaterialValue() = default;
	KRR_CALLABLE explicit MaterialValue(float value) : data{value, value, value, value} {}
	KRR_CALLABLE MaterialValue(float x, float y, float z, float w = 0) : data{x, y, z, w} {}
	KRR_CALLABLE float &operator[](int index) { return data[index]; }
	KRR_CALLABLE float operator[](int index) const { return data[index]; }
};

struct MaterialContext {
	MaterialValue uv;
	MaterialValue normal{0, 0, 1};
	MaterialValue tangent{1, 0, 0};
	MaterialValue bitangent{0, 1, 0};
};

struct MaterialInstruction {
	MaterialOp op{MaterialOp::Constant};
	MaterialValueType type{MaterialValueType::Float};
	uint8_t destination{0};
	uint8_t inputs[5]{};
	uint16_t auxiliary{0};
	MaterialValue value;
};

struct MaterialProgramView {
	const MaterialInstruction *instructions{nullptr};
	uint32_t count{0};
};

struct MaterialSimpleBinding {
	MaterialParameter parameter{MaterialParameter::BaseColor};
	uint16_t texture{0};
	int8_t channel{-1};
	bool emission{false};
};

struct MaterialValues {
	MaterialValue values[MaterialParameterCount]{};
	KRR_CALLABLE MaterialValue &operator[](MaterialParameter parameter) {
		return values[int(parameter)];
	}
	KRR_CALLABLE const MaterialValue &operator[](MaterialParameter parameter) const {
		return values[int(parameter)];
	}
};

KRR_CALLABLE int materialComponents(MaterialValueType type) {
	switch (type) {
		case MaterialValueType::Float:
		case MaterialValueType::Boolean:
			return 1;
		case MaterialValueType::Vector2:
			return 2;
		case MaterialValueType::Vector3:
		case MaterialValueType::Color3:
			return 3;
		default:
			return 4;
	}
}

KRR_CALLABLE float materialDot(MaterialValue a, MaterialValue b) {
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

KRR_CALLABLE MaterialValue materialNormalize(MaterialValue value) {
	float length = sqrtf(materialDot(value, value));
	if (length > 0)
		for (int i = 0; i < 3; ++i) value[i] /= length;
	return value;
}

KRR_CALLABLE MaterialValue materialCross(MaterialValue a, MaterialValue b) {
	return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

KRR_CALLABLE MaterialValue evaluateMaterialOperation(const MaterialInstruction &instruction,
													 MaterialValue a, MaterialValue b = {},
													 MaterialValue c = {}, MaterialValue d = {},
													 MaterialValue e = {}) {
	MaterialValue value;
	if (instruction.op == MaterialOp::Extract) return MaterialValue(a[instruction.auxiliary]);
	if (instruction.op == MaterialOp::Normalize) return materialNormalize(a);
	if (instruction.op == MaterialOp::Dot) return MaterialValue(materialDot(a, b));
	if (instruction.op == MaterialOp::Cross) return materialCross(a, b);
	if (instruction.op == MaterialOp::Combine) return {a[0], b[0], c[0], d[0]};
	if (instruction.op == MaterialOp::Convert) {
		int from = instruction.auxiliary;
		int to	 = materialComponents(instruction.type);
		if (to == 1 || from == 1) return MaterialValue(a[0]);
		for (int i = 0; i < to; ++i) value[i] = i < from ? a[i] : (i == 3 ? 1.f : 0.f);
		return value;
	}
	if (instruction.op == MaterialOp::Rotate3D) {
		MaterialValue axis = materialNormalize(c), cross = materialCross(axis, a);
		float angle	   = b[0] * 0.017453292519943295f;
		float cosAngle = cosf(angle), sinAngle = sinf(angle), projection = materialDot(axis, a);
		for (int i = 0; i < 3; ++i)
			value[i] =
				a[i] * cosAngle + cross[i] * sinAngle + axis[i] * projection * (1 - cosAngle);
		return value;
	}
	if (instruction.op == MaterialOp::NormalMap) {
		float x = (a[0] * 2 - 1) * b[0], y = (a[1] * 2 - 1) * b[0], z = a[2] * 2 - 1;
		for (int i = 0; i < 3; ++i) value[i] = d[i] * x + e[i] * y + c[i] * z;
		return materialNormalize(value);
	}
	if (instruction.op == MaterialOp::IfGreater) return a[0] > b[0] ? c : d;
	if (instruction.op == MaterialOp::IfGreaterEqual) return a[0] >= b[0] ? c : d;
	if (instruction.op == MaterialOp::IfEqual) return a[0] == b[0] ? c : d;
	for (int i = 0; i < 4; ++i) {
		switch (instruction.op) {
			case MaterialOp::Add:
				value[i] = a[i] + b[i];
				break;
			case MaterialOp::Subtract:
				value[i] = a[i] - b[i];
				break;
			case MaterialOp::Multiply:
				value[i] = a[i] * b[i];
				break;
			case MaterialOp::Divide:
				value[i] = b[i] == 0 ? 0 : a[i] / b[i];
				break;
			case MaterialOp::MultiplyAdd:
				value[i] = a[i] * b[i] + c[i];
				break;
			case MaterialOp::Minimum:
				value[i] = fminf(a[i], b[i]);
				break;
			case MaterialOp::Maximum:
				value[i] = fmaxf(a[i], b[i]);
				break;
			case MaterialOp::Power:
				value[i] = powf(fmaxf(a[i], 0), b[i]);
				break;
			case MaterialOp::Less:
				value[i] = a[i] < b[i];
				break;
			case MaterialOp::Greater:
				value[i] = a[i] > b[i];
				break;
			case MaterialOp::Equal:
				value[i] = a[i] == b[i];
				break;
			case MaterialOp::Mix:
				value[i] = a[i] * (1 - c[i]) + b[i] * c[i];
				break;
			case MaterialOp::Clamp:
				value[i] = fminf(fmaxf(a[i], b[i]), c[i]);
				break;
			case MaterialOp::Remap:
				value[i] =
					b[i] == c[i] ? d[i] : d[i] + (a[i] - b[i]) / (c[i] - b[i]) * (e[i] - d[i]);
				break;
			default:
				break;
		}
	}
	return value;
}

template <typename ImageSampler>
KRR_CALLABLE void evaluateMaterialProgram(MaterialProgramView program,
										  const MaterialValue *uniforms,
										  const MaterialContext &context, ImageSampler sampleImage,
										  MaterialValues &result) {
	MaterialValue registers[MaterialRegisterLimit];
	for (uint32_t index = 0; index < program.count; ++index) {
		const MaterialInstruction &instruction = program.instructions[index];
		MaterialValue value;
		switch (instruction.op) {
			case MaterialOp::Constant:
				value = instruction.value;
				break;
			case MaterialOp::Uniform:
				value = uniforms[instruction.auxiliary];
				break;
			case MaterialOp::UV:
				value = context.uv;
				break;
			case MaterialOp::Normal:
				value = context.normal;
				break;
			case MaterialOp::Tangent:
				value = context.tangent;
				break;
			case MaterialOp::Bitangent:
				value = context.bitangent;
				break;
			case MaterialOp::Image:
				value = sampleImage(instruction.auxiliary, registers[instruction.inputs[0]]);
				break;
			case MaterialOp::Store:
				result.values[instruction.auxiliary] = registers[instruction.inputs[0]];
				continue;
			default:
				value = evaluateMaterialOperation(
					instruction, registers[instruction.inputs[0]], registers[instruction.inputs[1]],
					registers[instruction.inputs[2]], registers[instruction.inputs[3]],
					registers[instruction.inputs[4]]);
		}
		registers[instruction.destination] = value;
	}
}

NAMESPACE_END(krr)
