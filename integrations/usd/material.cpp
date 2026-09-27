#include "material.h"
#include "scene/interop.h"
#include "core/material/description.h"

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/Util.h>
#include "tinyexr.h"
#include <cmath>
#include <limits>
#include <sstream>
#include <set>
#ifdef _WIN32
#include <Windows.h>
#endif

NAMESPACE_BEGIN(krr)
namespace interop {
namespace {
namespace mx = MaterialX;

mx::DocumentPtr library() {
	static auto document = [] {
		auto result	  = mx::createDocument();
		fs::path path = KRR_MATERIALX_STDLIB;
#ifdef _WIN32
		HMODULE module = nullptr;
		if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
								   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
							   reinterpret_cast<LPCWSTR>(&library), &module)) {
			wchar_t filename[32768];
			DWORD count = GetModuleFileNameW(module, filename, DWORD(std::size(filename)));
			if (count > 0 && count < std::size(filename)) {
				auto packaged = fs::path(filename).parent_path() / "libraries";
				if (fs::exists(packaged / "bxdf/open_pbr_surface.mtlx")) path = packaged;
			}
		}
#endif
		if (!fs::exists(path / "bxdf/open_pbr_surface.mtlx"))
			throw std::runtime_error("MaterialX definitions are missing; rebuild the SDK or "
									 "reinstall the Blender package");
		mx::loadLibraries({"stdlib", "bxdf"}, mx::FileSearchPath(path.string()), result);
		return result;
	}();
	return document;
}

MaterialValueType valueType(const std::string &type) {
	if (type == "float" || type == "integer" || type == "int" || type == "double")
		return MaterialValueType::Float;
	if (type == "boolean" || type == "bool") return MaterialValueType::Boolean;
	if (type == "vector2" || type == "float2" || type == "texCoord2f")
		return MaterialValueType::Vector2;
	if (type == "vector3" || type == "float3" || type == "normal3f" || type == "vector3f")
		return MaterialValueType::Vector3;
	if (type == "color3" || type == "color3f") return MaterialValueType::Color3;
	if (type == "vector4" || type == "float4") return MaterialValueType::Vector4;
	if (type == "color4" || type == "color4f") return MaterialValueType::Color4;
	throw std::runtime_error("Unsupported material value type: " + type);
}

MaterialValue materialValue(const json &value) {
	if (value.is_boolean()) return MaterialValue(value.get<bool>() ? 1.f : 0.f);
	if (value.is_number()) return MaterialValue(value.get<float>());
	if (!value.is_array() || value.empty() || value.size() > 4)
		throw std::runtime_error("Expected a scalar or a vector material value");
	MaterialValue result;
	for (size_t i = 0; i < value.size(); ++i) result[int(i)] = value[i].get<float>();
	return result;
}

json parseDefault(const mx::InputPtr &input) {
	if (!input || !input->hasValueString()) return nullptr;
	const auto &type = input->getType();
	const auto &text = input->getValueString();
	if (type == "string" || type == "filename") return text;
	if (type == "boolean") return text == "true";
	std::string numbers = text;
	std::replace(numbers.begin(), numbers.end(), ',', ' ');
	std::istringstream stream(numbers);
	json result = json::array();
	float number;
	while (stream >> number) result.push_back(number);
	return result.size() == 1 ? result[0] : result;
}

std::array<std::string, 2> textureWrap(const std::string &filename) {
	std::array<std::string, 2> modes{"black", "black"};
	if (IsEXR(filename.c_str()) != TINYEXR_SUCCESS) return modes;
	EXRVersion version;
	if (ParseEXRVersionFromFile(&version, filename.c_str()) != TINYEXR_SUCCESS)
		throw std::runtime_error("Cannot read EXR texture metadata: " + filename);
	EXRHeader header;
	InitEXRHeader(&header);
	struct Cleanup {
		EXRHeader *header;
		~Cleanup() { FreeEXRHeader(header); }
	} cleanup{&header};
	const char *error = nullptr;
	if (ParseEXRHeaderFromFile(&header, &version, filename.c_str(), &error) != TINYEXR_SUCCESS) {
		std::string message = error ? error : "Cannot read EXR texture metadata";
		if (error) FreeEXRErrorMessage(error);
		throw std::runtime_error(message);
	}
	for (int index = 0; index < header.num_custom_attributes; ++index) {
		const auto &attribute = header.custom_attributes[index];
		if (std::string(attribute.name) != "wrapmodes") continue;
		if (std::string(attribute.type) != "string" || attribute.size <= 0)
			throw std::runtime_error("Unsupported EXR wrapmodes metadata");
		std::string value(reinterpret_cast<const char *>(attribute.value), attribute.size);
		while (!value.empty() && value.back() == '\0') value.pop_back();
		auto separator = value.find(',');
		modes[0]	   = value.substr(0, separator);
		modes[1]	   = separator == std::string::npos ? modes[0] : value.substr(separator + 1);
	}
	return modes;
}

class Lowering {
public:
	Lowering(const MaterialNetwork &network) : network(network) {}
	std::shared_ptr<MaterialDescription> run() {
		if (!std::isfinite(network.emissionLuminanceScale) ||
			network.emissionLuminanceScale <= 0.0 ||
			network.emissionLuminanceScale > std::numeric_limits<float>::max())
			throw std::runtime_error("Emission luminance scale must be finite and positive");
		auto terminal = network.nodes.find(network.terminal);
		if (terminal == network.nodes.end())
			throw std::runtime_error("Material has no supported surface output");
		const auto &surface = terminal->second;
		if (surface.identifier == "ND_open_pbr_surface_surfaceshader")
			openPbr(surface);
		else if (surface.identifier == "UsdPreviewSurface")
			preview(surface);
		else
			throw std::runtime_error("Unsupported surface shader: " + surface.identifier);
		compileMaterial(*description);
		return description;
	}

private:
	const MaterialNetwork &network;
	std::shared_ptr<MaterialDescription> description = std::make_shared<MaterialDescription>();
	std::map<std::pair<std::string, std::string>, int> cache;
	std::set<std::pair<std::string, std::string>> active;

	MaterialInput input(const MaterialNodeInput &node, const std::string &name) {
		auto found = node.inputs.find(name);
		if (found != node.inputs.end() && !found->second.error.empty())
			throw std::runtime_error(node.identifier + "." + name + ": " + found->second.error);
		if (found != node.inputs.end() &&
			(!found->second.node.empty() || !found->second.value.is_null()))
			return found->second;
		auto definition = library()->getNodeDef(node.identifier);
		auto value		= definition ? definition->getActiveInput(name) : nullptr;
		MaterialInput result;
		if (value) {
			result.type	 = value->getType();
			result.value = parseDefault(value);
		}
		return result;
	}
	json constantInput(const MaterialNodeInput &node, const std::string &name,
					   json fallback = nullptr) {
		auto value = input(node, name);
		if (!value.node.empty()) throw std::runtime_error(name + " must be constant");
		return value.value.is_null() ? fallback : value.value;
	}
	int operation(MaterialOp op, MaterialValueType type, std::initializer_list<int> arguments,
				  int auxiliary = 0) {
		MaterialNode node;
		node.op		   = op;
		node.type	   = type;
		node.auxiliary = auxiliary;
		std::copy(arguments.begin(), arguments.end(), node.inputs.begin());
		return description->add(node);
	}
	int convert(int id, MaterialValueType type) {
		if (description->nodes[id].type == type) return id;
		return operation(MaterialOp::Convert, type, {id},
						 materialComponents(description->nodes[id].type));
	}
	int expression(const MaterialInput &value, MaterialValueType fallbackType) {
		if (!value.node.empty()) return convert(node(value.node, value.output), fallbackType);
		if (value.value.is_null())
			throw std::runtime_error("Material input has no value or connection");
		MaterialValue result = materialValue(value.value);
		if (fallbackType == MaterialValueType::Color3 ||
			fallbackType == MaterialValueType::Color4) {
			const auto &space = value.colorSpace;
			if (space == "sRGB" || space == "srgb_texture") {
				for (int channel = 0; channel < 3; ++channel)
					result[channel] = result[channel] <= .04045f
										  ? result[channel] / 12.92f
										  : std::pow((result[channel] + .055f) / 1.055f, 2.4f);
			} else if (!space.empty() && space != "raw" && space != "none" &&
					   space != "lin_rec709" && space != "lin_rec709_scene")
				throw std::runtime_error("Unsupported material input color space: " + space);
		}
		return description->constant(result, fallbackType);
	}
	int argument(const MaterialNodeInput &source, const std::string &name, MaterialValueType type,
				 json fallback = nullptr, MaterialOp geometry = MaterialOp::Constant) {
		auto value = input(source, name);
		if (value.node.empty() && value.value.is_null()) {
			if (geometry != MaterialOp::Constant) return operation(geometry, type, {});
			value.value = fallback;
		}
		return expression(value, type);
	}
	void parameter(const MaterialNodeInput &source, const char *name, MaterialParameter parameter,
				   MaterialValueType type = MaterialValueType::Float) {
		auto value = input(source, name);
		if (!value.node.empty() || !value.value.is_null())
			description->set(parameter, expression(value, type));
	}
	bool enabled(const MaterialNodeInput &source, const char *name, float fallback = 0.f) {
		auto value = input(source, name);
		return !value.node.empty() ||
			   (!value.value.is_null() ? value.value.get<float>() : fallback) != 0.f;
	}
	void rejectActive(const MaterialNodeInput &source, const char *name, json allowed = 0.f) {
		auto value = input(source, name);
		if (!value.node.empty() || (!value.value.is_null() && value.value != allowed))
			throw std::runtime_error(std::string("Unsupported active input: ") + name);
	}
	void openPbr(const MaterialNodeInput &surface) {
		description->model = MaterialModel::OpenPBR;
		rejectActive(surface, "subsurface_weight");
		rejectActive(surface, "thin_film_weight");
		if (enabled(surface, "transmission_weight")) {
			rejectActive(surface, "transmission_depth");
			rejectActive(surface, "transmission_scatter", json::array({0.f, 0.f, 0.f}));
			rejectActive(surface, "transmission_dispersion_scale");
		}
		if (enabled(surface, "coat_weight")) {
			rejectActive(surface, "coat_roughness_anisotropy");
			rejectActive(surface, "coat_darkening", 1.f);
		}
		using P		= MaterialParameter;
		auto active = [&](P parameter) {
			switch (parameter) {
				case P::TransmissionColor:
					return enabled(surface, "transmission_weight");
				case P::CoatColor:
				case P::CoatIor:
				case P::CoatRoughness:
				case P::CoatNormal:
					return enabled(surface, "coat_weight");
				case P::FuzzColor:
				case P::FuzzRoughness:
					return enabled(surface, "fuzz_weight");
				case P::EmissionColor:
					return enabled(surface, "emission_luminance");
				default:
					return true;
			}
		};
		for (const auto &[name, parameter] :
			 std::map<std::string, P>{{"base_weight", P::BaseWeight},
									  {"base_metalness", P::Metalness},
									  {"base_diffuse_roughness", P::DiffuseRoughness},
									  {"specular_weight", P::SpecularWeight},
									  {"specular_ior", P::SpecularIor},
									  {"specular_roughness", P::SpecularRoughness},
									  {"specular_roughness_anisotropy", P::SpecularAnisotropy},
									  {"transmission_weight", P::TransmissionWeight},
									  {"coat_weight", P::CoatWeight},
									  {"coat_ior", P::CoatIor},
									  {"coat_roughness", P::CoatRoughness},
									  {"fuzz_weight", P::FuzzWeight},
									  {"fuzz_roughness", P::FuzzRoughness},
									  {"geometry_opacity", P::Opacity},
									  {"emission_luminance", P::EmissionLuminance}})
			if (active(parameter))
				parameterValue(surface, name, parameter, MaterialValueType::Float);
		if (network.emissionLuminanceScale != 1.0) {
			int luminance = description->outputs[int(P::EmissionLuminance)];
			if (luminance >= 0)
				description->set(P::EmissionLuminance,
					operation(MaterialOp::Multiply, MaterialValueType::Float,
						{luminance, description->constant(MaterialValue(float(network.emissionLuminanceScale)))}));
		}
		for (const auto &[name, parameter] :
			 std::map<std::string, P>{{"base_color", P::BaseColor},
									  {"specular_color", P::SpecularColor},
									  {"transmission_color", P::TransmissionColor},
									  {"coat_color", P::CoatColor},
									  {"fuzz_color", P::FuzzColor},
									  {"emission_color", P::EmissionColor}})
			if (active(parameter))
				parameterValue(surface, name, parameter, MaterialValueType::Color3);
		parameter(surface, "geometry_normal", P::Normal, MaterialValueType::Vector3);
		if (active(P::CoatNormal))
			parameter(surface, "geometry_coat_normal", P::CoatNormal, MaterialValueType::Vector3);
		parameter(surface, "geometry_tangent", P::Tangent, MaterialValueType::Vector3);
		parameter(surface, "geometry_thin_walled", P::ThinWalled, MaterialValueType::Boolean);
	}
	void parameterValue(const MaterialNodeInput &source, const std::string &name,
						MaterialParameter id, MaterialValueType type) {
		parameter(source, name.c_str(), id, type);
	}
	void preview(const MaterialNodeInput &surface) {
		description->model = MaterialModel::PreviewSurface;
		rejectActive(surface, "useSpecularWorkflow");
		rejectActive(surface, "displacement");
		rejectActive(surface, "occlusion", 1.f);
		auto opacity	 = input(surface, "opacity");
		std::string mode = constantInput(surface, "opacityMode", "transparent").get<std::string>();
		if (!enabled(surface, "opacityThreshold")) {
			if (mode != "transparent" && mode != "presence")
				throw std::runtime_error("Unsupported Preview Surface opacityMode: " + mode);
			if (mode == "transparent" &&
				(!opacity.node.empty() ||
				 (!opacity.value.is_null() && opacity.value.get<float>() != 1.f)))
				throw std::runtime_error("Preview Surface transparent opacity is unsupported; use "
										 "presence or opacityThreshold for coverage");
		}
		using P = MaterialParameter;
		parameter(surface, "diffuseColor", P::BaseColor, MaterialValueType::Color3);
		parameter(surface, "metallic", P::Metalness);
		parameter(surface, "roughness", P::SpecularRoughness);
		parameter(surface, "ior", P::SpecularIor);
		parameter(surface, "clearcoat", P::CoatWeight);
		parameter(surface, "clearcoatRoughness", P::CoatRoughness);
		parameter(surface, "opacity", P::Opacity);
		if (enabled(surface, "opacityThreshold")) {
			int opacity	  = argument(surface, "opacity", MaterialValueType::Float, 1.f);
			int threshold = argument(surface, "opacityThreshold", MaterialValueType::Float, 0.f);
			description->set(P::Opacity,
							 operation(MaterialOp::IfGreaterEqual, MaterialValueType::Float,
									   {opacity, threshold, description->constant(MaterialValue(1)),
										description->constant(MaterialValue(0))}));
		}
		parameter(surface, "emissiveColor", P::EmissionColor, MaterialValueType::Color3);
		description->set(P::EmissionLuminance, description->constant(MaterialValue(1)));
		auto normal = input(surface, "normal");
		if (!normal.node.empty() || !normal.value.is_null()) {
			int n	 = expression(normal, MaterialValueType::Vector3);
			int half = description->constant(MaterialValue(.5f), MaterialValueType::Vector3);
			n		 = operation(
				   MaterialOp::Add, MaterialValueType::Vector3,
				   {operation(MaterialOp::Multiply, MaterialValueType::Vector3, {n, half}), half});
			description->set(
				P::Normal,
				operation(MaterialOp::NormalMap, MaterialValueType::Vector3,
						  {n, description->constant(MaterialValue(1)),
						   operation(MaterialOp::Normal, MaterialValueType::Vector3, {}),
						   operation(MaterialOp::Tangent, MaterialValueType::Vector3, {}),
						   operation(MaterialOp::Bitangent, MaterialValueType::Vector3, {})}));
		}
	}
	MaterialWrap wrap(const std::string &mode) {
		if (mode == "periodic" || mode == "repeat") return MaterialWrap::Repeat;
		if (mode == "clamp") return MaterialWrap::Clamp;
		if (mode == "mirror") return MaterialWrap::Mirror;
		if (mode == "constant" || mode == "black") return MaterialWrap::Border;
		throw std::runtime_error("Unsupported texture wrap mode: " + mode);
	}
	int image(const MaterialNodeInput &source, MaterialValueType type, bool preview,
			  const std::string &output) {
		MaterialTexture texture;
		std::string filename = constantInput(source, "file", "").get<std::string>();
		if (filename.empty() || filename.find("<UDIM>") != std::string::npos)
			throw std::runtime_error("Texture needs a resolved single image file");
		std::string colorSpace =
			preview ? constantInput(source, "sourceColorSpace", "auto").get<std::string>()
					: input(source, "file").colorSpace;
		bool srgb = colorSpace == "srgb_texture" || colorSpace == "sRGB";
		if (preview && colorSpace == "auto") srgb = !Image::isHdr(filename);
		if (!colorSpace.empty() && colorSpace != "auto" && colorSpace != "raw" &&
			colorSpace != "lin_rec709" && colorSpace != "lin_rec709_scene" &&
			colorSpace != "none" && !srgb)
			throw std::runtime_error("Unsupported image color space: " + colorSpace);
		texture.texture = Texture::createFromFile(filename, true, srgb);
		if (!texture.texture->hasImage())
			throw std::runtime_error("Cannot read texture: " + filename);
		std::string wrapU = constantInput(source, preview ? "wrapS" : "uaddressmode",
										  preview ? "useMetadata" : "periodic")
								.get<std::string>();
		std::string wrapV = constantInput(source, preview ? "wrapT" : "vaddressmode",
										  preview ? "useMetadata" : "periodic")
								.get<std::string>();
		if (preview && (wrapU == "useMetadata" || wrapV == "useMetadata")) {
			auto metadata = textureWrap(filename);
			if (wrapU == "useMetadata") wrapU = metadata[0];
			if (wrapV == "useMetadata") wrapV = metadata[1];
		}
		texture.wrapU	   = wrap(wrapU);
		texture.wrapV	   = wrap(wrapV);
		std::string filter = constantInput(source, "filtertype", "linear").get<std::string>();
		if (filter != "linear" && filter != "closest")
			throw std::runtime_error("Unsupported texture filter: " + filter);
		texture.filter = filter == "closest" ? MaterialFilter::Closest : MaterialFilter::Linear;
		texture.fallback =
			preview ? MaterialValue(0)
					: materialValue(constantInput(source, "default", json::array({0, 0, 0, 1})));
		if (!constantInput(source, "layer", "").get<std::string>().empty() ||
			!constantInput(source, "framerange", "").get<std::string>().empty())
			throw std::runtime_error("Layered and animated textures are unsupported");
		int binding = int(description->textures.size());
		description->textures.push_back(texture);
		int uv = preview ? argument(source, "st", MaterialValueType::Vector2, json::array({0, 0}))
						 : argument(source, "texcoord", MaterialValueType::Vector2, nullptr,
									MaterialOp::UV);
		int result = operation(MaterialOp::Image, type, {uv}, binding);
		if (preview) {
			int scale =
				argument(source, "scale", MaterialValueType::Color4, json::array({1, 1, 1, 1}));
			int bias =
				argument(source, "bias", MaterialValueType::Color4, json::array({0, 0, 0, 0}));
			result = operation(
				MaterialOp::Add, MaterialValueType::Color4,
				{operation(MaterialOp::Multiply, MaterialValueType::Color4, {result, scale}),
				 bias});
			if (output == "rgb") return convert(result, MaterialValueType::Color3);
			const std::string channels = "rgba";
			if (output.size() == 1 && channels.find(output) != std::string::npos)
				return operation(MaterialOp::Extract, MaterialValueType::Float, {result},
								 int(channels.find(output)));
		}
		return result;
	}
	int node(const std::string &path, const std::string &output) {
		auto key = std::make_pair(path, output);
		if (cache.count(key)) return cache.at(key);
		if (!active.insert(key).second)
			throw std::runtime_error("Material graph contains a cycle at " + path);
		auto found = network.nodes.find(path);
		if (found == network.nodes.end()) throw std::runtime_error("Missing shader node: " + path);
		const auto &source				  = found->second;
		int result						  = lower(source, output);
		description->nodes[result].source = path;
		active.erase(key);
		cache[key] = result;
		return result;
	}
	int lower(const MaterialNodeInput &source, const std::string &output) {
		if (source.identifier == "UsdUVTexture")
			return image(source, MaterialValueType::Color4, true, output);
		if (source.identifier == "UsdPrimvarReader_float2") {
			std::string name = constantInput(source, "varname", "").get<std::string>();
			if (name.empty())
				return argument(source, "fallback", MaterialValueType::Vector2,
								json::array({0, 0}));
			if (name != "st" && name != "UVMap")
				throw std::runtime_error("Only the active UV set is supported: " + name);
			return operation(MaterialOp::UV, MaterialValueType::Vector2, {});
		}
		if (source.identifier == "UsdTransform2d") {
			int uv = argument(source, "in", MaterialValueType::Vector2, json::array({0, 0}));
			uv	   = operation(
				MaterialOp::Multiply, MaterialValueType::Vector2,
				{uv, argument(source, "scale", MaterialValueType::Vector2, json::array({1, 1}))});
			int rotated = operation(
				MaterialOp::Rotate3D, MaterialValueType::Vector3,
				{convert(uv, MaterialValueType::Vector3),
				 argument(source, "rotation", MaterialValueType::Float, 0),
				 description->constant(MaterialValue(0, 0, 1), MaterialValueType::Vector3)});
			return operation(
				MaterialOp::Add, MaterialValueType::Vector2,
				{convert(rotated, MaterialValueType::Vector2),
				 argument(source, "translation", MaterialValueType::Vector2, json::array({0, 0}))});
		}
		auto definition = library()->getNodeDef(source.identifier);
		if (!definition) throw std::runtime_error("Unsupported shader node: " + source.identifier);
		std::string category = definition->getNodeString();
		auto outputs		 = definition->getActiveOutputs();
		MaterialValueType type =
			valueType(outputs.empty() ? definition->getType() : outputs.front()->getType());
		auto arg = [&](const char *name) {
			auto entry = definition->getActiveInput(name);
			return argument(source, name, entry ? valueType(entry->getType()) : type);
		};
		if (category == "image") return image(source, type, false, output);
		if (category == "constant") return arg("value");
		if (category == "texcoord") {
			if (constantInput(source, "index", 0).get<int>() != 0)
				throw std::runtime_error("Only the active UV set is supported");
			return operation(MaterialOp::UV, type, {});
		}
		if (category == "normal" || category == "tangent" || category == "bitangent") {
			if (constantInput(source, "space", "world") != "world")
				throw std::runtime_error("Only world-space shading frames are supported");
			return operation(category == "normal"	 ? MaterialOp::Normal
							 : category == "tangent" ? MaterialOp::Tangent
													 : MaterialOp::Bitangent,
							 type, {});
		}
		if (category == "convert") return convert(arg("in"), type);
		if (category == "combine2")
			return operation(MaterialOp::Combine, type, {arg("in1"), arg("in2")});
		if (category == "combine3")
			return operation(MaterialOp::Combine, type, {arg("in1"), arg("in2"), arg("in3")});
		if (category == "combine4")
			return operation(MaterialOp::Combine, type,
							 {arg("in1"), arg("in2"), arg("in3"), arg("in4")});
		if (category == "separate2" || category == "separate3" || category == "separate4") {
			const std::map<std::string, int> channels{{"outx", 0}, {"outy", 1}, {"outz", 2},
													  {"outw", 3}, {"outr", 0}, {"outg", 1},
													  {"outb", 2}, {"outa", 3}};
			if (!channels.count(output))
				throw std::runtime_error("Unsupported separate output: " + output);
			return operation(MaterialOp::Extract, MaterialValueType::Float, {arg("in")},
							 channels.at(output));
		}
		if (category == "extract")
			return operation(MaterialOp::Extract, type, {arg("in")},
							 constantInput(source, "index", 0).get<int>());
		if (category == "normalize") return operation(MaterialOp::Normalize, type, {arg("in")});
		if (category == "rotate3d")
			return operation(MaterialOp::Rotate3D, type, {arg("in"), arg("amount"), arg("axis")});
		if (category == "normalmap" && source.identifier == "ND_normalmap_float")
			return operation(MaterialOp::NormalMap, type,
							 {arg("in"), arg("scale"),
							  argument(source, "normal", type, nullptr, MaterialOp::Normal),
							  argument(source, "tangent", type, nullptr, MaterialOp::Tangent),
							  argument(source, "bitangent", type, nullptr, MaterialOp::Bitangent)});
		if (category == "mix")
			return operation(MaterialOp::Mix, type, {arg("bg"), arg("fg"), arg("mix")});
		if (category == "clamp")
			return operation(MaterialOp::Clamp, type, {arg("in"), arg("low"), arg("high")});
		if (category == "remap")
			return operation(
				MaterialOp::Remap, type,
				{arg("in"), arg("inlow"), arg("inhigh"), arg("outlow"), arg("outhigh")});
		if (category == "ifgreater" || category == "ifgreatereq" || category == "ifequal")
			return operation(category == "ifgreater"	 ? MaterialOp::IfGreater
							 : category == "ifgreatereq" ? MaterialOp::IfGreaterEqual
														 : MaterialOp::IfEqual,
							 type, {arg("value1"), arg("value2"), arg("in1"), arg("in2")});
		const std::map<std::string, MaterialOp> binary{
			{"add", MaterialOp::Add},			{"subtract", MaterialOp::Subtract},
			{"multiply", MaterialOp::Multiply}, {"divide", MaterialOp::Divide},
			{"min", MaterialOp::Minimum},		{"max", MaterialOp::Maximum},
			{"power", MaterialOp::Power},		{"dotproduct", MaterialOp::Dot},
			{"crossproduct", MaterialOp::Cross}};
		if (binary.count(category))
			return operation(binary.at(category), type, {arg("in1"), arg("in2")});
		throw std::runtime_error("Unsupported shader operation: " + source.identifier);
	}
};
} // namespace

Material::SharedPtr translateMaterial(const MaterialNetwork &network,
									  std::vector<std::string> *diagnostics) {
	try {
		if (!network.diagnostics.empty()) {
			if (diagnostics)
				for (const auto &message : network.diagnostics)
					diagnostics->push_back(network.name + ": " + message);
			for (const auto &message : network.diagnostics)
				Log(Warning, "Material %s: %s", network.name.c_str(), message.c_str());
			return errorMaterial(network.name);
		}
		auto description = Lowering(network).run();
		auto result		 = std::make_shared<Material>();
		result->setName(network.name);
		result->setDescription(std::move(description));
		return result;
	} catch (const std::exception &error) {
		if (diagnostics) diagnostics->push_back(network.name + ": " + error.what());
		Log(Warning, "Material %s is unsupported: %s", network.name.c_str(), error.what());
		return errorMaterial(network.name);
	}
}

} // namespace interop
NAMESPACE_END(krr)
