#include "scene/importer.h"
#include "core/material/description.h"
#include "materials/material.h"
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/base/vt/array.h>
#include <fstream>
#include <iostream>

using namespace krr;
PXR_NAMESPACE_USING_DIRECTIVE
namespace {
void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}
Material::SharedPtr findMaterial(Scene::SharedPtr scene, const std::string &name) {
	for (auto material : scene->getMaterials()) {
		const auto &path = material->getName();
		if (path == name || fs::path(path).filename().string() == name) return material;
	}
	for (auto material : scene->getMaterials())
		if (material->getName().find(name) != std::string::npos) return material;
	throw std::runtime_error("Missing material " + name);
}
CompiledMaterial translated(const interop::MaterialNetwork &network) {
	auto material = interop::translateMaterial(network);
	require(bool(material->getDescription()), "Supported material network became diagnostic");
	return compileMaterial(*material->getDescription());
}
MaterialValues evaluated(const CompiledMaterial &program, float u = .25f) {
	auto values = program.defaults;
	MaterialContext context;
	context.uv = {u, .5f, 0};
	auto sample = [](int, MaterialValue) { return MaterialValue{}; };
	evaluateMaterialProgram({program.surface.data(), uint32_t(program.surface.size())},
		program.uniforms.data(), context, sample, values);
	evaluateMaterialProgram({program.opacity.data(), uint32_t(program.opacity.size())},
		program.uniforms.data(), context, sample, values);
	evaluateMaterialProgram({program.emission.data(), uint32_t(program.emission.size())},
		program.uniforms.data(), context, sample, values);
	return values;
}
void testScatteringNetworks() {
	using P = MaterialParameter;
	interop::MaterialNetwork network;
	network.name = "Scattering translation";
	network.terminal = "surface";
	auto &surface = network.nodes["surface"];
	surface.identifier = "ND_surface";
	surface.inputs["bsdf"].node = "mix";
	auto &mix = network.nodes["mix"];
	mix.identifier = "ND_mix_bsdf";
	mix.inputs["bg"].node = "diffuse";
	mix.inputs["fg"].node = "conductor";
	mix.inputs["mix"].value = .25f;
	auto &diffuse = network.nodes["diffuse"];
	diffuse.identifier = "ND_oren_nayar_diffuse_bsdf";
	diffuse.inputs["color"].value = json::array({.2f, .4f, .6f});
	auto &conductor = network.nodes["conductor"];
	conductor.identifier = "ND_conductor_bsdf";
	conductor.inputs["ior"].value = json::array({1.f, 1.f, 1.f});
	conductor.inputs["extinction"].value = json::array({2.f, 2.f, 2.f});
	conductor.inputs["roughness"].value = json::array({.16f, .16f});
	auto program = translated(network);
	require(program.model == MaterialModel::Composite && program.components.size() == 2,
		"Mix lost its native BSDF components");
	auto first = evaluated(program.components[0]), second = evaluated(program.components[1]);
	require(first[P::Weight][0] == .75f && second[P::Weight][0] == .25f,
		"Mix physical weights were changed");
	require(program.components[0].model == MaterialModel::Diffuse &&
		program.components[1].model == MaterialModel::Conductor,
		"Native scattering identities were lost");
	require(second[P::BaseColor][0] == .5f &&
		std::abs(second[P::SpecularRoughness][0] - .4f) < 1e-6f,
		"Conductor F0 or MaterialX microfacet alpha mapping");

	conductor.inputs["ior"].node = conductor.inputs["extinction"].node = "artistic";
	conductor.inputs["ior"].output = "ior";
	conductor.inputs["extinction"].output = "extinction";
	network.nodes["artistic"].identifier = "ND_artistic_ior";
	network.nodes["artistic"].inputs["reflectivity"].value = json::array({.3f, .5f, .7f});
	program = translated(network);
	require(evaluated(program.components[1])[P::BaseColor][0] == .3f,
		"Artistic conductor reflectivity was lost");

	mix.inputs["mix"].node = "factor";
	network.nodes["factor"].identifier = "ND_extract_vector2";
	network.nodes["factor"].inputs["index"].value = 0;
	network.nodes["factor"].inputs["in"].node = "uv";
	network.nodes["uv"].identifier = "ND_texcoord_vector2";
	program = translated(network);
	require(evaluated(program.components[0], .7f)[P::Weight][0] == 1.f - .7f &&
		evaluated(program.components[1], .7f)[P::Weight][0] == .7f,
		"Connected Mix factor was not retained");
	mix.inputs["mix"].node.clear();
	mix.inputs["mix"].value = 0.f;
	conductor.identifier = "UnsupportedInactiveShader";
	program = translated(network);
	require(program.model == MaterialModel::Diffuse && program.components.empty(),
		"Zero-weight branch was evaluated or single leaf did not collapse");
	mix.inputs["mix"].node = "foldedFactor";
	auto &folded = network.nodes["foldedFactor"];
	folded.identifier = "ND_subtract_float";
	folded.inputs["in1"].value = folded.inputs["in2"].value = 1.f;
	require(translated(network).model == MaterialModel::Diffuse,
		"Constant-folded Mix factor did not prune its unsupported branch");
	std::swap(mix.inputs["bg"], mix.inputs["fg"]);
	folded.inputs["in2"].value = 0.f;
	require(translated(network).model == MaterialModel::Diffuse,
		"Constant-folded Mix endpoint one did not prune its unsupported branch");
	std::swap(mix.inputs["bg"], mix.inputs["fg"]);
	mix.inputs["mix"].node.clear();
	conductor.identifier = "ND_conductor_bsdf";
	mix.inputs["mix"].value = .25f;

	auto &add = network.nodes["add"];
	add.identifier = "ND_add_bsdf";
	add.inputs["in1"].node = add.inputs["in2"].node = "diffuse";
	surface.inputs["bsdf"].node = "add";
	program = translated(network);
	require(program.components.size() == 1 &&
		evaluated(program.components[0])[P::Weight][0] == 2.f,
		"Add was normalized or repeated leaves were not merged");
	add.inputs["in2"].node = "add";
	require(!interop::translateMaterial(network)->getDescription(),
		"A scattering cycle was not diagnosed");
	add.inputs["in2"].node = "diffuse";

	surface.inputs["bsdf"].node = "glass";
	auto &glass = network.nodes["glass"];
	glass.identifier = "ND_dielectric_bsdf";
	glass.inputs["scatter_mode"].value = "RT";
	glass.inputs["ior"].value = 1.f;
	glass.inputs["roughness"].value = json::array({0.f, 0.f});
	program = translated(network);
	require(program.model == MaterialModel::Dielectric &&
		evaluated(program)[P::SpecularIor][0] == 1.f &&
		evaluated(program)[P::SpecularRoughness][0] == 0.f,
		"Index-matched or delta glass parameters changed");

	surface.inputs["bsdf"].node = "mix";
	surface.inputs["edf"].node = "weightedEmission";
	network.nodes["emission"].identifier = "ND_uniform_edf";
	network.nodes["emission"].inputs["color"].value = json::array({2.f, 1.f, 0.f});
	auto &weighted = network.nodes["weightedEmission"];
	weighted.identifier = "ND_multiply_edfF";
	weighted.inputs["in1"].node = "emission";
	weighted.inputs["in2"].value = .25f;
	network.emissionLuminanceScale = 1000.f;
	program = translated(network);
	require(evaluated(program)[P::EmissionColor][0] == .5f &&
		evaluated(program)[P::EmissionLuminance][0] == 1.f,
		"Weighted EDF radiance was changed by OpenPBR emission-unit metadata");
	network.nodes["emission"].identifier = "UnsupportedInactiveEDF";
	weighted.inputs["in2"].value = 0.f;
	require(!translated(network).hasEmission,
		"Zero EDF multiplier did not prune its unsupported input");
	weighted.inputs["in2"].node = "foldedFactor";
	folded.inputs["in2"].value = 1.f;
	require(!translated(network).hasEmission,
		"Constant-folded EDF multiplier did not prune its unsupported input");
	weighted.inputs["in2"].node.clear();
	weighted.inputs["in2"].value = .25f;
	network.nodes["emission"].identifier = "ND_uniform_edf";

	network.opaqueMixBranch = "fg";
	surface.inputs["opacity"].value = 0.f;
	program = translated(network);
	auto values = evaluated(program);
	require(program.model == MaterialModel::Conductor && values[P::Opacity][0] == .25f,
		"White-transparent Mix did not select the opaque branch and coverage");
	require(values[P::EmissionColor][0] == 2.f,
		"Coverage was applied twice to transparent emission");
	network.opaqueMixBranch = "invalid";
	require(!interop::translateMaterial(network)->getDescription(),
		"Invalid producer transparency metadata was accepted");
}
void testBlenderSurfaces(const fs::path &fixtures) {
	using P = MaterialParameter;
	auto scene = std::make_shared<Scene>();
	importer::UsdImporter().import(fixtures / "blender52/surfaces.usda", scene);
	for (const auto &material : scene->getMaterials())
		require(bool(material->getDescription()),
			("Blender surface became diagnostic: " + material->getName()).c_str());
	auto compiled = [&](const char *name) {
		return compileMaterial(*findMaterial(scene, name)->getDescription());
	};
	for (const char *name : {"LeafDiffuse", "LeafRoughDiffuse", "LeafEmission", "LeafAddEmission"})
		require(compiled(name).model == MaterialModel::Diffuse,
			"Blender diffuse or emissive surface did not use a simple native material");
	require(compiled("LeafGlossy").model == MaterialModel::Conductor &&
		compiled("LeafGlass").model == MaterialModel::Dielectric,
		"Blender Glossy/Glass did not retain native scattering models");
	require(compiled("LeafIndexMatchedGlass").defaults[P::SpecularIor][0] == 1.f,
		"Actual Blender index-matched glass lost IOR one");
	for (const char *name : {"LeafMix", "LeafAdd"})
		require(compiled(name).components.size() == 2,
			"Actual Blender Mix/Add lost a scattering component");
	auto mix = compiled("LeafMix");
	require(std::abs(evaluated(mix.components[0])[P::Weight][0] - .65f) < 1e-6f &&
		std::abs(evaluated(mix.components[1])[P::Weight][0] - .35f) < 1e-6f,
		"Actual Blender Mix weights changed");
	auto mixEmission = compiled("LeafMixEmission");
	require(mixEmission.components.size() == 1 &&
		std::abs(evaluated(mixEmission.components[0])[P::Weight][0] - .65f) < 1e-6f &&
		std::abs(evaluated(mixEmission)[P::EmissionColor][0] - .14f) < 1e-6f,
		"Actual Blender scattering/emission mixture was not weighted consistently");
	for (const char *name : {"LeafTransparentDiffuse", "LeafTransparentEmission",
		"LeafTransparentDiffuseEmission"}) {
		auto material = compiled(name);
		require(material.model == MaterialModel::Diffuse &&
			std::abs(evaluated(material)[P::Opacity][0] - .3f) < 1e-6f,
			"Validated Blender transparent surface lost coverage or native scattering");
	}
	for (const char *name : {"LeafTransparentEmission", "LeafTransparentDiffuseEmission"})
		require(std::abs(evaluated(compiled(name))[P::EmissionColor][0] - 1.4f) < 1e-6f,
			"Blender transparent emission retained its opacity multiplier twice");
	auto cubic = compiled("LeafCubicPrincipled");
	require(cubic.model == MaterialModel::OpenPBR && cubic.textures.size() == 1 &&
		cubic.textures[0].filter == MaterialFilter::Cubic,
		"Blender Cubic interpolation was rejected or changed to linear");
}
void testClosureLimits() {
	for (bool emission : {false, true}) {
		for (bool diamond : {false, true}) {
			interop::MaterialNetwork network;
			network.name = "Bounded closure graph";
			network.terminal = "surface";
			network.nodes["surface"].identifier = "ND_surface";
			network.nodes["leaf"].identifier =
				emission ? "ND_uniform_edf" : "ND_oren_nayar_diffuse_bsdf";
			std::string previous = "leaf";
			for (int index = 0; index < (diamond ? 20 : MaterialNodeLimit + 1); ++index) {
				std::string name = "add" + std::to_string(index);
				auto &add = network.nodes[name];
				add.identifier = emission ? "ND_add_edf" : "ND_add_bsdf";
				add.inputs["in1"].node = previous;
				if (diamond) add.inputs["in2"].node = previous;
				previous = name;
			}
			network.nodes["surface"].inputs[emission ? "edf" : "bsdf"].node = previous;
			std::vector<std::string> diagnostics;
			require(!interop::translateMaterial(network, &diagnostics)->getDescription(),
				"Unbounded scattering/emission graph was accepted");
			require(diagnostics.size() == 1 &&
				diagnostics[0].find(diamond ? "expansion limit" : "depth limit") != std::string::npos,
				"Deep or exponentially expanded graph needs an actionable limit diagnostic");
		}
	}
}
} // namespace
int main(int argc, char **argv) {
	try {
		if (argc != 3)
			throw std::runtime_error("Expected fixture directory and artifact directory");
		fs::path fixtures = argv[1], artifacts = argv[2];
		testScatteringNetworks();
		testClosureLimits();
		testBlenderSurfaces(fixtures);
		fs::create_directories(artifacts);
		auto originalDirectory = fs::current_path();
		auto scene			   = std::make_shared<Scene>();
		importer::UsdImporter().import(fixtures / "blender52/materials.usda", scene);
		require(fs::current_path() == originalDirectory, "USD import changed working directory");
		int supported = 0;
		for (const auto &material : scene->getMaterials())
			if (material->getName().find("Supported") != std::string::npos) {
				require(bool(material->getDescription()),
						("Blender graph became diagnostic: " + material->getName()).c_str());
				require(material->getDescription()->model == MaterialModel::OpenPBR,
						"MaterialX surface must take precedence");
				++supported;
			}
		require(supported == 19, "Actual Blender expression fixtures are missing");
		for (const char *name : {"SupportedTextured", "SupportedGrouped"}) {
			auto material = findMaterial(scene, name);
			require(bool(material->getDescription()),
					"Blender supported graph became diagnostic material");
			require(material->getDescription()->model == MaterialModel::OpenPBR,
					"MaterialX surface must take precedence");
			auto program = compileMaterial(*material->getDescription());
			require(!program.surface.empty(), "Expected active Blender expression graph");
		}
		auto textured = findMaterial(scene, "SupportedTextured");
		require(compileMaterial(*textured->getDescription()).kind == MaterialProgramKind::Program,
				"Textured Blender fixture must exercise the expression interpreter");
		require(textured->getDescription()->textures.size() == 3, "Exporter image nodes were lost");
		require(textured->getDescription()->textures[0].texture->hasImage(),
				"Relative USD texture resolution");
		auto geometry = std::make_shared<Scene>();
		importer::UsdImporter().import(fixtures / "scene.usda", geometry);
		auto configured = std::make_shared<Scene>();
		SceneImporter().import(
			json{{"model", json::array({json{{"model", (fixtures / "scene.usda").string()}}})}},
			configured);
		require(configured->getMeshes().size() == geometry->getMeshes().size(),
				"Existing model config did not load USD");
		auto explicitCamera = std::make_shared<Scene>();
		SceneImporter().import(
			json{{"camera", {{"mData", {{"focalLength", 35.f}}}}},
				 {"model", json::array({json{{"model", (fixtures / "scene.usda").string()},
											 {"params", {{"camera_path", "/Root/B"}}}}})}},
			explicitCamera);
		require(explicitCamera->getCamera()->getFocalLength() == 35.f,
				"Explicit config camera lost precedence over USD");
		geometry->getSceneGraph()->update(0);
		require(geometry->getCamera()->getName() == "/Root/A", "Camera stable path ordering");
		require(geometry->getCamera()->getPosition().isApprox(Vector3f(0, 5, 0), 1e-5f),
				"USD camera units and up axis");
		require(std::abs(geometry->getCamera()->getViewMatrix().row(2).head<3>().norm() - 1.f) <
					1e-5f,
				"USD camera-space depth must retain meter units");
		bool located = false;
		for (const auto &instance : geometry->getMeshInstances()) {
			Vector3f translation = instance->getNode()->getGlobalTransform().translation();
			if (translation.isApprox(Vector3f(1, 3, -2), 1e-5f)) located = true;
		}
		require(located, "Default stage start time and instance transform");
		auto selected = std::make_shared<Scene>();
		importer::UsdImporter().import(fixtures / "scene.usda", selected, nullptr,
									   json{{"time_code", 4}, {"camera_path", "/Root/B"}});
		selected->getSceneGraph()->update(0);
		require(selected->getCamera()->getName() == "/Root/B", "Explicit USD camera selection");
		located = false;
		for (const auto &instance : selected->getMeshInstances())
			if (instance->getNode()->getGlobalTransform().translation().isApprox(Vector3f(2, 3, -2),
																				 1e-5f))
				located = true;
		require(located, "Selected-time instance snapshot");
		auto normals = std::make_shared<Scene>();
		importer::UsdImporter().import(fixtures / "indexed_normals.usda", normals);
		require(normals->getMeshes().size() == 1 &&
					normals->getMeshes()[0]->normals[0].z() == -1.f &&
					normals->getMeshes()[0]->normals[3].z() == 1.f,
				"Indexed split normals were lost");
		auto fullStage = UsdStage::Open((fixtures / "scene.usda").string());
		auto fullMesh  = fullStage->GetPrimAtPath(SdfPath("/Root/Prototype/Mesh"));
		fullMesh.GetRelationship(TfToken("material:binding")).ClearTargets(true);
		fullMesh.CreateRelationship(TfToken("material:binding:full"))
			.SetTargets({SdfPath("/Root/Surface")});
		fullStage->Export((artifacts / "full-binding.usdc").string());
		auto fullScene = std::make_shared<Scene>();
		SceneImporter::loadModel(artifacts / "full-binding.usdc", fullScene);
		require(bool(findMaterial(fullScene, "/Root/Surface")->getDescription()),
				"Full-purpose material binding was lost");

		auto emissionLuminance = [](const Material::SharedPtr &material) {
			require(bool(material->getDescription()), "Emitter became diagnostic material");
			auto program = compileMaterial(*material->getDescription());
			auto values = program.defaults;
			MaterialContext context;
			context.uv = {.25f, .5f, 0};
			evaluateMaterialProgram({program.emission.data(), uint32_t(program.emission.size())},
				program.uniforms.data(), context,
				[](int, MaterialValue) { return MaterialValue{}; }, values);
			return values[MaterialParameter::EmissionLuminance][0];
		};
		interop::MaterialNetwork emitter;
		emitter.name = "Emission units";
		emitter.terminal = "surface";
		emitter.nodes["surface"].identifier = "ND_open_pbr_surface_surfaceshader";
		emitter.nodes["surface"].inputs["emission_luminance"].value = 1.f;
		require(emissionLuminance(interop::translateMaterial(emitter)) == 1.f,
			"Ordinary OpenPBR luminance must retain nits");
		emitter.emissionLuminanceScale = 1000.0;
		require(emissionLuminance(interop::translateMaterial(emitter)) == 1000.f,
			"Blender scene-linear emission was not normalized to nits exactly once");
		emitter.nodes["surface"].inputs["emission_luminance"].node = "strength";
		emitter.nodes["strength"].identifier = "ND_extract_vector2";
		emitter.nodes["strength"].inputs["index"].value = 0;
		emitter.nodes["strength"].inputs["in"].node = "uv";
		emitter.nodes["uv"].identifier = "ND_texcoord_vector2";
		require(emissionLuminance(interop::translateMaterial(emitter)) == 250.f,
			"Connected Blender emission expression was not normalized exactly once");
		emitter.emissionLuminanceScale = 1.0;
		require(emissionLuminance(interop::translateMaterial(emitter)) == .25f,
			"Ordinary connected OpenPBR emission changed units");
		for (double scale : {0.0, -1.0, std::numeric_limits<double>::infinity(),
			std::numeric_limits<double>::quiet_NaN()}) {
			emitter.emissionLuminanceScale = scale;
			require(!interop::translateMaterial(emitter)->getDescription(),
				"Invalid emission unit metadata must produce a diagnostic");
		}
		auto rawEmitterScene = std::make_shared<Scene>();
		importer::UsdImporter().import(fixtures / "blender52/emission.usda", rawEmitterScene);
		require(emissionLuminance(findMaterial(rawEmitterScene, "Emitter")) == 1.f,
			"Unmarked USD exporter emission must retain its authored numeric value");
		auto emissionStage = UsdStage::Open((fixtures / "blender52/emission.usda").string());
		emissionStage->GetPrimAtPath(SdfPath("/root/_materials/Emitter"))
			.SetCustomDataByKey(TfToken("kiraray:emissionLuminanceScale"), VtValue(1000.0));
		emissionStage->Export((artifacts / "emission-units.usdc").string());
		auto normalizedEmitterScene = std::make_shared<Scene>();
		importer::UsdImporter().import(artifacts / "emission-units.usdc", normalizedEmitterScene);
		require(emissionLuminance(findMaterial(normalizedEmitterScene, "Emitter")) == 1000.f,
			"Validated Blender USD export lost its emission unit metadata");

		interop::MaterialNetwork preview;
		preview.name						= "Preview defaults";
		preview.terminal					= "surface";
		preview.nodes["surface"].identifier = "UsdPreviewSurface";
		preview.emissionLuminanceScale = 1000.0;
		require(emissionLuminance(interop::translateMaterial(preview)) == 1.f,
			"Blender OpenPBR normalization must not change Preview Surface emission");
		auto previewMaterial				= interop::translateMaterial(preview);
		require(compileMaterial(*previewMaterial->getDescription())
						.defaults[MaterialParameter::CoatRoughness][0] == .01f,
				"Preview coat roughness used another model's default");
		preview.nodes["surface"].inputs["opacity"].value = .4f;
		require(!interop::translateMaterial(preview)->getDescription(),
				"Transparent opacity was silently treated as coverage");
		preview.nodes["surface"].inputs["opacityMode"].value = "presence";
		require(bool(interop::translateMaterial(preview)->getDescription()),
				"Preview presence opacity was rejected");
		preview.nodes["surface"].inputs.clear();
		preview.nodes["surface"].inputs["diffuseColor"].node   = "texture";
		preview.nodes["surface"].inputs["diffuseColor"].output = "rgb";
		auto &image											   = preview.nodes["texture"];
		image.identifier									   = "UsdUVTexture";
		image.inputs["file"].value	   = (fixtures / "blender52/textures/color.png").string();
		image.inputs["fallback"].value = json::array({1, 0, 0, 1});
		auto evaluatePreviewUv		   = [&] {
			auto material = interop::translateMaterial(preview);
			require(bool(material->getDescription()),
							"Supported Preview texture graph was rejected");
			auto program = compileMaterial(*material->getDescription());
			require(program.textures[0].wrapU == MaterialWrap::Border &&
								program.textures[0].wrapV == MaterialWrap::Border,
							"USD texture without wrapping metadata must use black wrapping");
			require(program.textures[0].fallback[0] == 0 && program.textures[0].fallback[3] == 0,
							"USD missing-texture fallback was used as a border color");
			MaterialContext context;
			context.uv			  = {.8f, .9f, 0};
			MaterialValues values = program.defaults;
			auto sample			  = [](int, MaterialValue uv) {
				  return MaterialValue(uv[0], uv[1], .5f, 1.f);
			};
			evaluateMaterialProgram({program.surface.data(), uint32_t(program.surface.size())},
											program.uniforms.data(), context, sample, values);
			return values[MaterialParameter::BaseColor];
		};
		auto uvDefault = evaluatePreviewUv();
		require(uvDefault[0] == 0 && uvDefault[1] == 0,
				"Unauthored UsdUVTexture st must be constant zero");
		image.inputs["st"].node				  = "transform";
		preview.nodes["transform"].identifier = "UsdTransform2d";
		uvDefault							  = evaluatePreviewUv();
		require(uvDefault[0] == 0 && uvDefault[1] == 0,
				"Unauthored UsdTransform2d input must be constant zero");
		image.inputs["st"].node							 = "reader";
		preview.nodes["reader"].identifier				 = "UsdPrimvarReader_float2";
		preview.nodes["reader"].inputs["fallback"].value = json::array({.3f, .7f});
		uvDefault										 = evaluatePreviewUv();
		require(uvDefault[0] == .3f && uvDefault[1] == .7f,
				"Empty USD primvar name must use fallback");

		interop::MaterialNetwork unsupported;
		unsupported.name						= "unsupported";
		unsupported.terminal					= "surface";
		unsupported.nodes["surface"].identifier = "ND_open_pbr_surface_surfaceshader";
		unsupported.nodes["surface"].inputs["subsurface_weight"].value = 1.f;
		require(!interop::translateMaterial(unsupported)->getDescription(),
				"Unsupported lobe must produce diagnostic material");
		unsupported.nodes["surface"].inputs.clear();
		auto &color		 = unsupported.nodes["surface"].inputs["base_color"];
		color.value		 = json::array({.5f, .5f, .5f});
		color.colorSpace = "sRGB";
		auto srgb		 = interop::translateMaterial(unsupported);
		require(srgb->getDescription() && std::abs(compileMaterial(*srgb->getDescription())
													   .defaults[MaterialParameter::BaseColor][0] -
												   .21404114f) < 1e-6f,
				"Authored color-space metadata was ignored");
		color.colorSpace = "ACEScg";
		require(!interop::translateMaterial(unsupported)->getDescription(),
				"Unsupported input color space must be diagnostic");
		unsupported.nodes["surface"].inputs.clear();
		unsupported.nodes["surface"].inputs["coat_color"].error = "Unsupported matrix value";
		require(bool(interop::translateMaterial(unsupported)->getDescription()),
				"Inactive unsupported values must not reject the material");
		unsupported.nodes["surface"].inputs["coat_weight"].value = 1.f;
		require(!interop::translateMaterial(unsupported)->getDescription(),
				"Active unsupported values must not fall back to defaults");
		unsupported.nodes["surface"].inputs.clear();
		unsupported.diagnostics = {"Preflight rejected source ColorRamp"};
		require(interop::translateMaterial(unsupported)->getName().find("unsupported") !=
					std::string::npos,
				"Preflight diagnostic lost");

		fs::path annotated = artifacts / "annotated.usda";
		auto stage		   = UsdStage::Open((fixtures / "blender52/materials.usda").string());
		auto prim		   = stage->GetPrimAtPath(SdfPath("/root/_materials/SupportedGrouped"));
		prim.SetCustomDataByKey(TfToken("kiraray:diagnostics"),
								VtValue(VtArray<std::string>{"Unsupported producer node"}));
		stage->Export(annotated.string());
		auto annotatedScene = std::make_shared<Scene>();
		importer::UsdImporter().import(annotated, annotatedScene);
		require(findMaterial(annotatedScene, "SupportedGrouped")->getName().find("[unsupported]") !=
					std::string::npos,
				"Validated export diagnostics ignored");
		std::cout << "USD conversion and actual Blender graph tests passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
