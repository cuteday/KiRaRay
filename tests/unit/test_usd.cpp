#include "scene/importer.h"
#include "core/material/description.h"
#include "material.h"
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
	for (auto material : scene->getMaterials())
		if (material->getName().find(name) != std::string::npos) return material;
	throw std::runtime_error("Missing material " + name);
}
} // namespace
int main(int argc, char **argv) {
	try {
		if (argc != 3)
			throw std::runtime_error("Expected fixture directory and artifact directory");
		fs::path fixtures = argv[1], artifacts = argv[2];
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
