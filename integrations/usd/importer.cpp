#include "scene/importer.h"
#include "scene/interop.h"
#include "material.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/utils.h>
#include <pxr/usd/usdLux/blackbody.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/ar/resolverContextBinder.h>
#include <pxr/base/gf/camera.h>
#include <pxr/base/gf/frustum.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/plug/registry.h>
#include <set>

PXR_NAMESPACE_USING_DIRECTIVE
NAMESPACE_BEGIN(krr)
namespace importer {
namespace {

template <typename T> T attr(const UsdPrim &prim, const char *name, UsdTimeCode time, T fallback) {
	T result = fallback;
	prim.GetAttribute(TfToken(name)).Get(&result, time);
	return result;
}

Matrix4f matrix(const GfMatrix4d &source) {
	Matrix4f result;
	for (int row = 0; row < 4; ++row)
		for (int column = 0; column < 4; ++column) result(row, column) = float(source[column][row]);
	return result;
}

template <typename T> json vectorValue(const T &value, int size) {
	json result = json::array();
	for (int i = 0; i < size; ++i) result.push_back(value[i]);
	return result;
}

json value(const VtValue &source) {
	if (source.IsEmpty()) return nullptr;
	if (source.IsHolding<bool>()) return source.UncheckedGet<bool>();
	if (source.IsHolding<int>()) return source.UncheckedGet<int>();
	if (source.IsHolding<float>()) return source.UncheckedGet<float>();
	if (source.IsHolding<double>()) return source.UncheckedGet<double>();
	if (source.IsHolding<std::string>()) return source.UncheckedGet<std::string>();
	if (source.IsHolding<TfToken>()) return source.UncheckedGet<TfToken>().GetString();
	if (source.IsHolding<SdfAssetPath>()) {
		const auto &asset = source.UncheckedGet<SdfAssetPath>();
		return asset.GetResolvedPath().empty()
				   ? ArGetResolver().Resolve(asset.GetAssetPath()).GetPathString()
				   : asset.GetResolvedPath();
	}
	if (source.IsHolding<GfVec2f>()) return vectorValue(source.UncheckedGet<GfVec2f>(), 2);
	if (source.IsHolding<GfVec3f>()) return vectorValue(source.UncheckedGet<GfVec3f>(), 3);
	if (source.IsHolding<GfVec4f>()) return vectorValue(source.UncheckedGet<GfVec4f>(), 4);
	throw std::runtime_error("Unsupported USD material value: " + source.GetTypeName());
}

interop::Interpolation interpolation(TfToken token) {
	if (token == UsdGeomTokens->constant) return interop::Interpolation::Constant;
	if (token == UsdGeomTokens->uniform) return interop::Interpolation::Uniform;
	if (token == UsdGeomTokens->vertex || token == UsdGeomTokens->varying)
		return interop::Interpolation::Vertex;
	if (token == UsdGeomTokens->faceVarying) return interop::Interpolation::FaceVarying;
	throw std::runtime_error("Unsupported primvar interpolation: " + token.GetString());
}

class StageImport {
public:
	StageImport(UsdStageRefPtr stage, Scene::SharedPtr scene, SceneGraphNode::SharedPtr parent,
				UsdTimeCode time) :
		stage(stage), scene(scene), parent(parent), time(time), transforms(time) {
		float scale = float(UsdGeomGetStageMetersPerUnit(stage));
		if (!(scale > 0) || !std::isfinite(scale))
			throw std::runtime_error("USD metersPerUnit must be positive and finite");
		conversion = Affine3f(Eigen::Scaling(scale));
		if (UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z)
			conversion = Eigen::AngleAxisf(-M_PI * .5f, Vector3f::UnitX()) * conversion;
	}
	void run(const json &params) {
		auto layer = stage->GetRootLayer();
		if (layer->GetDocumentation().find("Blender v5.2") == 0 &&
			!layer->GetCustomLayerData().count("kiraray:producer"))
			Log(Warning, "Blender USD has no KiRaRay producer metadata; use KiRaRay Validated USD "
						 "export to preserve emission units and unsupported-material diagnostics. "
						 "Unannotated OpenPBR values are interpreted as standard nits.");
		std::vector<UsdPrim> cameras;
		std::set<SdfPath> pointPrototypes;
		for (auto prim : stage->Traverse(UsdTraverseInstanceProxies())) {
			if (UsdGeomPointInstancer instancer{prim}) {
				SdfPathVector paths;
				instancer.GetPrototypesRel().GetTargets(&paths);
				pointPrototypes.insert(paths.begin(), paths.end());
			}
		}
		for (auto prim : stage->Traverse(UsdTraverseInstanceProxies())) {
			bool prototype = false;
			for (const auto &path : pointPrototypes)
				if (prim.GetPath().HasPrefix(path)) {
					prototype = true;
					break;
				}
			if (prototype || !visible(prim)) continue;
			if (UsdGeomMesh mesh{prim})
				addMesh(mesh,
						conversion * Affine3f(matrix(transforms.GetLocalToWorldTransform(prim))));
			else if (UsdGeomPointInstancer instancer{prim})
				addInstances(instancer);
			else if (prim.IsA<UsdGeomCamera>())
				cameras.push_back(prim);
			else if (prim.HasAPI<UsdLuxLightAPI>())
				addLight(prim);
			else if (prim.GetTypeName() == TfToken("BasisCurves") ||
					 prim.GetTypeName() == TfToken("Volume"))
				Log(Warning, "USD %s: curves and volumes are not supported",
					prim.GetPath().GetText());
		}
		bool explicitCamera = scene->getConfig().contains("camera");
		if (!explicitCamera) {
			std::sort(cameras.begin(), cameras.end(),
					  [](const UsdPrim &a, const UsdPrim &b) { return a.GetPath() < b.GetPath(); });
			std::string cameraPath = params.value("camera_path", std::string());
			if (!cameraPath.empty()) {
				auto selected = stage->GetPrimAtPath(SdfPath(cameraPath));
				if (!selected || !selected.IsA<UsdGeomCamera>())
					throw std::runtime_error("USD camera_path does not identify a camera: " +
											 cameraPath);
				cameras = {selected};
			}
			for (auto prim : cameras) {
				UsdGeomCamera usdCamera(prim);
				if (attr(prim, "projection", time, UsdGeomTokens->perspective) !=
					UsdGeomTokens->perspective) {
					Log(Warning, "USD %s: only perspective cameras are supported",
						prim.GetPath().GetText());
					continue;
				}
				auto camera			= std::make_shared<Camera>();
				GfCamera data		= usdCamera.GetCamera(time);
				float metersPerUnit = float(UsdGeomGetStageMetersPerUnit(stage));
				auto clipping		= data.GetClippingRange();
				data.SetClippingRange(GfRange1f(clipping.GetMin() * metersPerUnit,
												clipping.GetMax() * metersPerUnit));
				camera->setProjection(matrix(data.GetFrustum().ComputeProjectionMatrix()), -1.f,
									  false);
				camera->setFilmSize({data.GetHorizontalAperture(), data.GetVerticalAperture()});
				camera->setAspectRatio(data.GetAspectRatio());
				camera->setFocalLength(data.GetFocalLength());
				auto node =
					scene->getSceneGraph()->attachLeaf(parent, camera, prim.GetPath().GetString());
				Affine3f cameraTransform =
					conversion * Affine3f(matrix(transforms.GetLocalToWorldTransform(prim)));
				cameraTransform.linear() /= metersPerUnit;
				node->setLocalTransform(cameraTransform);
				scene->setCamera(camera);
				scene->setCameraController(nullptr);
				if (attr(prim, "exposure", time, 0.f) != 0.f || data.GetFStop() != 0.f)
					Log(Warning, "USD %s: camera exposure and depth of field are not supported",
						prim.GetPath().GetText());
				break;
			}
		}
		if (!scene->getCamera()->getNode())
			scene->getSceneGraph()->attachLeaf(parent, scene->getCamera(), "Default Camera");
	}

private:
	UsdStageRefPtr stage;
	Scene::SharedPtr scene;
	SceneGraphNode::SharedPtr parent;
	UsdTimeCode time;
	UsdGeomXformCache transforms;
	Affine3f conversion;
	std::map<SdfPath, Material::SharedPtr> materials;
	std::map<std::string, std::vector<Mesh::SharedPtr>> meshes;

	bool visible(const UsdPrim &prim) {
		UsdGeomImageable imageable(prim);
		if (!imageable) return true;
		if (imageable.ComputeVisibility(time) == UsdGeomTokens->invisible) return false;
		auto purpose = imageable.ComputePurpose();
		return purpose != UsdGeomTokens->guide && purpose != UsdGeomTokens->proxy;
	}
	void shader(const UsdShadeShader &source, interop::MaterialNetwork &network) {
		std::string path = source.GetPath().GetString();
		if (network.nodes.count(path)) return;
		auto &node = network.nodes[path];
		TfToken identifier;
		source.GetIdAttr().Get(&identifier, time);
		node.identifier = identifier.GetString();
		for (auto input : source.GetInputs()) {
			auto &target = node.inputs[input.GetBaseName().GetString()];
			try {
				target.type		  = input.GetTypeName().GetAsToken().GetString();
				target.colorSpace = input.GetAttr().GetColorSpace().GetString();
				auto producers	  = UsdShadeUtils::GetValueProducingAttributes(input);
				if (producers.size() > 1)
					throw std::runtime_error("Multiple shader connections at " +
											 input.GetFullName().GetString());
				UsdAttribute attribute = producers.empty() ? input.GetAttr() : producers.front();
				if (attribute.GetName().GetString().find("outputs:") == 0 &&
					attribute.GetPrim().IsA<UsdShadeShader>()) {
					UsdShadeShader upstream(attribute.GetPrim());
					target.node	  = upstream.GetPath().GetString();
					target.output = UsdShadeOutput(attribute).GetBaseName().GetString();
					shader(upstream, network);
				} else {
					VtValue data;
					attribute.Get(&data, time);
					target.value = value(data);
					if (target.colorSpace.empty())
						target.colorSpace = attribute.GetColorSpace().GetString();
				}
			} catch (const std::exception &error) {
				target.error = error.what();
			}
		}
	}
	Material::SharedPtr material(const UsdShadeMaterial &source) {
		if (!source) {
			static const SdfPath fallback("/__krr_unbound");
			if (!materials.count(fallback)) {
				auto result = std::make_shared<Material>();
				result->setName("USD default");
				result->mMaterialParams.diffuse = RGBA{.18f, .18f, .18f, 1.f};
				materials[fallback]				= result;
			}
			return materials[fallback];
		}
		if (materials.count(source.GetPath())) return materials.at(source.GetPath());
		interop::MaterialNetwork network;
		network.name = source.GetPath().GetString();
		try {
			auto emissionScale =
				source.GetPrim().GetCustomDataByKey(TfToken("kiraray:emissionLuminanceScale"));
			if (!emissionScale.IsEmpty())
				network.emissionLuminanceScale = value(emissionScale).get<double>();
			VtValue diagnostic =
				source.GetPrim().GetCustomDataByKey(TfToken("kiraray:diagnostics"));
			if (diagnostic.IsHolding<VtArray<std::string>>())
				for (const auto &message : diagnostic.UncheckedGet<VtArray<std::string>>())
					network.diagnostics.push_back(message);
			for (UsdPrim prim = source.GetPrim(); prim; prim = prim.GetParent()) {
				auto colorSpace = attr(prim, "colorSpace:name", time, TfToken());
				if (colorSpace.IsEmpty()) continue;
				if (colorSpace != TfToken("lin_rec709_scene") &&
					colorSpace != TfToken("lin_rec709"))
					network.diagnostics.push_back("Unsupported working color space: " +
												  colorSpace.GetString());
				break;
			}
			UsdShadeOutput surface = source.GetSurfaceOutput(TfToken("mtlx"));
			if (!surface || !surface.HasConnectedSource()) surface = source.GetSurfaceOutput();
			auto producers = UsdShadeUtils::GetValueProducingAttributes(surface, true);
			if (producers.size() != 1 || !producers.front().GetPrim().IsA<UsdShadeShader>())
				throw std::runtime_error("Expected one surface shader connection");
			UsdShadeShader terminal(producers.front().GetPrim());
			network.terminal = terminal.GetPath().GetString();
			shader(terminal, network);
			for (auto context : {TfToken(), TfToken("mtlx")}) {
				auto displacement = source.GetDisplacementOutput(context);
				if (displacement && displacement.HasConnectedSource())
					network.diagnostics.push_back("Displacement is not supported");
				auto volume = source.GetVolumeOutput(context);
				if (volume && volume.HasConnectedSource())
					network.diagnostics.push_back("Volume materials are not supported");
			}
		} catch (const std::exception &error) {
			network.diagnostics.push_back(error.what());
		}
		return materials[source.GetPath()] = interop::translateMaterial(network);
	}
	std::vector<Mesh::SharedPtr> mesh(const UsdGeomMesh &source) {
		interop::MeshInput input;
		input.name = source.GetPath().GetString();
		UsdShadeMaterial binding =
			UsdShadeMaterialBindingAPI(source).ComputeBoundMaterial(UsdShadeTokens->full);
		input.materials.push_back(material(binding));
		std::string key =
			(source.GetPrim().IsInstanceProxy() ? source.GetPrim().GetPrimInPrototype().GetPath()
												: source.GetPath())
				.GetString();
		key += "|" + binding.GetPath().GetString();
		VtIntArray counts, indices, holes;
		VtVec3fArray points, normals;
		source.GetFaceVertexCountsAttr().Get(&counts, time);
		source.GetFaceVertexIndicesAttr().Get(&indices, time);
		source.GetHoleIndicesAttr().Get(&holes, time);
		source.GetPointsAttr().Get(&points, time);
		input.faceCounts.assign(counts.begin(), counts.end());
		input.faceIndices.assign(indices.begin(), indices.end());
		input.holeFaces.assign(holes.begin(), holes.end());
		for (const auto &p : points) input.points.emplace_back(p[0], p[1], p[2]);
		input.leftHanded = attr(source.GetPrim(), "orientation", time,
								UsdGeomTokens->rightHanded) == UsdGeomTokens->leftHanded;
		source.GetNormalsAttr().Get(&normals, time);
		input.normals.interpolation = interpolation(source.GetNormalsInterpolation());
		for (const auto &n : normals) input.normals.values.emplace_back(n[0], n[1], n[2]);
		UsdGeomPrimvarsAPI primvars(source);
		if (auto primvar = primvars.FindPrimvarWithInheritance(TfToken("normals"))) {
			VtVec3fArray values;
			VtIntArray normalIndices;
			if (!primvar.Get(&values, time))
				throw std::runtime_error(input.name + ": normals must have three components");
			primvar.GetIndices(&normalIndices, time);
			input.normals.values.clear();
			for (const auto &normal : values)
				input.normals.values.emplace_back(normal[0], normal[1], normal[2]);
			input.normals.indices.assign(normalIndices.begin(), normalIndices.end());
			input.normals.interpolation = interpolation(primvar.GetInterpolation());
		}
		auto uv = primvars.FindPrimvarWithInheritance(TfToken("st"));
		if (!uv) uv = primvars.FindPrimvarWithInheritance(TfToken("UVMap"));
		if (uv) {
			VtVec2fArray values;
			VtIntArray indices;
			if (!uv.Get(&values, time))
				throw std::runtime_error(input.name + ": active UV set must have two components");
			uv.GetIndices(&indices, time);
			input.texcoords.indices.assign(indices.begin(), indices.end());
			input.texcoords.interpolation = interpolation(uv.GetInterpolation());
			for (const auto &v : values) input.texcoords.values.emplace_back(v[0], v[1]);
		}
		input.faceMaterials.resize(counts.size(), 0);
		std::set<int> assigned;
		for (const auto &subset : UsdShadeMaterialBindingAPI(source).GetMaterialBindSubsets()) {
			auto subsetBinding =
				UsdShadeMaterialBindingAPI(subset).ComputeBoundMaterial(UsdShadeTokens->full);
			key += "|" + subsetBinding.GetPath().GetString();
			int slot = int(input.materials.size());
			input.materials.push_back(material(subsetBinding));
			VtIntArray faces;
			subset.GetIndicesAttr().Get(&faces, time);
			for (int face : faces) {
				if (face < 0 || size_t(face) >= counts.size() || !assigned.insert(face).second)
					throw std::runtime_error(input.name +
											 ": invalid or overlapping material subsets");
				input.faceMaterials[face] = slot;
			}
		}
		if (meshes.count(key)) return meshes.at(key);
		if (attr(source.GetPrim(), "subdivisionScheme", time, UsdGeomTokens->catmullClark) !=
			UsdGeomTokens->none)
			Log(Warning, "USD %s: subdivision is deferred; using authored polygons",
				source.GetPath().GetText());
		return meshes[key] = interop::makeMeshes(input);
	}
	void addMesh(const UsdGeomMesh &source, const Affine3f &transform) {
		interop::attachMeshes(scene, mesh(source), transform, source.GetPath().GetString(), parent);
	}
	void addInstances(const UsdGeomPointInstancer &instancer) {
		SdfPathVector prototypes;
		instancer.GetPrototypesRel().GetTargets(&prototypes);
		VtIntArray indices;
		instancer.GetProtoIndicesAttr().Get(&indices, time);
		VtMatrix4dArray instanceTransforms;
		if (!instancer.ComputeInstanceTransformsAtTime(&instanceTransforms, time, time,
													   UsdGeomPointInstancer::IncludeProtoXform,
													   UsdGeomPointInstancer::IgnoreMask))
			throw std::runtime_error("Cannot evaluate USD point instances: " +
									 instancer.GetPath().GetString());
		auto mask = instancer.ComputeMaskAtTime(time);
		if (indices.size() != instanceTransforms.size())
			throw std::runtime_error("Point instance transform count mismatch");
		if (!mask.empty() && mask.size() != indices.size())
			throw std::runtime_error("Point instance visibility mask count mismatch");
		Affine3f world =
			conversion * Affine3f(matrix(transforms.GetLocalToWorldTransform(instancer.GetPrim())));
		for (size_t i = 0; i < indices.size(); ++i) {
			if (!mask.empty() && !mask[i]) continue;
			if (indices[i] < 0 || size_t(indices[i]) >= prototypes.size())
				throw std::runtime_error("Invalid point instance prototype index");
			auto root		   = stage->GetPrimAtPath(prototypes[indices[i]]);
			GfMatrix4d inverse = transforms.GetLocalToWorldTransform(root).GetInverse();
			for (const auto &prim : UsdPrimRange(root, UsdTraverseInstanceProxies())) {
				if (!prim.IsA<UsdGeomMesh>() || !visible(prim)) continue;
				GfMatrix4d relative = transforms.GetLocalToWorldTransform(prim) * inverse;
				addMesh(UsdGeomMesh(prim), world * Affine3f(matrix(instanceTransforms[i])) *
											   Affine3f(matrix(relative)));
			}
		}
	}
	void addLight(const UsdPrim &prim) {
		interop::LightInput light;
		light.name		= prim.GetPath().GetString();
		light.transform = conversion * Affine3f(matrix(transforms.GetLocalToWorldTransform(prim)));
		auto color		= attr(prim, "inputs:color", time, GfVec3f(1));
		if (attr(prim, "inputs:enableColorTemperature", time, false))
			color = GfCompMult(color, UsdLuxBlackbodyTemperatureAsRgb(
										  attr(prim, "inputs:colorTemperature", time, 6500.f)));
		light.color		 = {color[0], color[1], color[2]};
		light.intensity	 = attr(prim, "inputs:intensity", time, 1.f);
		light.exposure	 = attr(prim, "inputs:exposure", time, 0.f);
		light.normalize	 = attr(prim, "inputs:normalize", time, false);
		std::string type = prim.GetTypeName().GetString();
		if (type == "RectLight") {
			light.type	 = interop::LightInput::Type::Rectangle;
			light.width	 = attr(prim, "inputs:width", time, 1.f);
			light.height = attr(prim, "inputs:height", time, 1.f);
		} else if (type == "DiskLight") {
			light.type	 = interop::LightInput::Type::Disk;
			light.radius = attr(prim, "inputs:radius", time, .5f);
		} else if (type == "SphereLight") {
			light.radius = attr(prim, "inputs:radius", time, .5f);
			bool point	 = attr(prim, "treatAsPoint", time, false) || light.radius == 0.f;
			light.type =
				point ? interop::LightInput::Type::Point : interop::LightInput::Type::Sphere;
			if (prim.GetAttribute(TfToken("inputs:shaping:cone:angle")).HasAuthoredValueOpinion()) {
				light.type		   = interop::LightInput::Type::Spot;
				light.coneAngle	   = attr(prim, "inputs:shaping:cone:angle", time, 90.f);
				light.coneSoftness = attr(prim, "inputs:shaping:cone:softness", time, 0.f);
				if (!point)
					Log(Warning, "USD %s: finite-radius spotlights use a point source",
						prim.GetPath().GetText());
			}
		} else if (type == "DistantLight") {
			light.type		   = interop::LightInput::Type::Distant;
			light.distantAngle = attr(prim, "inputs:angle", time, 0.f);
		} else if (type == "DomeLight") {
			light.type	 = interop::LightInput::Type::Dome;
			auto texture = attr(prim, "inputs:texture:file", time, SdfAssetPath());
			if (!texture.GetAssetPath().empty())
				light.texture = value(VtValue(texture)).get<std::string>();
			auto format = attr(prim, "inputs:texture:format", time, TfToken("latlong"));
			if (format != TfToken("latlong") && format != TfToken("automatic"))
				throw std::runtime_error(
					"Only latitude-longitude environment textures are supported");
		} else {
			Log(Warning, "USD %s: unsupported light type %s", prim.GetPath().GetText(),
				type.c_str());
			return;
		}
		interop::attachLight(scene, light, parent);
	}
};
} // namespace

bool UsdImporter::import(const fs::path filepath, Scene::SharedPtr scene,
						 SceneGraphNode::SharedPtr node, const json &params) {
	const json options = params.is_null() ? json::object() : params;
	if (!options.is_object()) throw std::invalid_argument("USD import params must be an object");
#if !KRR_ENABLE_HYDRA
	static const auto plugins =
		PlugRegistry::GetInstance().RegisterPlugins(std::string(KRR_USD_ROOT) + "/lib/usd");
#endif
	auto stage = UsdStage::Open(filepath.string());
	if (!stage) throw std::runtime_error("Cannot open USD stage: " + filepath.string());
	ArResolverContextBinder binder(stage->GetPathResolverContext());
	double time = options.value("time_code", stage->GetStartTimeCode());
	if (!std::isfinite(time)) throw std::invalid_argument("USD time_code must be finite");
	auto graph = scene->getSceneGraph();
	if (!graph->getRoot()) graph->setRoot(std::make_shared<SceneGraphNode>("Root"));
	auto container = std::make_shared<SceneGraphNode>(filepath.filename().string());
	graph->attach(node ? node : graph->getRoot(), container);
	StageImport(stage, scene, container, UsdTimeCode(time)).run(options);
	return true;
}

} // namespace importer
NAMESPACE_END(krr)
