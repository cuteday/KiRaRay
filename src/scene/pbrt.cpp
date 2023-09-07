#define STBI_MSC_SECURE_CRT
#include <pbrtParser/Scene.h>
#include "stb_image.h"
#include "util/image.h"
#include "render/materials/fresnel.h"
#include "importer.h"

KRR_NAMESPACE_BEGIN

using namespace importer;

namespace {
static MaterialLoader sMaterialLoader;

// helper functions
Matrixf<4, 4> cast(pbrt::affine3f xfm) {
	return Matrixf<4, 4>{{xfm.l.vx.x, xfm.l.vy.x, xfm.l.vz.x, xfm.p.x},
						 {xfm.l.vx.y, xfm.l.vy.y, xfm.l.vz.y, xfm.p.y},
						 {xfm.l.vx.z, xfm.l.vy.z, xfm.l.vz.z, xfm.p.z},
						 {0, 0, 0, 1}};
}
Vector3i cast(const pbrt::vec3i &vec) { return {vec.x, vec.y, vec.z}; }
Vector3f cast(const pbrt::vec3f &vec) { return {vec.x, vec.y, vec.z}; }
Vector2f cast(const pbrt::vec2f &vec) { return {vec.x, vec.y}; }
AABB cast(const pbrt::box3f &aabb) { return {cast(aabb.lower), cast(aabb.upper)}; }

inline Vector3f eta_to_reflectivity(const Vector3f &eta,
									const Vector3f &eta_k) {
	return ((eta - Vector3f(1)).cwiseProduct(eta - Vector3f(1)) +
			eta_k.cwiseProduct(eta_k)) /
		   ((eta + Vector3f(1)).cwiseProduct(eta + Vector3f(1)) +
			eta_k.cwiseProduct(eta_k));
}
} // namespace

void loadTexture(Material::SharedPtr material, pbrt::Texture::SP texture,
				 Material::TextureType type, const fs::path &basedir) {
	if (auto t = std::dynamic_pointer_cast<pbrt::ImageTexture>(texture)) {
		fs::path filename = basedir / t->fileName;
		sMaterialLoader.loadTexture(material, type, filename.string(), true);
	} else {
		Log(Warning, "Encountered unsupported pbrt texture: %s",
			texture->toString().c_str());
	}
}

Material::SharedPtr PbrtImporter::loadMaterial(pbrt::Material::SP mat) {

	static std::map<pbrt::Material::SP, Material::SharedPtr> loadedMaterials;
	if (loadedMaterials.count(mat)) return loadedMaterials[mat];

	Material::SharedPtr material = Material::SharedPtr(new Material(mat->name));
	material->mBsdfType		= MaterialType::Disney;
	material->mShadingModel = Material::ShadingModel::MetallicRoughness;
	Material::MaterialParams &matParams = material->mMaterialParams;

	auto remap_roughness = [](const float &roughness) {
		if (roughness == 0.f) return 0.f;
		float x	  = log(roughness);
		float val = 1.62142f + 0.819955f * x + 0.1734f * pow2(x) +
					0.0171201f * pow3(x) + 0.000640711f * pow4(x);
		return (val);
	};

	if (auto m = std::dynamic_pointer_cast<pbrt::DisneyMaterial>(mat)) {
		Log(Debug, "Encountered disney material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		matParams.diffuse	    = Vector4f(m->color.x, m->color.y, m->color.z, 1);
		matParams.IoR				   = m->eta;
		matParams.specular[2]		   = m->metallic;
		matParams.specular[1]		   = m->roughness;
		matParams.specularTransmission = m->specTrans;
	} else if (auto m = std::dynamic_pointer_cast<pbrt::PlasticMaterial>(mat)) {
		Log(Debug, "Encountered plastic material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		matParams.diffuse		= Vector4f(cast(m->kd), 1);
		if (m->map_kd) {
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				matParams.diffuse = Vector4f(cast(const_tex->value), 1);
			else
				loadTexture(material, m->map_kd, Material::TextureType::Diffuse,
							mBasepath);
		}
		const Vector3f ks(cast(m->ks));
		matParams.specular[3] = luminance(ks);
		float roughness		  = m->roughness;
		if (m->remapRoughness) roughness = remap_roughness(roughness);
		matParams.specular[1] = sqrt(roughness); // sqrt'ed
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MatteMaterial>(mat)) {
		Log(Debug, "Encountered matte material: %s", mat->name.c_str());
		matParams.diffuse = Vector4f(cast(m->kd), 1);
		if (m->map_kd) {
			Log(Debug, "A diffuse texture is found for %s texture %s",
				m->toString().c_str(), m->name.c_str());
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				matParams.diffuse = Vector4f(cast(const_tex->value), 1);
			else
				loadTexture(material, m->map_kd, Material::TextureType::Diffuse, mBasepath);
		}
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.specular[3]	= 0;
	} else if (auto m =
				   std::dynamic_pointer_cast<pbrt::SubstrateMaterial>(mat)) {
		Log(Debug, "Encountered substrate material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.diffuse		= Vector4f(cast(m->kd), 1);
		if (m->map_kd) {
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				matParams.diffuse = Vector4f(cast(const_tex->value), 1);
			else
				loadTexture(material, m->map_kd, Material::TextureType::Diffuse, mBasepath);
		}
		Vector3f ks(cast(m->ks));
		float roughness = (m->uRoughness + m->vRoughness) / 2;
		if (m->remapRoughness) roughness = remap_roughness(roughness);
		matParams.specular = Vector4f(ks, 1 - sqrt(roughness)); // sqrt'ed
	} else if (auto m = std::dynamic_pointer_cast<pbrt::UberMaterial>(mat)) {
		Log(Debug, "Encountered uber material: %s", mat->name.c_str());
		Vector3f diffuse	  = cast(m->kd);
		Vector3f specular	  = cast(m->ks);
		Vector3f transmission = cast(m->kt);
		float roughness		  = m->roughness;
		// roughness				= remap_roughness(roughness);
		if (m->map_kd) {
			Log(Debug, "A diffuse texture is found for %s texture %s",
				m->toString().c_str(), m->name.c_str());
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				diffuse = cast(const_tex->value);
			else
				loadTexture(material, m->map_kd, Material::TextureType::Diffuse, mBasepath);
		}
		if (m->map_ks) {
			Log(Debug, "A specular texture is found for %s texture %s",
				m->toString().c_str(), m->name.c_str());
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_ks))
				specular = cast(const_tex->value);
			else
				loadTexture(material, m->map_ks, Material::TextureType::Specular, mBasepath);
		}
		if (m->map_opacity) {
			Log(Info, "An opacity map is found for %s texture %s",
				m->toString().c_str(), m->name.c_str());
			if (auto tex = std::dynamic_pointer_cast<pbrt::ImageTexture>(
					m->map_opacity))
				loadTexture(material, m->map_opacity, Material::TextureType::Transmission,
							mBasepath);
		} else if (cast(m->opacity) != Vector3f(1)) {
			Color3f transmission = Vector3f(1) - cast(m->opacity);
			material->setConstantTexture(Material::TextureType::Transmission,
										 Color4f(transmission, 1));
		}
		if (m->map_roughness) {
			Log(Info, "A roughness map is found for %s texture %s",
				m->toString().c_str(), m->name.c_str());
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_ks))
				roughness = luminance(cast(const_tex->value));
			else
				Log(Warning,
					"An image roughness map is currently not supported!");
		}
		if (m->map_bump) {
			Log(Info, "A bump map is found for %s texture %s",
				m->toString().c_str(), m->name.c_str());
			if (auto tex =
					std::dynamic_pointer_cast<pbrt::ImageTexture>(m->map_bump))
				loadTexture(material, m->map_bump, Material::TextureType::Normal, mBasepath);
			// TODO: theres no tangent space calculated for pbrt models, so
			// bumpmap is meaningless.
		}
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.diffuse		= Vector4f(diffuse, matParams.diffuse[3]);
		matParams.specular = Vector4f(specular, 1 - sqrt(roughness)); // sqrt'ed
		matParams.specularTransmission = luminance(transmission);
	} else if (auto m =
				   std::dynamic_pointer_cast<pbrt::TranslucentMaterial>(mat)) {
		Log(Debug, "Encountered not well-supported transluscent material: %s",
			mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		matParams.diffuse		= Vector4f(cast(m->kd), matParams.diffuse[3]);
		if (m->map_kd) {
			if (auto const_tex =
					std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				matParams.diffuse =
					Vector4f(cast(const_tex->value), matParams.diffuse[3]);
			else
				loadTexture(material, m->map_kd, Material::TextureType::Diffuse, mBasepath);
		}
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MirrorMaterial>(mat)) {
		// TODO: needs to make disney mirror-reflect when 0-roughness.
		Log(Debug, "Encountered mirror material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.diffuse		= Vector4f(Vector3f(0), 1);
		matParams.specular		= Vector4f(cast(m->kr), 1);
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MetalMaterial>(mat)) {
		Log(Debug, "Encountered not well-supported metal material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		Vector3f eta			= cast(m->eta);
		Vector3f eta_k			= cast(m->k);
		// matParams.diffuse		= Vector4f(eta_to_reflectivity(eta, eta_k), 0);
		matParams.diffuse = Vector4f(bsdf::FrComplex(1, eta, eta_k), 0);
		float roughness	  = m->roughness;
		roughness		  = (m->uRoughness + m->vRoughness) / 2;
		if (m->remapRoughness) roughness = remap_roughness(roughness);
		matParams.specular[1] = sqrt(roughness); // sqrt'ed
		matParams.specular[2] = 0.8;			 // manually set metallic
	} else if (auto m = std::dynamic_pointer_cast<pbrt::GlassMaterial>(mat)) {
		Log(Debug, "Encountered not well-supported glass material: %s",
			mat->name.c_str());
		Log(Debug, "Glass material %s has an index of %f", mat->name.c_str(),
			m->index);
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		material->mBsdfType		= MaterialType::Dielectric;
		matParams.specularTransmission = luminance(cast(m->kt));
		matParams.diffuse  = Vector4f(cast(m->kt), 1);
		matParams.specular = Vector4f{0, 0, 0, 0};
		matParams.IoR	   = m->index == 1 ? 1.01 : m->index;
	} else {
		Log(Warning, "Encountered unsupported %s material: %s",
			mat->toString().c_str(), mat->name.c_str());
		return nullptr; // falling back to the default material 0...
	}
	if (matParams.IoR == 1) // 1-ETA is not plausible for transmission
		matParams.IoR = 1.001;

	mScene->addMaterial(material);
	loadedMaterials[mat] = material;
	return material;
}

void createAreaLight(Mesh::SharedPtr mesh, pbrt::AreaLight::SP areaLight) {
	if (auto l =
			std::dynamic_pointer_cast<pbrt::DiffuseAreaLightRGB>(areaLight)) {
		mesh->Le = cast(l->L);
	} else if (auto l = std::dynamic_pointer_cast<pbrt::DiffuseAreaLightBB>(
				   areaLight)) {
		// convert blackbody strength & temporature to linear RGB...
		mesh->Le = cast(l->LinRGB());
	} else {
		Log(Warning, "Encountered unsupported area light: %s",
			areaLight->toString().c_str());
	}
}

Mesh::SharedPtr PbrtImporter::loadMesh(pbrt::TriangleMesh::SP pbrtMesh) {
	static std::map<pbrt::TriangleMesh::SP, Mesh::SharedPtr> loadedMeshes;
	if (loadedMeshes.count(pbrtMesh)) return loadedMeshes[pbrtMesh];

	Mesh::SharedPtr mesh = std::make_shared<Mesh>();
	int n_vertices		 = pbrtMesh->vertex.size();
	int n_faces			 = pbrtMesh->getNumPrims();
	Log(Debug, "The current mesh %s has %d vertices and %d faces", pbrtMesh->toString().c_str(),
		n_vertices, n_faces);
	if (pbrtMesh->normal.size() < n_vertices)
		Log(Debug,
			"The current mesh has %zd normals but %d vertices, thus the "
			"normal(s) are ignored.",
			pbrtMesh->normal.size(), n_vertices);
	mesh->indices.reserve(n_vertices);
	for (int i = 0; i < n_vertices; i++) {
		if (pbrtMesh->normal.size()) mesh->normals.push_back(cast(pbrtMesh->normal[i]));
		if (pbrtMesh->texcoord.size()) mesh->texcoords.push_back(cast(pbrtMesh->texcoord[i]));
		mesh->positions.push_back(cast(pbrtMesh->vertex[i]));
	}
	for (int i = 0; i < n_faces; i++) mesh->indices.push_back(cast(pbrtMesh->index[i]));
	mesh->aabb = cast(pbrtMesh->getBounds());
	if (pbrtMesh->material) mesh->material = loadMaterial(pbrtMesh->material);
	else mesh->material = mScene->getMaterials()[0];
	if (pbrtMesh->areaLight) createAreaLight(mesh, pbrtMesh->areaLight);
	if (pbrtMesh->mediumInterface) {
		mesh->inside = loadMedium(pbrtMesh->mediumInterface->inside);
		mesh->outside = loadMedium(pbrtMesh->mediumInterface->outside);
	}

	mesh->setName(pbrtMesh->toString());
	loadedMeshes[pbrtMesh] = mesh;
	mScene->addMesh(mesh);
	return mesh;
}

Volume::SharedPtr PbrtImporter::loadMedium(pbrt::Medium::SP pbrtMedium) {
	static std::map<pbrt::Medium::SP, Volume::SharedPtr> loadedMedia;
	static SceneGraphNode::SharedPtr mediaContainer = nullptr;

	if (!pbrtMedium) return nullptr;

	auto sceneGraph = mScene->getSceneGraph();
	if (!mediaContainer) {
		mediaContainer = std::make_shared<SceneGraphNode>();
		mediaContainer->setName("Media Container");
		sceneGraph->attach(sceneGraph->getRoot(), mediaContainer);
	}
	if (loadedMedia.count(pbrtMedium)) return loadedMedia[pbrtMedium];

	Volume::SharedPtr result = nullptr;
	if (auto m = std::dynamic_pointer_cast<pbrt::HomogeneousMedium>(pbrtMedium)) {
		auto medium = std::make_shared<HomogeneousVolume>();
		medium->sigma_a = cast(m->sigmaScale * m->sigma_a);
		medium->sigma_s = cast(m->sigmaScale * m->sigma_s);
		medium->Le		= cast(m->LeScale * m->Le);
		medium->g		= m->g;
		result = medium;
	} else {
		Log(Warning, "Encountered unsupported medium: %s", pbrtMedium->toString().c_str());
		return nullptr;
	}
	
	auto mediaNode = std::make_shared<SceneGraphNode>();
	mediaNode->setLeaf(result);
	mediaNode->setName(pbrtMedium->name);
	sceneGraph->attach(mediaContainer, mediaNode);
	loadedMedia[pbrtMedium] = result;
	return result;
}

bool PbrtImporter::import(const fs::path filepath, Scene::SharedPtr pScene) {
	string basepath = fs::path(filepath).parent_path().string();
	mBasepath		= basepath;
	mScene			= pScene;

	Log(Info, "Attempting to load pbrt scene from %s...", filepath.c_str());
	pbrt::Scene::SP scene = pbrt::importPBRT(filepath.string());
	if (!scene) {
		Log(Fatal, "Failed to load pbrt file from %s...", filepath.c_str());
		return false;
	}
	scene->makeSingleLevel(); // since currently kiraray supports only single gas.

	auto root = std::make_shared<SceneGraphNode>();
	mScene->getSceneGraph()->setRoot(root);

	pScene->addMaterial(std::make_shared<Material>("default material")); 
	// the default material for shapes without material

	// build scenegraph
	for (const pbrt::Instance::SP inst : scene->world->instances) {
		Affine3f transform = cast(inst->xfm); // the instance's local transform
		auto node		   = std::make_shared<SceneGraphNode>();
		mScene->getSceneGraph()->attach(root, node);
		node->setScaling(transform.scaling());
		node->setRotation(Quaternionf(transform.rotation()));
		node->setTranslation(transform.translation());
		for (const pbrt::Shape::SP geom : inst->object->shapes) {
			if (auto m = std::dynamic_pointer_cast<pbrt::TriangleMesh>(geom)) {
				Mesh::SharedPtr mesh = loadMesh(m);
				auto meshInstance	 = std::make_shared<MeshInstance>(mesh);
				mScene->getSceneGraph()->attachLeaf(node, meshInstance);
			} else {
				Log(Debug, "Encountered unsupported pbrt shape type: %s",
					geom->toString().c_str());
			}
		}
	}

	for (const pbrt::LightSource::SP light : scene->world->lightSources) {
		if (auto l = std::dynamic_pointer_cast<pbrt::InfiniteLightSource>(light)) 
			Log(Debug, "Encountered infinite light %s", l->mapName.c_str());
	}

	Log(Info, "PBRT scene loaded %zd meshes", mScene->getMeshes().size());
	return true;
}

string PbrtImporter::resolve(string path) {
	return (fs::path(mBasepath) / path).string();
}

KRR_NAMESPACE_END