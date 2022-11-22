#define STBI_MSC_SECURE_CRT
#include <pbrtParser/Scene.h>
#include "stb_image.h"
#include "util/image.h"
#include "render/materials/fresnel.h"
#include "importer.h"

KRR_NAMESPACE_BEGIN

using namespace math;
using namespace importer;

namespace {
static uint textureIdAllocator	= 0;
static uint materialIdAllocator = 0;
static MaterialLoader sMaterialLoader;

// helper functions
Matrixf<4, 4> cast(pbrt::affine3f xfm) {
	return Matrixf<4, 4>{ { xfm.l.vx.x, xfm.l.vy.x, xfm.l.vz.x, xfm.p.x },
						  { xfm.l.vx.y, xfm.l.vy.y, xfm.l.vz.y, xfm.p.y },
						  { xfm.l.vx.z, xfm.l.vy.z, xfm.l.vz.z, xfm.p.z },
						  { 0, 0, 0, 1 } };
}
Vector3i cast(const pbrt::vec3i &vec) { return { vec.x, vec.y, vec.z }; }
Vector3f cast(const pbrt::vec3f &vec) { return { vec.x, vec.y, vec.z }; }
Vector2f cast(const pbrt::vec2f &vec) { return { vec.x, vec.y }; }
AABB cast(const pbrt::box3f& aabb) { return { cast(aabb.lower), cast(aabb.upper) }; }

inline Vector3f eta_to_reflectivity(const Vector3f& eta, const Vector3f& eta_k) {
	return ((eta - Vector3f(1)).cwiseProduct(eta - Vector3f(1)) + eta_k.cwiseProduct(eta_k)) /
		   ((eta + Vector3f(1)).cwiseProduct(eta + Vector3f(1)) + eta_k.cwiseProduct(eta_k));
}
} 

void loadTexture(Material::SharedPtr material,
	pbrt::Texture::SP texture, 
	Material::TextureType type,
	const fs::path &basedir) {
	if (auto t = std::dynamic_pointer_cast<pbrt::ImageTexture>(texture)) {
		fs::path filename = basedir / t->fileName;
		stbi_set_flip_vertically_on_load(true);	// pbrt textures do not need filp.
		sMaterialLoader.loadTexture(material, type, filename.string());
		stbi_set_flip_vertically_on_load(false);
	} else {
		Log(Warning, "Encountered unsupported pbrt texture: %s", texture->toString().c_str());
	}
}

size_t loadMaterial(Scene::SharedPtr scene,
	pbrt::Material::SP mat,
	std::map<pbrt::Material::SP, size_t> &materials, 
	const string& basedir) {
	
	if (materials.count(mat))	// already loaded...
		return materials[mat];

	Material::SharedPtr material = Material::SharedPtr(new Material(++materialIdAllocator, mat->name));
	material->mBsdfType			 = MaterialType::Disney;
	material->mShadingModel		 = Material::ShadingModel::MetallicRoughness;
	Material::MaterialParams &matParams = material->mMaterialParams;

	auto remap_roughness = [](const float &roughness) {
		float x = log(roughness);
		float val = 1.62142f + 0.819955f * x + 0.1734f * pow2(x) + 0.0171201f * pow3(x) +
					0.000640711f * pow4(x);
		return (val);
	};

	if (auto m = std::dynamic_pointer_cast<pbrt::DisneyMaterial>(mat)) {
		Log(Info, "Encountered disney material: %s", mat->name.c_str());
		material->mShadingModel	   = Material::ShadingModel::MetallicRoughness;
		matParams.diffuse = Vector4f(m->color.x, m->color.y, m->color.z, 1);
		matParams.IoR			   = m->eta;
		matParams.specular[2]	   = m->metallic;
		matParams.specular[1]	   = m->roughness;
	} else if (auto m = std::dynamic_pointer_cast<pbrt::PlasticMaterial>(mat)) {
		Log(Info, "Encountered plastic material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		matParams.diffuse = Vector4f(cast(m->kd), 1);
		if (m->map_kd) {
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				matParams.diffuse = Vector4f(cast(const_tex->value), 1);
			else loadTexture(material, m->map_kd, Material::TextureType::Diffuse, basedir);
		}
		const Vector3f ks(cast(m->ks));
		matParams.specular[3] = luminance(ks);
		float roughness		  = m->roughness;
		if (m->remapRoughness) roughness = remap_roughness(roughness);
		matParams.specular[1] = sqrt(roughness);		// sqrt'ed
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MatteMaterial>(mat)) {
		Log(Info, "Encountered matte material: %s", mat->name.c_str());
		matParams.diffuse = Vector4f(cast(m->kd), 1);
		if (m->map_kd) {
			Log(Debug, "A diffuse texture is found for %s texture %s", m->toString().c_str(),
				m->name.c_str());
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) 
				matParams.diffuse = Vector4f(cast(const_tex->value), 1);
			else loadTexture(material, m->map_kd, Material::TextureType::Diffuse, basedir);
		}
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.specular[3]	= 0;
	} else if (auto m = std::dynamic_pointer_cast<pbrt::SubstrateMaterial>(mat)) {
		Log(Info, "Encountered substrate material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.diffuse = Vector4f(cast(m->kd), 1);
		if (m->map_kd) {
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) 
				matParams.diffuse = Vector4f(cast(const_tex->value), 1);
			else loadTexture(material, m->map_kd, Material::TextureType::Diffuse, basedir);
		}
		Vector3f ks(cast(m->ks));
		float roughness = (m->uRoughness + m->vRoughness) / 2;
		if (m->remapRoughness) roughness = remap_roughness(roughness);
		matParams.specular = Vector4f(ks, 1 - sqrt(roughness));		// sqrt'ed
	} else if (auto m = std::dynamic_pointer_cast<pbrt::UberMaterial>(mat)) {
		Log(Info, "Encountered uber material: %s", mat->name.c_str());
		Vector3f diffuse		= cast(m->kd);
		Vector3f specular		= cast(m->ks);
		Vector3f transmission	= cast(m->kt);
		float roughness			= m->roughness;
		//roughness				= remap_roughness(roughness);
		if (m->map_kd) {
			Log(Debug, "A diffuse texture is found for %s texture %s", m->toString().c_str(),
				m->name.c_str());
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) 
				diffuse = cast(const_tex->value);
			else loadTexture(material, m->map_kd, Material::TextureType::Diffuse, basedir);
		}
		if (m->map_ks) {
			Log(Debug, "A specular texture is found for %s texture %s", m->toString().c_str(),
				m->name.c_str());
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_ks)) 
				specular = cast(const_tex->value);
			 else loadTexture(material, m->map_ks, Material::TextureType::Specular, basedir);
		}
		if (m->map_opacity) {
			Log(Info, "An opacity map is found for %s texture %s", m->toString().c_str(),
				m->name.c_str());
			if (auto tex = std::dynamic_pointer_cast<pbrt::ImageTexture>(m->map_opacity))
				loadTexture(material, m->map_opacity, Material::TextureType::Transmission, basedir);
		} 
		else if (cast(m->opacity) != Vector3f(1)) {
			Color3f transmission = Vector3f(1) - cast(m->opacity);
			material->setConstantTexture(Material::TextureType::Transmission, Color4f(transmission, 1));	
		}
		if (m->map_roughness) {
			Log(Info, "A roughness map is found for %s texture %s", m->toString().c_str(),
				m->name.c_str());
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_ks))
				roughness = luminance(cast(const_tex->value));
			else Log(Warning, "An image roughness map is currently not supported!");
		} 
		if (m->map_bump) {
			Log(Info, "A bump map is found for %s texture %s", m->toString().c_str(),
				m->name.c_str());
			if (auto tex = std::dynamic_pointer_cast<pbrt::ImageTexture>(m->map_bump))
				loadTexture(material, m->map_bump, Material::TextureType::Normal, basedir);

		}
		material->mShadingModel		   = Material::ShadingModel::SpecularGlossiness;
		matParams.diffuse			   = Vector4f(diffuse, matParams.diffuse[3]);
		matParams.specular			   = Vector4f(specular, 1 - sqrt(roughness));	// sqrt'ed
		matParams.specularTransmission = luminance(transmission);
	} else if (auto m = std::dynamic_pointer_cast<pbrt::TranslucentMaterial>(mat)) {
		Log(Warning, "Encountered not well-supported transluscent material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		matParams.diffuse = Vector4f(cast(m->kd), matParams.diffuse[3]);
		if (m->map_kd) {
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd))
				matParams.diffuse = Vector4f(cast(const_tex->value), matParams.diffuse[3]);
			else
				loadTexture(material, m->map_kd, Material::TextureType::Diffuse, basedir);
		}
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MirrorMaterial>(mat)) {
		// TODO: needs to make disney mirror-reflect when 0-roughness.
		Log(Info, "Encountered mirror material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::SpecularGlossiness;
		matParams.diffuse		= Vector4f(Vector3f(0), 1);
		matParams.specular		= Vector4f(cast(m->kr), 1);
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MetalMaterial>(mat)) {
		Log(Warning, "Encountered not well-supported metal material: %s", mat->name.c_str());
		material->mShadingModel = Material::ShadingModel::MetallicRoughness;
		Vector3f eta			= cast(m->eta);
		Vector3f eta_k			= cast(m->k);
		//matParams.diffuse		= Vector4f(eta_to_reflectivity(eta, eta_k), 0);
		matParams.diffuse = Vector4f(bsdf::FrComplex(1, eta, eta_k), 0);
		float roughness			= m->roughness;
		roughness				= (m->uRoughness + m->vRoughness) / 2;
		if (m->remapRoughness) roughness = remap_roughness(roughness);
		matParams.specular[1] = sqrt(roughness);	// sqrt'ed
		matParams.specular[2] = 0.8;				// manually set metallic
	} else if (auto m = std::dynamic_pointer_cast<pbrt::GlassMaterial>(mat)) {
		Log(Warning, "Encountered not well-supported glass material: %s", mat->name.c_str());
		material->mShadingModel		   = Material::ShadingModel::MetallicRoughness;
		matParams.specularTransmission = luminance(cast(m->kt));
		matParams.diffuse			   = Vector4f(cast(m->kt), 1);
		matParams.specular			   = Vector4f{ 0, 0, 0.2, 0 };
		matParams.IoR				   = m->index;
	} else {
		Log(Warning, "Encountered unsupported %s material: %s", 
			mat->toString().c_str(), mat->name.c_str());
		return 0;				// falling back to the default material 0...
	}
	if (matParams.IoR == 1)		// 1-ETA is not plausible for transmission
		matParams.IoR = 1.001;
	material->toDevice();
	size_t materialId = scene->mData.materials->size();
	scene->mData.materials->push_back(*material);
	materials[mat] = materialId;
	return materialId;
}

Mesh createMesh(pbrt::Shape::SP shape, const Transformf<> transform) {
	Mesh mesh;
	pbrt::TriangleMesh::SP m = std::dynamic_pointer_cast<pbrt::TriangleMesh>(shape);
	int n_vertices			 = m->vertex.size();
	int n_faces				 = m->getNumPrims();
	Log(Debug, "The current mesh %s has %d vertices and %d faces", m->toString().c_str(), n_vertices, n_faces);
	if (m->normal.size() < n_vertices)
		Log(Debug, "The current mesh has %zd normals but %d vertices, thus the normal(s) are ignored.",
			m->normal.size(), n_vertices);
	Matrixf<3, 3> rot		 = transform.rotation().inverse().transpose();
	mesh.vertices.reserve(n_vertices);
	mesh.indices.reserve(n_vertices);
	for (int i = 0; i < n_vertices; i++) {
		VertexAttribute vertex;
		Vector4f local_vertex(cast(m->vertex[i]), 1);
		Vector4f transformed_vertex = transform * local_vertex;
		Vector3f transformed_normal{};
		if (m->normal.size())
			transformed_normal = rot * cast(m->normal[i]);
		vertex.vertex				= transformed_vertex;
		vertex.normal				= transformed_normal;
		if (m->texcoord.size())
			vertex.texcoord			= cast(m->texcoord[i]);
		vertex.tangent				= getPerpendicular(vertex.normal);
		vertex.bitangent			= normalize(cross(vertex.normal, vertex.tangent));
		mesh.vertices.push_back(vertex);
	}
	for (int i = 0; i < n_faces; i++) {
		mesh.indices.push_back(cast(m->index[i]));
	}
	return mesh;
}

void createAreaLight(Mesh& mesh, pbrt::AreaLight::SP areaLight) {
	if (pbrt::DiffuseAreaLightRGB::SP l =
			std::dynamic_pointer_cast<pbrt::DiffuseAreaLightRGB>(areaLight)) {
		mesh.Le	 = cast(l->L);
	} else {
		Log(Warning, "Encountered unsupported area light: %s", areaLight->toString().c_str());
	}
}

bool PbrtImporter::import(const string &filepath, Scene::SharedPtr pScene) {
	string basepath		  = fs::path(filepath).parent_path().string();
	mBasepath			  = basepath;

	Log(Info, "Attempting to load pbrt scene from %s...", filepath.c_str());
	pbrt::Scene::SP scene = pbrt::importPBRT(filepath);
	if (!scene) {
		Log(Fatal, "Failed to load pbrt file from %s...", filepath.c_str());
		return false;
	}
		
	scene->makeSingleLevel();						// since currently kiraray supports only single gas...
	pScene->mData.materials->push_back(Material(0, "default material")); // the default material for shapes without
														 // material

	std::map<pbrt::Material::SP, size_t> pbrtMaterials; // loaded materials and its parametric id
	
	for (const pbrt::Shape::SP shape : scene->world->shapes) {
		// this makes you iterate through all shapes of the scene...
	}

	for (const pbrt::Instance::SP inst : scene->world->instances) {
		Matrixf<4, 4> transform = cast(inst->xfm);	// the instance's local transform
		for (const pbrt::Shape::SP geom: inst->object->shapes) {
			if (pbrt::TriangleMesh::SP m = std::dynamic_pointer_cast<pbrt::TriangleMesh>(geom)) {
				Mesh mesh = createMesh(m, transform);
				if (m->material) { 
					mesh.materialId = loadMaterial(pScene, m->material, pbrtMaterials, basepath);
				}
				if (m->areaLight) {
					createAreaLight(mesh, m->areaLight);
				}
				pScene->meshes.push_back(mesh);
				pScene->mAABB.extend(mesh.getAABB());
			} else {
				Log(Warning, "Encountered unsupported pbrt shape type: %s", geom->toString().c_str());
			}
		}
	}

	for (const pbrt::LightSource::SP light : scene->world->lightSources) {
		if (auto l = std::dynamic_pointer_cast<pbrt::InfiniteLightSource>(light)) {
			Log(Info, "Encountered infinite light source %s", l->mapName.c_str());
			//Texture image;
			//image.loadImage(resolve(l->mapName));
			//Vector2i size = image.getImage().getSize();
			//Color4f* rgba = image::convertEqualAeraOctahedralMappingToSpherical((Color4f *) image.getImage().data(),
			//													size[0], size[1]);
			//delete[] image.getImage().data();
			//image.getImage().reset((uchar *) rgba);
			//image.toDevice();
			//pScene->addInfiniteLight(InfiniteLight(image));
#ifdef USE_PBRT_ENVMAP
			pScene->addInfiniteLight(InfiniteLight(resolve(l->mapName)));
#endif
		}
	}
	return true;
}

string PbrtImporter::resolve(string path) {
	return (fs::path(mBasepath) / path).string();
}

KRR_NAMESPACE_END