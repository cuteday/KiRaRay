#include <pbrtParser/Scene.h>
#include "importer.h"

KRR_NAMESPACE_BEGIN

using namespace importer;

namespace {
	static uint textureIdAllocator	= 0;
	static uint materialIdAllocator = 0;
	static MaterialLoader sMaterialLoader;

	Matrixf<4, 4> cast(pbrt::affine3f xfm) {
		return Matrixf<4, 4>{
			{ xfm.l.vx.x, xfm.l.vy.x, xfm.l.vz.x, xfm.p.x },
			{ xfm.l.vx.y, xfm.l.vy.y, xfm.l.vz.y, xfm.p.y },
			{ xfm.l.vx.z, xfm.l.vy.z, xfm.l.vz.z, xfm.p.z },
			{ 0, 0, 0, 1 }
		};
		//return Matrixf<4, 4>{
		//	{ xfm.l.vx.x, xfm.l.vx.y, xfm.l.vx.z, 0.f },
		//	{ xfm.l.vy.x, xfm.l.vy.y, xfm.l.vy.z, 0.f },
		//	{ xfm.l.vz.x, xfm.l.vz.y, xfm.l.vz.z, 0.f },
		//	{ xfm.p.x, xfm.p.y, xfm.p.z, 1.f }
		//};
	}
} 

Material::SharedPtr loadMaterial(pbrt::Material::SP mat,
	std::map<pbrt::Material::SP, Material::SharedPtr>& materials, 
	std::map<pbrt::Texture::SP, Texture>& texture) {
	
	if (materials.count(mat))	// already loaded...
		return materials[mat];

	int materialId = materials.size() + 1; 
	Material::SharedPtr material = Material::SharedPtr(new Material(materialId, mat->name));
	material->mBsdfType			 = BsdfType::Disney;
	material->mShadingModel		 = Material::ShadingModel::MetallicRoughness;
	Material::MaterialParams &matParams = material->mMaterialParams;


	if (auto m = std::dynamic_pointer_cast<pbrt::DisneyMaterial>(mat)) {
		matParams.diffuse = Vector4f(m->color.x, m->color.y, m->color.z, matParams.diffuse[3]);
		matParams.IoR			   = m->eta;
		matParams.specular[2]	   = m->metallic;
		matParams.specular[1]	   = m->roughness;
	} else if (auto m = std::dynamic_pointer_cast<pbrt::PlasticMaterial>(mat)) {
		matParams.diffuse = Vector4f(m->kd.x, m->kd.y, m->kd.z, matParams.diffuse[3]);
		if (m->map_kd) {
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) {
				matParams.diffuse = Vector4f(const_tex->value.x, const_tex->value.y,
											 const_tex->value.z, matParams.diffuse[3]);
			} else {
				// TODO: load image textures!
			}
		}

		const Vector3f ks(m->ks.x, m->ks.y, m->ks.z);
		//loaded_mat.specular	 = luminance(ks);
		//loaded_mat.roughness = m->roughness;
	} else if (auto m = std::dynamic_pointer_cast<pbrt::MatteMaterial>(mat)) {
		matParams.diffuse = Vector4f(m->kd.x, m->kd.y, m->kd.z, matParams.diffuse[3]);
		if (m->map_kd) {
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) {
				matParams.diffuse = Vector4f(const_tex->value.x, const_tex->value.y,
											 const_tex->value.z, matParams.diffuse[3]);
			} else {

			}
		}
	} else if (auto m = std::dynamic_pointer_cast<pbrt::SubstrateMaterial>(mat)) {
		matParams.diffuse = Vector4f(m->kd.x, m->kd.y, m->kd.z, matParams.diffuse[3]);
		if (m->map_kd) {
			if (auto const_tex = std::dynamic_pointer_cast<pbrt::ConstantTexture>(m->map_kd)) {
				matParams.diffuse = Vector4f(const_tex->value.x, const_tex->value.y,
											 const_tex->value.z, matParams.diffuse[3]);
			} else {

			}
		}
		// Sounds like this is kind of what the SubstrateMaterial acts like? Diffuse with a
		// specular and clearcoat?
		const Vector3f ks(m->ks.x, m->ks.y, m->ks.z);
		//loaded_mat.specular		   = luminance(ks);
		//loaded_mat.roughness	   = 1;
	} else {
		Log(Warning, "Unsupported material type: %s", mat->toString().c_str());
		return nullptr;
	}
}

bool PbrtImporter::import(const string &filepath, Scene::SharedPtr pScene) {
	string basepath		  = fs::path(filepath).parent_path().string();
	pbrt::Scene::SP scene = pbrt::importPBRT(filepath, basepath);
	if (!scene) {
		Log(Fatal, "Failed to load pbrt file from %s...", filepath.c_str());
		return false;
	}
		
	scene->makeSingleLevel();		// since currently kiraray supports only single gas...

	std::map<pbrt::Material::SP, Material::SharedPtr> pbrtMaterials; // loaded materials and its parametric id
	std::map<pbrt::Texture::SP, Texture> pbrtTextures;	 // loaded materials and its parametric id

	for (const pbrt::Shape::SP shape : scene->world->shapes) {
		// iterate through all shapes of the scene...
	}

	for (const pbrt::Instance::SP inst : scene->world->instances) {
		Matrixf<4, 4> transform = cast(inst->xfm);
	}

	return true;
}

KRR_NAMESPACE_END