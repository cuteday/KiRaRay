#include <filesystem>

#include "assimp/DefaultLogger.hpp"
#include "assimp/Importer.hpp"
#include "assimp/pbrmaterial.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include "light.h"
#include "logger.h"

#include "scene.h"
#include "scene/importer.h"
#include "util/string.h"

KRR_NAMESPACE_BEGIN

namespace fs = std::filesystem;
using namespace importer;

namespace assimp {

using ImportMode = AssimpImporter::ImportMode;

MaterialLoader sMaterialLoader;

Vector3f aiCast(const aiColor3D &ai) { return Vector3f(ai.r, ai.g, ai.b); }
Vector3f aiCast(const aiVector3D &val) { return Vector3f(val.x, val.y, val.z); }
Quaternionf aiCast(const aiQuaternion &q) {
	return Quaternionf{q.w, q.x, q.y, q.z};
}
AABB aiCast(const aiAABB &aabb) {
	return AABB(aiCast(aabb.mMin), aiCast(aabb.mMax));
}
Matrix4f aiCast(const aiMatrix4x4 &m) { return Matrix4f{{m.a1, m.a2, m.a3, m.a4}, 
					{m.b1, m.b2, m.b3, m.b4},
					{m.c1, m.c2, m.c3, m.c4},
					{m.d1, m.d2, m.d3, m.d4}};
}

struct TextureMapping {
	aiTextureType aiType;
	unsigned int aiIndex;
	Material::TextureType targetType;
};

static const std::vector<TextureMapping> sTextureMapping = {
	{aiTextureType_DIFFUSE, 0, Material::TextureType::Diffuse},
	{aiTextureType_SPECULAR, 0, Material::TextureType::Specular},
	{aiTextureType_EMISSIVE, 0, Material::TextureType::Emissive},
	// OBJ does not offer a normal map, thus we use the bump map instead.
	{aiTextureType_NORMALS, 0, Material::TextureType::Normal},
	{aiTextureType_HEIGHT, 0, Material::TextureType::Normal},
	{aiTextureType_DISPLACEMENT, 0, Material::TextureType::Normal},
	// for GLTF2
	{AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
	 Material::TextureType::Specular}};

float convertSpecPowerToRoughness(float specPower) {
	// normally, the specular weight (Ns) ranges 0 - 1000.
	// for rendering specular surfaces, we let it be specular when Ns >= 1000.
	if (specPower >= 1000) return 0;
	return clamp(sqrt(2.0f / (specPower + 2.0f)), 0.f, 1.f);
}

void loadTextures(const aiMaterial *pAiMaterial, const std::string &folder,
				  Material::SharedPtr pMaterial) {

	for (const auto &source : sTextureMapping) {
		// Skip if texture of requested type is not available
		if (pAiMaterial->GetTextureCount(source.aiType) < source.aiIndex + 1)
			continue;

		// Get the texture name
		aiString aiPath;
		pAiMaterial->GetTexture(source.aiType, source.aiIndex, &aiPath);
		std::string path(aiPath.data);
		if (path.empty()) {
			logWarning("Texture has empty file name, ignoring.");
			continue;
		}
		// Load the texture
		std::filesystem::path filepath(folder);
		filepath /= path;
		std::string filename = filepath.string();
		sMaterialLoader.loadTexture(pMaterial, source.targetType, filename);
	}
}

Material::SharedPtr createMaterial(const aiMaterial *pAiMaterial, const string &modelFolder,
			   ImportMode importMode = ImportMode::Default) {
	aiString name;
	pAiMaterial->Get(AI_MATKEY_NAME, name);

	// Parse the name
	std::string nameStr = std::string(name.C_Str());
	if (nameStr.empty()) {
		logWarning("Material with no name found -> renaming to 'unnamed'");
		nameStr = "unnamed";
	}
	Material::SharedPtr pMaterial = std::make_shared<Material>(nameStr);

	logDebug("Importing material: " + nameStr);
	// Load textures. Note that loading is affected by the current shading
	// model.
	loadTextures(pAiMaterial, modelFolder, pMaterial);

	// Opacity
	float opacity = 1;
	if (pAiMaterial->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
		pMaterial->mMaterialParams.diffuse[3] = opacity;
		if (opacity == 0.f) {
			pMaterial->mMaterialParams.specularTransmission = 1 - opacity;
			pMaterial->mBsdfType = MaterialType::Dielectric;
			Log(Info, "The material %s has a opacity: %f", nameStr.c_str(),
				opacity);
		}
	}

	// Shininess
	float shininess;
	if (pAiMaterial->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
		// Convert OBJ/MTL Phong exponent to glossiness.
		if (importMode == ImportMode::OBJ) {
			Log(Debug, "The OBJ material %s has a shininess of %f",
				nameStr.c_str(), shininess);
			float roughness = convertSpecPowerToRoughness(shininess);
			shininess		= 1.f - roughness;
		}
		pMaterial->mMaterialParams.specular[3] = shininess;
	}

	// Refraction
	float refraction;
	if (pAiMaterial->Get(AI_MATKEY_REFRACTI, refraction) == AI_SUCCESS) {
		pMaterial->mMaterialParams.IoR = refraction;
		logDebug("IoR: " + to_string(refraction));
	}

	// Diffuse color
	aiColor3D color;

	if (pAiMaterial->Get(AI_MATKEY_COLOR_TRANSPARENT, color) == AI_SUCCESS) {
		Color transmission = 1.f - Color(color[0], color[1], color[2]);
		pMaterial->mMaterialParams.specularTransmission =
			luminance(transmission);
		if (luminance(transmission) > 1 - M_EPSILON)
			pMaterial->mBsdfType = MaterialType::Dielectric; 
		logDebug("transmission: " + to_string(luminance(transmission)));
	}

	if (pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
		Vector4f diffuse = Vector4f(color[0], color[1], color[2],
									pMaterial->mMaterialParams.diffuse[3]);
		pMaterial->mMaterialParams.diffuse = diffuse;
	}

	// Specular color
	if (pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
		Vector4f specular = Vector4f(color[0], color[1], color[2],
									 pMaterial->mMaterialParams.specular[3]);
		pMaterial->mMaterialParams.specular = specular;
		logDebug("specular : " + to_string(specular[0]) + " " +
				 to_string(specular[1]) + " " + to_string(specular[2]) + " ");
	}

	// Emissive color
	if (pAiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS) {
		Color3f emissive = Vector3f(color[0], color[1], color[2]);
		if (emissive.any()) {
			pMaterial->setConstantTexture(Material::TextureType::Emissive,
										  Color4f(emissive, 1));
		}
	}

	// Double-Sided
	int isDoubleSided;
	if (pAiMaterial->Get(AI_MATKEY_TWOSIDED, isDoubleSided) == AI_SUCCESS) {
		// pMaterial->mDoubleSided = true;
	}

	if (importMode == ImportMode::GLTF2) {
		if (pAiMaterial->Get(
				AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR, color) ==
			AI_SUCCESS) {
			Vector4f baseColor =
				Vector4f(color[0], color[1], color[2],
						 pMaterial->mMaterialParams.diffuse[3]);
			pMaterial->mMaterialParams.diffuse = baseColor;
		}
		Vector4f specularParams = pMaterial->mMaterialParams.specular;
		float metallic;
		if (pAiMaterial->Get(
				AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR,
				metallic) == AI_SUCCESS) {
			specularParams[2] = metallic;
		}
		float roughness;
		if (pAiMaterial->Get(
				AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR,
				roughness) == AI_SUCCESS) {
			specularParams[1] = roughness;
		}

		pMaterial->mMaterialParams.specular = specularParams;
	}

	pMaterial->mShadingModel = importMode == ImportMode::OBJ
								   ? Material::ShadingModel::SpecularGlossiness
								   : Material::ShadingModel::MetallicRoughness;

	return pMaterial;
}
} // namespace assimp

void MaterialLoader::loadTexture(const Material::SharedPtr &pMaterial,
								 TextureType type, const std::string &filename,
								 bool flip) {
	assert(pMaterial);
	bool srgb = mUseSrgb && pMaterial->determineSrgb(filename, type);
	if (!fs::exists(filename)) {
		logWarning("Can't find texture image file '" + filename + "'");
		return;
	}
	TextureKey textureKey{filename, srgb};
	if (!mTextureCache.count(textureKey)) {
		mTextureCache[textureKey] =
			std::make_shared<Texture>(filename, flip, srgb);
	}
	pMaterial->setTexture(type, mTextureCache[textureKey]);
}

using namespace krr::assimp;
bool AssimpImporter::import(const fs::path filepath,
							const Scene::SharedPtr pScene) {
	Assimp::DefaultLogger::create("", Assimp::Logger::NORMAL,
								  aiDefaultLogStream_STDOUT);
	Assimp::DefaultLogger::get()->info(
		"KRR::Assimp::DefaultLogger initialized!");

	unsigned int postProcessSteps = 0 
									| aiProcess_CalcTangentSpace
									| aiProcess_FindDegenerates
									| aiProcess_OptimizeMeshes
									| aiProcess_OptimizeGraph
									| aiProcess_JoinIdenticalVertices 
									| aiProcess_FindInvalidData
									//| aiProcess_MakeLeftHanded
									| aiProcess_Triangulate
									//| aiProcess_RemoveComponent
									//| aiProcess_GenNormals
									| aiProcess_GenSmoothNormals
									//| aiProcess_RemoveRedundantMaterials
									//| aiProcess_FixInfacingNormals
									| aiProcess_SortByPType 
									| aiProcess_GenUVCoords
									//| aiProcess_TransformUVCoords
									//| aiProcess_PreTransformVertices
									| aiProcess_FlipUVs
									//| aiProcess_FlipWindingOrder 
									| aiProcess_GenBoundingBoxes;
	int removeFlags = aiComponent_COLORS;
	for (uint32_t uvLayer = 1; uvLayer < AI_MAX_NUMBER_OF_TEXTURECOORDS;
		 uvLayer++)
		removeFlags |= aiComponent_TEXCOORDSn(uvLayer);

	mFilepath = filepath.string();
	mScene	  = pScene;

	Assimp::Importer importer;
	importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeFlags);
	importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 60);

	logDebug("Start loading scene with assimp importer");
	mAiScene = (aiScene *) importer.ReadFile(filepath.string(), postProcessSteps);
	if (!mAiScene) logFatal("Assimp::load model failed");

	logDebug("Start loading materials");

	string modelFolder = filepath.parent_path().string();
	string modelSuffix = filepath.extension().string();
	if (modelSuffix == ".obj") {
		mImportMode = ImportMode::OBJ;
		logInfo(
			"Importing OBJ model, using Specular Glossiness shading method.");
	} else if (modelSuffix == ".gltf" || modelSuffix == ".glb") {
		mImportMode = ImportMode::GLTF2;
		logInfo("Importing GLTF2 model.");
	}

	auto root = std::make_shared<SceneGraphNode>();
	mScene->getSceneGraph()->setRoot(root);
	loadMaterials(modelFolder);
	loadMeshes();
	traverseNode(mAiScene->mRootNode, root);
	root->setName(filepath.filename().string());
	Assimp::DefaultLogger::kill();
	return true;
}

void AssimpImporter::traverseNode(aiNode *assimpNode, SceneGraphNode::SharedPtr graphNode) {
	Affine3f transform = aiCast(assimpNode->mTransformation);
	graphNode->setName(std::string(assimpNode->mName.C_Str()));
	graphNode->setRotation(Quaternionf(transform.rotation()));
	graphNode->setScaling(transform.scaling());
	graphNode->setTranslation(transform.translation());
	for (int i = 0; i < assimpNode->mNumMeshes; i++) {
		// add this mesh into scenegraph
		Mesh::SharedPtr mesh = mScene->getMeshes()[assimpNode->mMeshes[i]];
		auto meshInstance = std::make_shared<MeshInstance>(mesh);
		mScene->getSceneGraph()->attachLeaf(graphNode, meshInstance);
	}
	for (int i = 0; i < assimpNode->mNumChildren; i++) {
		SceneGraphNode::SharedPtr childNode = std::make_shared<SceneGraphNode>();
		mScene->getSceneGraph()->attach(graphNode, childNode);
		traverseNode(assimpNode->mChildren[i], childNode);
	}
}

void AssimpImporter::loadMaterials(const string &modelFolder) {
	auto defaultMaterial = std::make_shared<Material>("default material");
	mScene->addMaterial(defaultMaterial);
	for (uint i = 0; i < mAiScene->mNumMaterials; i++) {
		const aiMaterial *aiMaterial = mAiScene->mMaterials[i];
		Material::SharedPtr pMaterial = createMaterial(aiMaterial, modelFolder, mImportMode);
		if (pMaterial == nullptr) {
			logError("Failed to create material...");
			return;
		}
		mScene->addMaterial(pMaterial);
	}
}

void AssimpImporter::loadMeshes() {
	for (uint i = 0; i < mAiScene->mNumMeshes; i++) {
		aiMesh *pAiMesh = mAiScene->mMeshes[i];
		auto mesh		= std::make_shared<Mesh>();
		mesh->indices.reserve(pAiMesh->mNumFaces);

		assert(pAiMesh->HasNormals());
		for (uint i = 0; i < pAiMesh->mNumVertices; i++) {
			Vector3f normal = normalize(aiCast(pAiMesh->mNormals[i]));
			Vector3f T, B;

			if (pAiMesh->HasTangentsAndBitangents() && (pAiMesh->mTangents != NULL)) {
				T = aiCast(pAiMesh->mTangents[i]);
				//  in assimp the tangents and bitangents are not necessarily
				//  orthogonal! however we need them to be orthonormal since we use
				//  tbn as world-local transformations
				T = normalize(T - normal * dot(normal, T));
				mesh->tangents.push_back(T);
			}
			mesh->positions.push_back(aiCast(pAiMesh->mVertices[i]));
			if (pAiMesh->HasTextureCoords(0)) {
				Vector3f texcoord = Vector3f(aiCast(pAiMesh->mTextureCoords[0][i]));
				mesh->texcoords.push_back(texcoord);
			} else
				Log(Debug, "Lost UV coords when importing with Assimp");
			mesh->normals.push_back(normal);
		}

		for (uint i = 0; i < pAiMesh->mNumFaces; i++) {
			aiFace face = pAiMesh->mFaces[i];
			assert(face.mNumIndices == 3);
			Vector3i indices = {(int) face.mIndices[0], (int) face.mIndices[1],
								(int) face.mIndices[2]};
			mesh->indices.push_back(indices);
		}

		mesh->material = mScene->getMaterials()[pAiMesh->mMaterialIndex + 1];
		mesh->aabb	   = aiCast(pAiMesh->mAABB);
		mesh->setName(std::string(pAiMesh->mName.C_Str()));
		mScene->getSceneGraph()->addMesh(mesh);
	}
}

KRR_NAMESPACE_END