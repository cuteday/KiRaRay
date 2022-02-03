#include <filesystem>

#include "assimp/Importer.hpp"
#include "assimp/DefaultLogger.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/pbrmaterial.h"

#include "scene/importer.h"
#include "logger.h"

KRR_NAMESPACE_BEGIN

namespace fs = std::filesystem;

namespace assimp {

	MaterialLoader sMaterialLoader;

    vec3f aiCast(const aiColor3D& ai) { return vec3f(ai.r, ai.g, ai.b); }
    vec3f aiCast(const aiVector3D& val) { return vec3f(val.x, val.y, val.z); }
    quat aiCast(const aiQuaternion& q) { return quat(q.w, q.x, q.y, q.z); }

	struct TextureMapping
	{
		aiTextureType aiType;
		unsigned int aiIndex;
		Material::TextureType targetType;
	};

	enum class ImportMode {
		Default,
		OBJ,
		GLTF2,
	};

	static const std::vector<TextureMapping> sTextureMapping = {
		{ aiTextureType_DIFFUSE, 0, Material::TextureType::Diffuse },
		{ aiTextureType_SPECULAR, 0, Material::TextureType::Specular },
		{ aiTextureType_EMISSIVE, 0, Material::TextureType::Emissive },
		// OBJ does not offer a normal map, thus we use the bump map instead.
		{ aiTextureType_NORMALS, 0, Material::TextureType::Normal },
		{ aiTextureType_HEIGHT, 0, Material::TextureType::Normal },
		{ aiTextureType_DISPLACEMENT, 0, Material::TextureType::Normal },
	};

	float convertSpecPowerToRoughness(float specPower){
		return clamp(sqrt(2.0f / (specPower + 2.0f)), 0.f, 1.f);
	}

	void loadTextures(const aiMaterial* pAiMaterial, const std::string& folder, Material::SharedPtr pMaterial) {

		for (const auto& source : sTextureMapping)
		{
			// Skip if texture of requested type is not available
			if (pAiMaterial->GetTextureCount(source.aiType) < source.aiIndex + 1) continue;

			// Get the texture name
			aiString aiPath;
			pAiMaterial->GetTexture(source.aiType, source.aiIndex, &aiPath);
			std::string path(aiPath.data);
			if (path.empty())
			{
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

    Material::SharedPtr createMaterial(const aiMaterial* pAiMaterial, const string &modelFolder, ImportMode importMode = ImportMode::Default) {
		aiString name;
		pAiMaterial->Get(AI_MATKEY_NAME, name);

		// Parse the name
		std::string nameStr = std::string(name.C_Str());
		if (nameStr.empty())
		{
			logWarning("Material with no name found -> renaming to 'unnamed'");
			nameStr = "unnamed";
		}
		Material::SharedPtr pMaterial = Material::SharedPtr(new Material());

		// Load textures. Note that loading is affected by the current shading model.
		loadTextures(pAiMaterial, modelFolder, pMaterial);

		// Opacity
		float opacity = 1.f;
		if (pAiMaterial->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
		{
			pMaterial->mMaterialParams.diffuse.a = opacity;
		}

		// Shininess
		float shininess;
		if (pAiMaterial->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS)
		{
			// Convert OBJ/MTL Phong exponent to glossiness.
			if (importMode == ImportMode::OBJ)
			{
				float roughness = convertSpecPowerToRoughness(shininess);
				shininess = 1.f - roughness;
			}
			pMaterial->mMaterialParams.specular.a = shininess;
		}

		//// Refraction
		float refraction;
		if (pAiMaterial->Get(AI_MATKEY_REFRACTI, refraction) == AI_SUCCESS) 
			pMaterial->mMaterialParams.IoR = refraction;

		//// Diffuse color
		aiColor3D color;
		if (pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
		{
			vec4f diffuse = vec4f(color.r, color.g, color.b, 
				pMaterial->mMaterialParams.diffuse.a);
			pMaterial->mMaterialParams.diffuse = diffuse;
		}

		// Specular color
		if (pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
		{
			vec4f specular = vec4f(color.r, color.g, color.b, pMaterial->mMaterialParams.specular.a);
			pMaterial->mMaterialParams.specular = specular;
		}

		// Emissive color
		if (pAiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS)
		{
			vec3f emissive = vec3f(color.r, color.g, color.b);
			pMaterial->mMaterialParams.emissive = emissive;
		}

		//// Double-Sided
		//int isDoubleSided;
		//if (pAiMaterial->Get(AI_MATKEY_TWOSIDED, isDoubleSided) == AI_SUCCESS)
		//{
		//	pMaterial->setDoubleSided(isDoubleSided);
		//}

		//// Use scalar opacity value for controlling specular transmission
		//// TODO: Remove this workaround when we have a better way to define materials.
		//if (opacity < 1.f)
		//{
		//	pMaterial->setSpecularTransmission(1.f - opacity);
		//}

		return pMaterial;
    }
}

void MaterialLoader::loadTexture(const Material::SharedPtr& pMaterial, TextureType type, const std::string& filename)
{
	assert(pMaterial);
	bool srgb = mUseSrgb;
	std::string fullPath;
	if (!fs::exists(filename)) {
		logWarning("Can't find texture image file '" + filename + "'");
		return;
	}
	TextureKey textureKey{ fullPath, srgb };
	if (mTextureCache.count(textureKey)) {
		mTextureCache[textureKey] = Texture::createFromFile(fullPath, srgb);
	}

	pMaterial->setTexture(type, *mTextureCache[textureKey]);
}

using namespace krr::assimp;
bool AssimpImporter::import(const string& filepath, const Scene::SharedPtr pScene) {
    Assimp::DefaultLogger::create("", Assimp::Logger::NORMAL, aiDefaultLogStream_STDOUT);
    Assimp::DefaultLogger::get()->info("KRR::Assimp::DefaultLogger initialized!");

    unsigned int postProcessSteps = 0
        | aiProcess_CalcTangentSpace
        //| aiProcess_JoinIdenticalVertices
        //| aiProcess_MakeLeftHanded
        | aiProcess_Triangulate
        //| aiProcess_RemoveComponent
        //| aiProcess_GenNormals
        | aiProcess_GenSmoothNormals
        | aiProcess_RemoveRedundantMaterials
        //| aiProcess_FixInfacingNormals
        | aiProcess_SortByPType
        | aiProcess_GenUVCoords
        | aiProcess_TransformUVCoords
        //| aiProcess_FlipUVs
        //| aiProcess_FlipWindingOrder
        ;

    Assimp::Importer importer;
    mFilepath = filepath;
    mpScene = pScene;
    mpAiScene = (aiScene*)importer.ReadFile(filepath, postProcessSteps);
    if (!mpAiScene) logFatal("Assimp::load model failed");

    string modelFolder = std::filesystem::path(filepath).parent_path().string();
    loadMaterials(modelFolder);

    traverseNode(mpAiScene->mRootNode, aiMatrix4x4());

    logDebug("Total imported meshes: " + std::to_string(mpScene->meshes.size()));

    Assimp::DefaultLogger::kill();
    
    return true;
}

void AssimpImporter::processMesh(aiMesh* pAiMesh, aiMatrix4x4 transform)
{
    Mesh mesh;

    for (uint i = 0; i < pAiMesh->mNumVertices; i++) {
        vec3f vertex = aiCast(pAiMesh->mVertices[i]);
        mesh.vertices.push_back(vertex);

        assert(pAiMesh->HasNormals());
        vec3f normal = aiCast(pAiMesh->mNormals[i]);
        mesh.normals.push_back(normal);

        if (pAiMesh->mTextureCoords[0]) {
            vec3f texcoord = aiCast(pAiMesh->mTextureCoords[0][i]);
            mesh.texcoords.push_back({ texcoord.x, texcoord.y });
        }

    }

    for (uint i = 0; i < pAiMesh->mNumFaces; i++) {
        aiFace face = pAiMesh->mFaces[i];
        assert(face.mNumIndices == 3);
        vec3i indices = { (int)face.mIndices[0], (int)face.mIndices[1], (int)face.mIndices[2] };

        mesh.indices.push_back(indices);
    }

	//mesh.mMaterial = mpScene->materials[pAiMesh->mMaterialIndex];
    mpScene->meshes.push_back(mesh);
}

void AssimpImporter::traverseNode(aiNode* node, aiMatrix4x4 transform)
{
    transform = transform * node->mTransformation;

    for (int i = 0; i < node->mNumMeshes; i++) {
        aiMesh* mesh = mpAiScene->mMeshes[node->mMeshes[i]];
        processMesh(mesh, transform);
    }

    for (int i = 0; i < node->mNumChildren; i++) {
        traverseNode(node->mChildren[i], transform);
    }
}

void AssimpImporter::loadMaterials(const string &modelFolder)
{
    for (uint i = 0; i < mpAiScene->mNumMaterials; i++) {
        const aiMaterial* aiMaterial = mpAiScene->mMaterials[i];
        Material::SharedPtr pMaterial = createMaterial(aiMaterial, modelFolder);
        if (pMaterial == nullptr) {
            logError("Failed to create material...");
            return;
        }
		// we transfer alll material data to gpu memory here.
		pMaterial->toDevice();
        //mpScene->materials.push_back(*pMaterial);
    }
}

KRR_NAMESPACE_END