#include <filesystem>

#include "assimp/DefaultLogger.hpp"
#include "assimp/Importer.hpp"
#include "assimp/pbrmaterial.h"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

#include "light.h"
#include "logger.h"
#include "math/math.h"
#include "scene/sgimporter.h"
#include "util/string.h"

KRR_NAMESPACE_BEGIN

namespace fs = std::filesystem;

namespace {
static uint textureIdAllocator	= 0;
static uint materialIdAllocator = 0;

static uint sIdGroup{ 0 };
static uint sIdMesh{ 0 };
static uint sIdInstance{ 0 };

using ImportMode = SceneGraphImporter::ImportMode;

MaterialLoader sMaterialLoader;

Vector3f aiCast(const aiColor3D &ai) { return Vector3f(ai[0], ai[1], ai[2]); }
Vector3f aiCast(const aiVector3D &val) { return Vector3f(val[0], val[1], val[2]); }
Quaternionf aiCast(const aiQuaternion &q) { return Quaternionf{ q.w, q.x, q.y, q.z }; }
AABB aiCast(const aiAABB &aabb) { return AABB(aiCast(aabb.mMin), aiCast(aabb.mMax)); }
Matrixf<4, 4> aiCast(const aiMatrix4x4 &m) {
	Matrixf<4, 4> ret;
	for (int i = 0; i < 4; ++i) 
		for (int j = 0; j < 4; ++j) 
			ret(i, j) = m[i][j];
	return ret;
}

struct TextureMapping {
	aiTextureType aiType;
	unsigned int aiIndex;
	Material::TextureType targetType;
};

static const std::vector<TextureMapping> sTextureMapping = {
	{ aiTextureType_DIFFUSE, 0, Material::TextureType::Diffuse },
	{ aiTextureType_SPECULAR, 0, Material::TextureType::Specular },
	{ aiTextureType_EMISSIVE, 0, Material::TextureType::Emissive },
	// OBJ does not offer a normal map, thus we use the bump map instead.
	{ aiTextureType_NORMALS, 0, Material::TextureType::Normal },
	{ aiTextureType_HEIGHT, 0, Material::TextureType::Normal },
	{ aiTextureType_DISPLACEMENT, 0, Material::TextureType::Normal },
	// for GLTF2
	{ AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE,
	  Material::TextureType::Specular }
};

float convertSpecPowerToRoughness(float specPower) {
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
	Material::SharedPtr pMaterial =
		Material::SharedPtr(new Material(++materialIdAllocator, nameStr));

	logDebug("Importing material: " + nameStr);
	// Load textures. Note that loading is affected by the current shading model.
	loadTextures(pAiMaterial, modelFolder, pMaterial);

	// Opacity
	float opacity = 1;
	if (pAiMaterial->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS) {
		pMaterial->mMaterialParams.diffuse[3] = opacity;
		if (opacity < 1.f)
			pMaterial->mMaterialParams.specularTransmission = 1 - opacity;
		logDebug("opacity: " + to_string(opacity));
	}

	// Shininess
	float shininess;
	if (pAiMaterial->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS) {
		// Convert OBJ/MTL Phong exponent to glossiness.
		if (importMode == ImportMode::OBJ) {
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
		Color transmission								= 1.f - Color(color[0], color[1], color[2]);
		pMaterial->mMaterialParams.specularTransmission = luminance(transmission);
		logDebug("transmission: " + to_string(luminance(transmission)));
	}

	if (pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS) {
		Vector4f diffuse =
			Vector4f(color[0], color[1], color[2], pMaterial->mMaterialParams.diffuse[3]);
		pMaterial->mMaterialParams.diffuse = diffuse;
	}

	// Specular color
	if (pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS) {
		Vector4f specular =
			Vector4f(color[0], color[1], color[2], pMaterial->mMaterialParams.specular[3]);
		pMaterial->mMaterialParams.specular = specular;
		logDebug("specular : " + to_string(specular[0]) + " " + to_string(specular[1]) + " " +
				 to_string(specular[2]) + " ");
	}

	// Emissive color
	if (pAiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS) {
		Vector3f emissive					= Vector3f(color[0], color[1], color[2]);
		pMaterial->mMaterialParams.emissive = emissive;
	}

	// Double-Sided
	int isDoubleSided;
	if (pAiMaterial->Get(AI_MATKEY_TWOSIDED, isDoubleSided) == AI_SUCCESS) {
		pMaterial->mDoubleSided = true;
	}

	if (importMode == ImportMode::GLTF2) {
		if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR, color) ==
			AI_SUCCESS) {
			Vector4f baseColor =
				Vector4f(color[0], color[1], color[2], pMaterial->mMaterialParams.diffuse[3]);
			pMaterial->mMaterialParams.diffuse = baseColor;
		}
		Vector4f specularParams = pMaterial->mMaterialParams.specular;
		float metallic;
		if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, metallic) ==
			AI_SUCCESS) {
			specularParams[2] = metallic;
		}
		float roughness;
		if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR, roughness) ==
			AI_SUCCESS) {
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

using namespace texture;

void MaterialLoader::loadTexture(const Material::SharedPtr &pMaterial, TextureType type,
								 const std::string &filename) {
	assert(pMaterial);
	bool srgb = mUseSrgb && pMaterial->determineSrgb(filename, type);
	if (!fs::exists(filename)) {
		logWarning("Can't find texture image file '" + filename + "'");
		return;
	}
	TextureKey textureKey{ filename, srgb };
	if (!mTextureCache.count(textureKey)) {
		mTextureCache[textureKey] = Texture(filename, srgb, ++textureIdAllocator);
	}
	pMaterial->setTexture(type, mTextureCache[textureKey]);
}

bool SceneGraphImporter::import(const string &filepath, const Scene::SharedPtr pScene) {
	Assimp::DefaultLogger::create("", Assimp::Logger::NORMAL, aiDefaultLogStream_STDOUT);
	Assimp::DefaultLogger::get()->info("KRR::Assimp::DefaultLogger initialized!");

	std::ifstream fin(filepath);
	if (fin.fail()) {
		logError("Scene file not found!", false);
		return false;
	} else {
		fin.close();
	}
	
	unsigned int postProcessSteps = 0 |
									aiProcess_CalcTangentSpace
									//| aiProcess_OptimizeMeshes
									| aiProcess_JoinIdenticalVertices |
									aiProcess_FindInvalidData
									//| aiProcess_MakeLeftHanded
									| aiProcess_Triangulate
									//| aiProcess_RemoveComponent
									//| aiProcess_GenNormals
									| aiProcess_GenSmoothNormals
									//| aiProcess_RemoveRedundantMaterials
									//| aiProcess_FixInfacingNormals
									| aiProcess_SortByPType |
									aiProcess_GenUVCoords
									//| aiProcess_TransformUVCoords
									| aiProcess_FlipUVs
									//| aiProcess_FlipWindingOrder
									//| aiProcess_PreTransformVertices 
									| aiProcess_GenBoundingBoxes;
	int removeFlags = aiComponent_COLORS;
	for (uint32_t uvLayer = 1; uvLayer < AI_MAX_NUMBER_OF_TEXTURECOORDS; uvLayer++)
		removeFlags |= aiComponent_TEXCOORDSn(uvLayer);

	mFilepath = filepath;
	mpScene	  = pScene;

	Assimp::Importer importer;
	importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeFlags);
	importer.SetPropertyFloat(AI_CONFIG_PP_GSN_MAX_SMOOTHING_ANGLE, 60);

	logDebug("Start loading scene with assimp importer");
	mpAiScene = (aiScene *) importer.ReadFile(filepath, postProcessSteps);
	if (!mpAiScene) {
		logError("Assimp::load model failed");
		Assimp::DefaultLogger::get()->info(importer.GetErrorString());
		Assimp::DefaultLogger::kill(); // Kill it after the work is done
		return false;
	}

	logDebug("Start loading materials");
	string modelFolder = getFileDir(filepath);
	string modelSuffix = getFileExt(filepath);
	if (modelSuffix == ".obj") {
		mImportMode = ImportMode::OBJ;
		logInfo("Importing OBJ model, using Specular Glossiness shading method.");
	} else if (modelSuffix == ".gltf" || modelSuffix == ".glb") {
		mImportMode = ImportMode::GLTF2;
		logInfo("Importing GLTF2 model.");
	}
	loadMaterials(modelFolder);
	loadMeshes();

	logDebug("Start traversing scene nodes");
	traverseNode(mpAiScene->mRootNode);
	logDebug("Total imported meshes: " + std::to_string(mpScene->meshes.size()));

	Assimp::DefaultLogger::kill();
	return true;
}

void SceneGraphImporter::loadMeshes() {
	for (int i = 0; i < mpAiScene->mNumMeshes; i++) {
		const aiMesh *pAiMesh = mpAiScene->mMeshes[i];

		unsigned int remapMeshToGeometry =
			~0u; // Remap mesh index to geometry index. ~0 means there was no geometry for a mesh.

		// The post-processor took care of meshes per primitive type and triangulation.
		// Need to do a bitwise comparison of the mPrimitiveTypes here because newer ASSIMP versions
		// indicate triangulated former polygons with the additional
		// aiPrimitiveType_NGONEncodingFlag.
		if ((pAiMesh->mPrimitiveTypes & aiPrimitiveType_TRIANGLE) && 2 < pAiMesh->mNumVertices) {

			remapMeshToGeometry = static_cast<unsigned int>(mpScene->mSceneMeshes.size());
			mRemappedMeshIndices.push_back(remapMeshToGeometry);
			SceneMesh::SharedPtr mesh(new SceneMesh(sIdMesh++));
			mpScene->mSceneMeshes.push_back(mesh);

			inter::vector<VertexAttribute> &attributes = mesh->vertices;
			inter::vector<Vector3i> &indices		   = mesh->indices;
			attributes.reserve(pAiMesh->mNumVertices);
			indices.reserve(pAiMesh->mNumFaces);

			for (unsigned int iVertex = 0; iVertex < pAiMesh->mNumVertices; ++iVertex) {
				VertexAttribute &attrib = attributes[iVertex];
				attrib.vertex			= aiCast(pAiMesh->mVertices[iVertex]);

				if (pAiMesh->HasNormals()) {
					attrib.normal = aiCast(pAiMesh->mNormals[iVertex]);
				} else {
					attrib.normal = Vector3f(0.0f, 0.0f, 1.0f);
				}

				if (pAiMesh->HasTangentsAndBitangents()) {
					attrib.tangent = aiCast(pAiMesh->mTangents[iVertex]);
				} else {
					attrib.tangent = getPerpendicular(attrib.normal);
				}
				attrib.bitangent = normalize(cross(attrib.normal, attrib.tangent));

				if (pAiMesh->HasTextureCoords(0)) {
					Vector3f texcoord = aiCast(pAiMesh->mTextureCoords[0][iVertex]);
					attrib.texcoord = texcoord;
				} else {
					attrib.texcoord = Vector2f(0, 0);
				}
			}

			for (unsigned int iFace = 0; iFace < pAiMesh->mNumFaces; ++iFace) {
				const struct aiFace &face = pAiMesh->mFaces[iFace];
				assert(face->mNumIndices == 3); // Must be true because of aiProcess_Triangulate.
				Vector3i index = { (int) face.mIndices[0], (int) face.mIndices[1],
								   (int) face.mIndices[2] };
				indices.push_back(index);
			}
		}
	}
}

SceneGroup::SharedPtr SceneGraphImporter::traverseNode(aiNode *node) {
	SceneGroup::SharedPtr group(new SceneGroup(sIdGroup++));
	Matrixf<4, 4> transform = aiCast(node->mTransformation);
	
	for (unsigned int iChild = 0; iChild < node->mNumChildren; ++iChild) {
		SceneGroup::SharedPtr child = traverseNode(node->mChildren[iChild]);

		// Create an instance which holds the subtree.
		SceneInstance::SharedPtr instance(new SceneInstance(sIdInstance++));

		instance->setTransform(transform);
		instance->setChild(child);

		group->addChild(instance);
	}

	for (uint iMesh = 0; iMesh < node->mNumMeshes; ++iMesh) {
		const uint indexMesh = node->mMeshes[iMesh]; 
		assert(indexMesh < mRemappedMeshIndices.size());

		if (mRemappedMeshIndices[indexMesh] != ~0u) // If there exists a Triangles geometry for
													 // this assimp mesh, then build the Instance.
		{
			const unsigned int indexGeometry = mRemappedMeshIndices[indexMesh];

			// Create an instance with the current nodes transformation and append it to the parent
			// group.
			SceneInstance::SharedPtr instance(new SceneInstance(sIdInstance++));

			const struct aiMesh *mesh = mpAiScene->mMeshes[indexMesh];

			instance->setTransform(transform);
			instance->setChild(mpScene->mSceneMeshes[indexGeometry]);
			instance->setMaterial(mesh->mMaterialIndex);


			group->addChild(instance);
		}
	}

	return group;
}

void SceneGraphImporter::loadMaterials(const string &modelFolder) {
	mpScene->mData.materials->reserve(mpAiScene->mNumMaterials + 1LL);
	mpScene->mData.materials->push_back(Material());	// the default material, if object has no material
	for (uint i = 0; i < mpAiScene->mNumMaterials; i++) {
		const aiMaterial *aiMaterial  = mpAiScene->mMaterials[i];
		Material::SharedPtr pMaterial = createMaterial(aiMaterial, modelFolder, mImportMode);
		if (pMaterial == nullptr) {
			logError("Failed to create material...");
			return;
		}
		pMaterial->toDevice();
		mpScene->mData.materials->push_back(*pMaterial);
	}
}

KRR_NAMESPACE_END