#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/pbrmaterial.h"

#include "importer.h"

KRR_NAMESPACE_BEGIN

namespace assimp
{
	static const bool kCameraEnableWarping = true;

	using BoneMeshMap = std::map<std::string, std::vector<uint>>;
	using MeshInstanceList = std::vector<std::vector<const aiNode*>>;

	/** Converts specular power to roughness. Note there is no "the conversion".
		Reference: http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
		\param specPower specular power of an obsolete Phong BSDF
	*/
	float convertSpecPowerToRoughness(float specPower)
	{
		return clamp(sqrt(2.0f / (specPower + 2.0f)), 0.f, 1.f);
	}

	enum class ImportMode {
		Default,
		OBJ,
		GLTF2,
	};

	vec3f aiCast(const aiColor3D& ai) { return vec3f(ai.r, ai.g, ai.b); }
	vec3f aiCast(const aiVector3D& val) { return vec3f(val.x, val.y, val.z); }
	quat aiCast(const aiQuaternion& q) { return quat(q.w, q.x, q.y, q.z); }

	/** Mapping from ASSIMP to Falcor texture type.
	*/
	struct TextureMapping
	{
		aiTextureType aiType;
		unsigned int aiIndex;
		Material::TextureType targetType;
	};

	/** Mapping tables for different import modes.
	*/
	static const std::vector<TextureMapping> kTextureMappings[3] =
	{
		// Default mappings
		{
			{ aiTextureType_DIFFUSE, 0, Material::TextureType::Diffuse },
			{ aiTextureType_SPECULAR, 0, Material::TextureType::Specular },
			{ aiTextureType_EMISSIVE, 0, Material::TextureType::Emissive },
			{ aiTextureType_NORMALS, 0, Material::TextureType::Normal },
		},
		// OBJ mappings
		{
			{ aiTextureType_DIFFUSE, 0, Material::TextureType::Diffuse },
			{ aiTextureType_SPECULAR, 0, Material::TextureType::Specular },
			{ aiTextureType_EMISSIVE, 0, Material::TextureType::Emissive },
			// OBJ does not offer a normal map, thus we use the bump map instead.
			{ aiTextureType_HEIGHT, 0, Material::TextureType::Normal },
			{ aiTextureType_DISPLACEMENT, 0, Material::TextureType::Normal },
		},
		// GLTF2 mappings
		{
			{ aiTextureType_DIFFUSE, 0, Material::TextureType::Diffuse },
			{ aiTextureType_EMISSIVE, 0, Material::TextureType::Emissive },
			{ aiTextureType_NORMALS, 0, Material::TextureType::Normal },
			// GLTF2 exposes metallic roughness texture.
			{ AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, Material::TextureType::Specular },
		}
	};

	void createTexCrdList(const aiVector3D* pAiTexCrd, uint count, std::vector<vec2f>& texCrds)
	{
		texCrds.resize(count);
		for (uint i = 0; i < count; i++)
		{
			assert(pAiTexCrd[i].z == 0);
			texCrds[i] = vec2f(pAiTexCrd[i].x, pAiTexCrd[i].y);
		}
	}

	void createTangentList(const aiVector3D* pAiTangent, const aiVector3D* pAiBitangent, const aiVector3D* pAiNormal, uint count, std::vector<vec4f>& tangents)
	{
		tangents.resize(count);
		for (uint i = 0; i < count; i++)
		{
			// We compute the bitangent at runtime as defined by MikkTSpace: cross(N, tangent.xyz) * tangent.w.
			// Compute the orientation of the loaded bitangent here to set the sign (w) correctly.
			vec3f T = vec3f(pAiTangent[i].x, pAiTangent[i].y, pAiTangent[i].z);
			vec3f B = vec3f(pAiBitangent[i].x, pAiBitangent[i].y, pAiBitangent[i].z);
			vec3f N = vec3f(pAiNormal[i].x, pAiNormal[i].y, pAiNormal[i].z);
			float sign = dot(cross(N, T), B) >= 0.f ? 1.f : -1.f;
			tangents[i] = vec4f(glm::normalize(T), sign);
		}
	}

	void createIndexList(const aiMesh* pAiMesh, std::vector<uint>& indices)
	{
		const uint perFaceIndexCount = pAiMesh->mFaces[0].mNumIndices;
		const uint indexCount = pAiMesh->mNumFaces * perFaceIndexCount;

		indices.resize(indexCount);
		for (uint i = 0; i < pAiMesh->mNumFaces; i++)
		{
			assert(pAiMesh->mFaces[i].mNumIndices == perFaceIndexCount); // Mesh contains mixed primitive types, can be solved using aiProcess_SortByPType
			for (uint j = 0; j < perFaceIndexCount; j++)
			{
				indices[i * perFaceIndexCount + j] = (uint)(pAiMesh->mFaces[i].mIndices[j]);
			}
		}
	}

	void createMeshes(const aiScene* pScene)
	{
		const bool loadTangents = true;

		uint meshCount = pScene->mNumMeshes;

		// Pre-process meshes.
		std::vector<SceneBuilder::ProcessedMesh> processedMeshes(meshCount);
		auto range = NumericRange<uint>(0, meshCount);
		std::for_each(std::execution::par, range.begin(), range.end(), [&](uint i) {
			const aiMesh* pAiMesh = pScene->mMeshes[i];
			const uint perFaceIndexCount = pAiMesh->mFaces[0].mNumIndices;

			SceneBuilder::Mesh mesh;
			mesh.name = pAiMesh->mName.C_Str();
			mesh.faceCount = pAiMesh->mNumFaces;

			// Temporary memory for the vertex and index data.
			std::vector<uint> indexList;
			std::vector<vec2f> texCrds;
			std::vector<vec4f> tangents;

			// Indices
			createIndexList(pAiMesh, indexList);
			assert(indexList.size() <= std::numeric_limits<uint>::max());
			mesh.indexCount = (uint)indexList.size();
			mesh.pIndices = indexList.data();

			// Vertices
			assert(pAiMesh->mVertices);
			mesh.vertexCount = pAiMesh->mNumVertices;
			static_assert(sizeof(pAiMesh->mVertices[0]) == sizeof(mesh.positions.pData[0]));
			static_assert(sizeof(pAiMesh->mNormals[0]) == sizeof(mesh.normals.pData[0]));
			mesh.positions.pData = reinterpret_cast<vec3f*>(pAiMesh->mVertices);
			mesh.positions.frequency = SceneBuilder::Mesh::AttributeFrequency::Vertex;
			mesh.normals.pData = reinterpret_cast<vec3f*>(pAiMesh->mNormals);
			mesh.normals.frequency = SceneBuilder::Mesh::AttributeFrequency::Vertex;

			if (pAiMesh->HasTextureCoords(0))
			{
				createTexCrdList(pAiMesh->mTextureCoords[0], pAiMesh->mNumVertices, texCrds);
				assert(!texCrds.empty());
				mesh.texCrds.pData = texCrds.data();
				mesh.texCrds.frequency = SceneBuilder::Mesh::AttributeFrequency::Vertex;
			}

			if (loadTangents && pAiMesh->HasTangentsAndBitangents())
			{
				createTangentList(pAiMesh->mTangents, pAiMesh->mBitangents, pAiMesh->mNormals, pAiMesh->mNumVertices, tangents);
				assert(!tangents.empty());
				mesh.tangents.pData = tangents.data();
				mesh.tangents.frequency = SceneBuilder::Mesh::AttributeFrequency::Vertex;
			}

			if (pAiMesh->HasBones())
			{
				loadBones(pAiMesh, data, boneWeights, boneIds);
				mesh.boneIDs.pData = boneIds.data();
				mesh.boneIDs.frequency = SceneBuilder::Mesh::AttributeFrequency::Vertex;
				mesh.boneWeights.pData = boneWeights.data();
				mesh.boneWeights.frequency = SceneBuilder::Mesh::AttributeFrequency::Vertex;
			}

			switch (perFaceIndexCount)
			{
			case 1: mesh.topology = Vao::Topology::PointList; break;
			case 2: mesh.topology = Vao::Topology::LineList; break;
			case 3: mesh.topology = Vao::Topology::TriangleList; break;
			default:
				logError("Error when creating mesh. Unknown topology with " + std::to_string(perFaceIndexCount) + " indices per face.");
				should_not_get_here();
			}

			mesh.pMaterial = data.materialMap.at(pAiMesh->mMaterialIndex);

			processedMeshes[i] = data.builder.processMesh(mesh);
			});

		// Add meshes to the scene.
		// We retain a deterministic order of the meshes in the global scene buffer by adding
		// them sequentially after being processed in parallel.
		uint i = 0;
		for (const auto& mesh : processedMeshes)
		{
			uint meshID = data.builder.addProcessedMesh(mesh);
			data.meshMap[i++] = meshID;
		}
	}

	bool createSceneGraph(aiScene* pScene)
	{
		aiNode* pRoot = pScene->mRootNode;
		bool success = parseNode(data, pRoot, false);
		return success;
	}

	void addMeshInstances(ImporterData& data, aiNode* pNode)
	{
		uint nodeID = data.getFalcorNodeID(pNode);
		for (uint mesh = 0; mesh < pNode->mNumMeshes; mesh++)
		{
			uint meshID = data.meshMap.at(pNode->mMeshes[mesh]);

			if (data.modelInstances.size())
			{
				for (size_t instance = 0; instance < data.modelInstances.size(); instance++)
				{
					uint instanceNodeID = nodeID;
					if (data.modelInstances[instance] != glm::mat4())
					{
						// Add nodes
						SceneBuilder::Node n;
						n.name = "Node" + std::to_string(nodeID) + ".instance" + std::to_string(instance);
						n.parent = nodeID;
						n.transform = data.modelInstances[instance];
						instanceNodeID = data.builder.addNode(n);
					}
					data.builder.addMeshInstance(instanceNodeID, meshID);
				}
			}
			else data.builder.addMeshInstance(nodeID, meshID);
		}

		// Visit the children
		for (uint i = 0; i < pNode->mNumChildren; i++)
		{
			addMeshInstances(data, pNode->mChildren[i]);
		}
	}

	void loadTextures(ImporterData& data, const aiMaterial* pAiMaterial, const std::string& folder, const Material::SharedPtr& pMaterial, ImportMode importMode)
	{
		const auto& textureMappings = kTextureMappings[int(importMode)];

		for (const auto& source : textureMappings)
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
			std::string filename = canonicalizeFilename(folder + '/' + path);
			data.builder.loadMaterialTexture(pMaterial, source.targetType, filename);
		}
	}

	Material::SharedPtr createMaterial(ImporterData& data, const aiMaterial* pAiMaterial, const std::string& folder, ImportMode importMode)
	{
		aiString name;
		pAiMaterial->Get(AI_MATKEY_NAME, name);

		// Parse the name
		std::string nameStr = std::string(name.C_Str());
		if (nameStr.empty())
		{
			logWarning("Material with no name found -> renaming to 'unnamed'");
			nameStr = "unnamed";
		}
		Material::SharedPtr pMaterial = Material::create(nameStr);

		// Determine shading model.
		// MetalRough is the default for everything except OBJ. Check that both flags aren't set simultaneously.
		SceneBuilder::Flags builderFlags = data.builder.getFlags();
		assert(!(is_set(builderFlags, SceneBuilder::Flags::UseSpecGlossMaterials) && is_set(builderFlags, SceneBuilder::Flags::UseMetalRoughMaterials)));
		if (is_set(builderFlags, SceneBuilder::Flags::UseSpecGlossMaterials) || (importMode == ImportMode::OBJ && !is_set(builderFlags, SceneBuilder::Flags::UseMetalRoughMaterials)))
		{
			pMaterial->setShadingModel(ShadingModelSpecGloss);
		}

		// Load textures. Note that loading is affected by the current shading model.
		loadTextures(data, pAiMaterial, folder, pMaterial, importMode);

		// Opacity
		float opacity = 1.f;
		if (pAiMaterial->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
		{
			vec4f diffuse = pMaterial->getDiffuse();
			diffuse.a = opacity;
			pMaterial->setDiffuse(diffuse);
		}

		// Bump scaling
		float bumpScaling;
		if (pAiMaterial->Get(AI_MATKEY_BUMPSCALING, bumpScaling) == AI_SUCCESS)
		{
			// TODO this should probably be a multiplier to the normal map
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
			vec4f spec = pMaterial->getSpecularParams();
			spec.a = shininess;
			pMaterial->setSpecularParams(spec);
		}

		// Refraction
		float refraction;
		if (pAiMaterial->Get(AI_MATKEY_REFRACTI, refraction) == AI_SUCCESS) pMaterial->setIndexOfRefraction(refraction);

		// Diffuse color
		aiColor3D color;
		if (pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
		{
			vec4f diffuse = vec4f(color.r, color.g, color.b, pMaterial->getDiffuse().a);
			pMaterial->setDiffuse(diffuse);
		}

		// Specular color
		if (pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
		{
			vec4f specular = vec4f(color.r, color.g, color.b, pMaterial->getSpecularParams().a);
			pMaterial->setSpecularParams(specular);
		}

		// Emissive color
		if (pAiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS)
		{
			vec3f emissive = vec3f(color.r, color.g, color.b);
			pMaterial->setEmissiveColor(emissive);
		}

		// Double-Sided
		int isDoubleSided;
		if (pAiMaterial->Get(AI_MATKEY_TWOSIDED, isDoubleSided) == AI_SUCCESS)
		{
			pMaterial->setDoubleSided((isDoubleSided != 0));
		}

		// Handle GLTF2 PBR materials
		if (importMode == ImportMode::GLTF2)
		{
			if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR, color) == AI_SUCCESS)
			{
				vec4f Diffuse = vec4f(color.r, color.g, color.b, pMaterial->getDiffuse().a);
				pMaterial->setDiffuse(Diffuse);
			}

			vec4f specularParams = pMaterial->getSpecularParams();

			float metallic;
			if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, metallic) == AI_SUCCESS)
			{
				specularParams.b = metallic;
			}

			float roughness;
			if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
			{
				specularParams.g = roughness;
			}

			pMaterial->setSpecularParams(specularParams);
		}

		// Parse the information contained in the name
		// Tokens following a '.' are interpreted as special flags
		auto nameVec = splitString(nameStr, ".");
		if (nameVec.size() > 1)
		{
			for (size_t i = 1; i < nameVec.size(); i++)
			{
				std::string str = nameVec[i];
				std::transform(str.begin(), str.end(), str.begin(), ::tolower);
				if (str == "doublesided") pMaterial->setDoubleSided(true);
				else logWarning("Unknown material property found in the material's name - '" + nameVec[i] + "'");
			}
		}

		// Use scalar opacity value for controlling specular transmission
		// TODO: Remove this workaround when we have a better way to define materials.
		if (opacity < 1.f)
		{
			pMaterial->setSpecularTransmission(1.f - opacity);
		}

		return pMaterial;
	}

	bool createAllMaterials(ImporterData& data, const std::string& modelFolder, ImportMode importMode)
	{
		for (uint i = 0; i < data.pScene->mNumMaterials; i++)
		{
			const aiMaterial* pAiMaterial = data.pScene->mMaterials[i];
			auto pMaterial = createMaterial(data, pAiMaterial, modelFolder, importMode);
			if (pMaterial == nullptr)
			{
				logError("Can't allocate memory for material");
				return false;
			}
			data.materialMap[i] = pMaterial;
		}

		return true;
	}

	BoneMeshMap createBoneMap(const aiScene* pScene)
	{
		BoneMeshMap boneMap;

		for (uint meshID = 0; meshID < pScene->mNumMeshes; meshID++)
		{
			const aiMesh* pMesh = pScene->mMeshes[meshID];
			for (uint boneID = 0; boneID < pMesh->mNumBones; boneID++)
			{
				boneMap[pMesh->mBones[boneID]->mName.C_Str()].push_back(meshID);
			}
		}

		return boneMap;
	}

	MeshInstanceList countMeshInstances(const aiScene* pScene)
	{
		MeshInstanceList meshInstances(pScene->mNumMeshes);

		std::function<void(const aiNode*)> countNodeMeshs = [&](const aiNode* pNode)
		{
			for (uint i = 0; i < pNode->mNumMeshes; i++)
			{
				meshInstances[pNode->mMeshes[i]].push_back(pNode);
			}

			for (uint i = 0; i < pNode->mNumChildren; i++)
			{
				countNodeMeshs(pNode->mChildren[i]);
			}
		};
		countNodeMeshs(pScene->mRootNode);

		return meshInstances;
	}

	void verifyScene(const aiScene* pScene)
	{
		bool b = true;

		// No internal textures
		if (pScene->mTextures != 0)
		{
			b = false;
			logWarning("Model has internal textures which Falcor doesn't support");
		}

		b = validateBones(pScene);
		assert(b);
	}
}

using namespace krr::assimp;
bool AssimpImporter::import(const std::string& filename)
{
	std::string fullpath;

	uint assimpFlags = aiProcessPreset_TargetRealtime_MaxQuality |
		aiProcess_FlipUVs |
		aiProcess_RemoveComponent;

	assimpFlags &= ~(aiProcess_CalcTangentSpace); // Never use Assimp's tangent gen code
	assimpFlags &= ~(aiProcess_FindDegenerates); // Avoid converting degenerated triangles to lines
	assimpFlags &= ~(aiProcess_OptimizeGraph); // Never use as it doesn't handle transforms with negative determinants
	assimpFlags &= ~(aiProcess_RemoveRedundantMaterials); // Avoid merging materials as it doesn't load all fields we care about, we merge in 'SceneBuilder' instead.
	assimpFlags &= ~(aiProcess_SplitLargeMeshes); // Avoid splitting large meshes

	// Configure importer to remove vertex components we don't support.
	// It'll load faster and helps 'aiProcess_JoinIdenticalVertices' find identical vertices.
	int removeFlags = aiComponent_COLORS;
	for (uint uvLayer = 1; uvLayer < AI_MAX_NUMBER_OF_TEXTURECOORDS; uvLayer++) 
		removeFlags |= aiComponent_TEXCOORDSn(uvLayer);

	Assimp::Importer importer;
	importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, removeFlags);

	const aiScene* pScene = importer.ReadFile(fullpath, assimpFlags);

	if (pScene == nullptr)
	{
		std::string str("Can't open file '");
		str = str + std::string(filename) + "'\n" + importer.GetErrorString();
		logError(str);
		return false;
	}

	verifyScene(pScene);

	// Extract the folder name
	auto last = fullpath.find_last_of("/\\");
	std::string modelFolder = fullpath.substr(0, last);

	ImporterData data(pScene, builder, instances);

	// Enable special treatment for obj and gltf files
	ImportMode importMode = ImportMode::Default;
	if (hasSuffix(filename, ".obj", false)) importMode = ImportMode::OBJ;
	if (hasSuffix(filename, ".gltf", false) || hasSuffix(filename, ".glb", false)) importMode = ImportMode::GLTF2;

	if (createAllMaterials(data, modelFolder, importMode) == false)
	{
		logError("Can't create materials for model " + filename);
		return false;
	}

	if (createSceneGraph(data) == false)
	{
		logError("Can't create draw lists for model " + filename);
		return false;
	}

	createMeshes(data);
	addMeshInstances(data, data.pScene->mRootNode);


	return true;
}

KRR_NAMESPACE_END
