#include "importer.h"
#include "util/volume.h"

NAMESPACE_BEGIN(krr)
namespace importer {

/* Construct an axis-aligned bounding box with null-material as the container of the volume. */
bool OpenVDBImporter::import(const fs::path filepath, Scene::SharedPtr scene,
							 SceneGraphNode::SharedPtr node, const json &params) {
	mFilepath = filepath.string();
	mScene	  = scene;
	
	node = node ? node : scene->getSceneGraph()->getRoot();
	if (!node) {
		node = std::make_shared<SceneGraphNode>();
		scene->getSceneGraph()->setRoot(node);
	}
	
	auto sigma_t			= params.value<Array3f>("sigma_t", Array3f{1, 1, 1});
	auto albedo				= params.value<Array3f>("albedo", Array3f{0.5, 0.5, 0.5});
	auto key_density		= params.value<string>("key_density", "density");
	auto key_temperature	= params.value<string>("key_temperature", "temperature");
	auto key_albedo			= params.value<string>("key_albedo", "albedo");
	float g					= params.value<float>("g", 0);
	float temperaetureScale = params.value<float>("scale_temperature", 1);
	float temperatureOffset = params.value<float>("offset_temperature", 0);
	float scale				= params.value<float>("scale", 1);
	float LeScale			= params.value<float>("scale_le", 1);
	if (params.contains("sigma_a") || params.contains("sigma_s")) {
		auto sigma_a = params.value<Array3f>("sigma_a", Array3f{0.5, 0.5, 0.5});
		auto sigma_s = params.value<Array3f>("sigma_s", Array3f{0.5, 0.5, 0.5});
		sigma_t		 = sigma_a + sigma_s;
		albedo		 = sigma_s / sigma_t;
	}
	auto densityGrid	 = loadOpenVDB(filepath, key_density);
	auto temperatureGrid = std::dynamic_pointer_cast<NanoVDBGrid<float>>(loadOpenVDB(filepath, key_temperature));
	auto albedoGrid      = std::dynamic_pointer_cast<NanoVDBGrid<Array3f>>(loadOpenVDB(filepath, key_albedo));
	auto volume = std::make_shared<VDBVolume>(sigma_t, albedo, g, densityGrid, temperatureGrid, albedoGrid,
											  scale, LeScale, temperaetureScale, temperatureOffset);

	auto mesh	  = std::make_shared<Mesh>();
	auto instance = std::make_shared<MeshInstance>(mesh);
	/* initialize a intersection bounding box for this volume */
	auto aabb	  = volume->densityGrid->getBounds();
	mesh->indices = {{4, 2, 0}, {2, 7, 3}, {6, 5, 7}, {1, 7, 5}, {0, 3, 1}, {4, 1, 5}, 
		{4, 6, 2}, {2, 6, 7}, {6, 4, 5}, {1, 3, 7}, {0, 2, 3}, {4, 0, 1}};
	mesh->positions = {
		{aabb.max()[0], aabb.max()[1], aabb.min()[2]},
		{aabb.max()[0], aabb.min()[1], aabb.min()[2]},
		{aabb.max()[0], aabb.max()[1], aabb.max()[2]},
		{aabb.max()[0], aabb.min()[1], aabb.max()[2]},
		{aabb.min()[0], aabb.max()[1], aabb.min()[2]},
		{aabb.min()[0], aabb.min()[1], aabb.min()[2]},
		{aabb.min()[0], aabb.max()[1], aabb.max()[2]},
		{aabb.min()[0], aabb.min()[1], aabb.max()[2]},
	};

	mesh->setName("Medium box-" + filepath.filename().string());
	mesh->setMaterial(nullptr);
	mesh->setMedium(volume, nullptr);
	mesh->computeBoundingBox();
	scene->addMesh(mesh);
	scene->getSceneGraph()->attachLeaf(node, instance);
	scene->getSceneGraph()->attachLeaf(node, volume);
	return true; 
}

} // namespace importer
NAMESPACE_END(krr)