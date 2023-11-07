#include "importer.h"
#include "util/volume.h"

KRR_NAMESPACE_BEGIN
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
	
	auto sigma_a = params.value<Array3f>("sigma_a", Array3f::Ones());
	auto sigma_s = params.value<Array3f>("sigma_s", Array3f::Zero());
	float g		 = params.value<float>("g", 0);

	auto mesh	  = std::make_shared<Mesh>();
	auto instance = std::make_shared<MeshInstance>(mesh);
	auto volume	  = std::make_shared<VDBVolume>(sigma_a, sigma_s, g, filepath);

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
KRR_NAMESPACE_END