#include "importer.h"
#include "util/volume.h"

KRR_NAMESPACE_BEGIN
namespace importer {

/* Construct an axis-aligned bounding box with null-material as the container of the volume. */
bool OpenVDBImporter::import(const fs::path filepath, Scene::SharedPtr scene) {
	mFilepath = filepath.string();
	mScene	  = scene;
	
	if (!scene->getSceneGraph()->getRoot())
		scene->getSceneGraph()->setRoot(std::make_shared<SceneGraphNode>());

	auto mesh	  = std::make_shared<Mesh>();
	auto instance = std::make_shared<MeshInstance>(mesh);
	auto root	  = scene->getSceneGraph()->getRoot();
	auto volume	  = std::make_shared<VDBVolume>(10, 10, 0, filepath);

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
	scene->getSceneGraph()->attachLeaf(root, instance);
	scene->getSceneGraph()->attachLeaf(root, volume);
	return true; 
}

} // namespace importer
KRR_NAMESPACE_END