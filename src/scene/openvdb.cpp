#include "importer.h"
#include "util/volume.h"

KRR_NAMESPACE_BEGIN
namespace importer {

/* Construct an axis-aligned bounding box with null-material as the container of the volume. */
bool OpenVDBImporter::import(const fs::path filepath, Scene::SharedPtr scene) {
	mFilepath = filepath.string();
	mScene	  = scene;
	
	auto mesh	  = std::make_shared<Mesh>();
	auto instance = std::make_shared<MeshInstance>(mesh);
	auto root	  = scene->getSceneGraph()->getRoot();
	auto volume	  = std::make_shared<VDBVolume>(0.5, 0.5, 1, filepath);

	scene->getSceneGraph()->attachLeaf(root, instance);
	scene->getSceneGraph()->attachLeaf(root, volume);
	return true; 
}

} // namespace importer
KRR_NAMESPACE_END