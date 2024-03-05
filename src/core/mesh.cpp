#pragma once
#include "mesh.h"
#include "shape.h"
#include "window.h"
#include "scenegraph.h"

NAMESPACE_BEGIN(krr)

AABB Mesh::computeBoundingBox() {
	aabb = {};
	for (const auto &v : positions) 
		aabb.extend(v);
	return aabb;
}

void rt::InstanceData::getObjectData(std::shared_ptr<SceneGraphLeaf> object,
						std::shared_ptr<Blob> data, bool initialize) const {
	auto inst  = std::dynamic_pointer_cast<MeshInstance>(object);
	auto gdata = reinterpret_cast<rt::InstanceData *>(data->data());
	if (initialize) {
		auto &meshData = inst->getNode()->getGraph()->getScene()->getSceneRT()->getMeshData();
		gdata->mesh	   = &meshData[inst->getMesh()->getMeshId()];	
	}
	gdata->transform = inst->getNode()->getGlobalTransform();
	
}

void Mesh::renderUI() {
	ui::Text("%zd vertices", positions.size());
	ui::Text("%zd faces", indices.size());
	if (texcoords.size()) ui::Text("%zd UV coordinates", texcoords.size());
	if (tangents.size()) ui::Text("%zd tangents", tangents.size());
	if (normals.size()) ui::Text("%zd normals", normals.size());
	ui::BulletText("Local bounding box");
	ui::Text(("Min: " + aabb.min().string()).c_str());
	ui::Text(("Max: " + aabb.max().string()).c_str());
	if (material) {
		ui::BulletText("Material:");
		if (ui::TreeNode(material->getName().c_str())) {
			material->renderUI();
			ui::TreePop();
		}
	}
	if (inside || outside) {
		ui::BulletText("Medium Interface:");
		if (inside && ui::TreeNode(("Inside: " + inside->getName()).c_str())) {
			inside->renderUI();
			ui::TreePop();
		}
		if (outside && ui::TreeNode(("Outside: " + outside->getName()).c_str())) {
			outside->renderUI();
			ui::TreePop();
		}
	}
}

NAMESPACE_END(krr)