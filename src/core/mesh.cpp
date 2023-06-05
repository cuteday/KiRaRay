#pragma once
#include "mesh.h"
#include "shape.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

AABB Mesh::computeBoundingBox() {
	aabb = {};
	for (const auto &v : positions) 
		aabb.extend(v);
	return aabb;
}

void Mesh::renderUI() {
	ui::Text("%zd vertices", positions.size());
	ui::Text("%zd faces", indices.size());
	if (texcoords.size()) ui::Text("%zd UV coordinates", texcoords.size());
	if (tangents.size()) ui::Text("%zd tangents", tangents.size());
	if (normals.size()) ui::Text("%zd normals", normals.size());
	if (material) {
		ui::Text("Material:");
		if (ui::TreeNode(material->getName().c_str())) {
			material->renderUI();
			ui::TreePop();
		}
	}
}

KRR_NAMESPACE_END