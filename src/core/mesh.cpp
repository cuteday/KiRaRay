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
	auto createTrianglePrimitives = [](Mesh::SharedPtr mesh,
									   rt::InstanceData *instance) -> std::vector<Triangle> {
		uint nTriangles = mesh->indices.size();
		std::vector<Triangle> triangles;
		for (uint i = 0; i < nTriangles; i++) triangles.push_back(Triangle(i, instance));
		return triangles;
	};

	auto inst  = std::dynamic_pointer_cast<MeshInstance>(object);
	auto gdata = reinterpret_cast<rt::InstanceData *>(data->data());
	if (initialize) {
		new (gdata) rt::InstanceData();
		auto scene		   = inst->getNode()->getGraph()->getScene()->getSceneRT();
		auto &meshes	   = scene->getMeshData();
		auto &instances	   = scene->getInstanceData();
		auto &materials	   = scene->getMaterialData();
		gdata->mesh		   = &meshes[inst->getMesh()->getMeshId()];	
		/* process mesh lights for light sampling purposes */
		const auto &mesh	 = inst->getMesh();
		const auto &material = mesh->getMaterial();

		if ((material && material->hasEmission()) || mesh->Le.any()) {
			rt::MaterialData &materialData = materials[material->getMaterialId()];
			rt::MeshData &meshData		   = meshes[mesh->getMeshId()];
			rt::TextureData &textureData = materialData.getTexture(Material::TextureType::Emissive);
			rt::InstanceData &instanceData = instances[inst->getInstanceId()];
			RGB Le = material->hasEmission() ? RGB(textureData.getConstant()) : mesh->Le;
			Log(Debug, "Emissive diffuse area light detected, number of shapes: %lld",
				" constant emission(?): %f", mesh->indices.size(), luminance(Le));
			float scale = Le.maxCoeff();
			Le /= scale;
			std::vector<Triangle> primitives = createTrianglePrimitives(mesh, const_cast<rt::InstanceData*>(this));
			size_t n_primitives = primitives.size();
			gdata->primitives.alloc_and_copy_from_host(primitives);
			std::vector<rt::DiffuseAreaLight> lights(n_primitives);
			Log(Info, "Uploading a light with scale %f; emission %s", scale, Le.string().c_str());
			for (size_t triId = 0; triId < n_primitives; triId++) {
				lights[triId] =
					rt::DiffuseAreaLight(Shape(&gdata->primitives[triId]), textureData, Le, false, scale);
			}
			gdata->lights.alloc_and_copy_from_host(lights);
		}
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