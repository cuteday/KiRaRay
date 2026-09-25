#include "light.h"
#include "window.h"

#include <cmath>

NAMESPACE_BEGIN(krr)

namespace rt {
void PointLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {
	auto light = std::dynamic_pointer_cast<krr::PointLight>(object);
	auto gdata = reinterpret_cast<rt::PointLight*>(data->data());
	*gdata = rt::PointLight(light->getPosition(), light->getColor(), light->getScale());
}

void DirectionalLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data,
									 bool initialize) const {
	auto light = std::dynamic_pointer_cast<krr::DirectionalLight>(object);
	auto gdata = reinterpret_cast<rt::DirectionalLight *>(data->data());
	auto transform = light->getNode()->getGlobalTransform();
	float sceneRadius =
		light->getNode()->getGraph()->getRoot()->getGlobalBoundingBox().diagonal().norm();
	*gdata = rt::DirectionalLight(transform.rotation(), light->getColor(), light->getScale(),
								 sceneRadius);
}

void InfiniteLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {
	auto light = std::dynamic_pointer_cast<krr::InfiniteLight>(object);
	auto gdata = reinterpret_cast<rt::InfiniteLight *>(data->data());
	auto transform = light->getNode()->getGlobalTransform();
	float sceneRadius =
		light->getNode()->getGraph()->getRoot()->getGlobalBoundingBox().diagonal().norm();
	if (initialize) {
		gdata->tint = RGB::Ones();
		gdata->sceneRadius = sceneRadius;
		gdata->colorSpace = KRR_DEFAULT_COLORSPACE;
		gdata->image.initializeFromHost(light->getTexture());
	}
	gdata->rotation = transform.rotation();
	gdata->scale	= light->getScale();
}

void SpotLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data,
							  bool initialize) const {
	auto light	   = std::dynamic_pointer_cast<krr::SpotLight>(object);
	auto gdata	   = reinterpret_cast<rt::SpotLight *>(data->data());
	auto transform = light->getNode()->getGlobalTransform();
	*gdata = rt::SpotLight(transform, light->getColor(), light->getScale(),
						  light->getInnerConeAngle(), light->getOuterConeAngle());
}
void DiffuseAreaLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {
	Log(Error,
		"Currently diffuse area lights (mesh lights) should not be updated as managed objects");
}
}

NAMESPACE_END(krr)
