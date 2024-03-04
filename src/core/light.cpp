#include "light.h"
#include "window.h"

#include <cmath>

NAMESPACE_BEGIN(krr)

namespace rt {
void PointLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {}

void DirectionalLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {}

void InfiniteLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {}

void SpotLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {}

void DiffuseAreaLight::getObjectData(SceneGraphLeaf::SharedPtr object, Blob::SharedPtr data, bool initialize) const {}
}

NAMESPACE_END(krr)