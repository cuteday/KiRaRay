#include "light.h"
#include "window.h"

#include <cmath>

KRR_NAMESPACE_BEGIN

void InfiniteLight::renderUI() {
	ui::SliderFloat("Intensity", &scale, 0, 10, "%.02f");
	ui::SliderFloat("Rotation", &rotation, 0, 1, "%.03f");
	ui::ColorEdit3("Tint", (float *) &tint);
	if (image.isValid()) image.renderUI();
}

KRR_NAMESPACE_END