#include "gbuffer.h"
#include "window.h"

KRR_NAMESPACE_BEGIN

extern "C" char GBUFFER_PTX[];

void GBufferPass::render(RenderContext* context) {

}

void GBufferPass::renderUI() {
	ui::BulletText("Enabled components");
	ui::Checkbox("Depth", &mEnableDepth);
	ui::Checkbox("Diffuse", &mEnableDiffuse);
	ui::Checkbox("Specular", &mEnableSpecular);
	ui::Checkbox("Normal", &mEnableNormal);
	ui::Checkbox("Emissive", &mEnableEmissive);
	ui::Checkbox("Motion", &mEnableMotion);
}

KRR_NAMESPACE_END