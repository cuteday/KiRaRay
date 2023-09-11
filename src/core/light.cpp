#include "light.h"
#include "window.h"

#include <cmath>

KRR_NAMESPACE_BEGIN

namespace rt {

KRR_DEVICE LightSample DiffuseAreaLight::sampleLi(Vector2f u,
													   const LightSampleContext &ctx) const {
	LightSample ls				= {};
	ShapeSampleContext shapeCtx = {ctx.p, ctx.n};
	ShapeSample ss				= shape.sample(u, shapeCtx);
	DCHECK(!isnan(ss.pdf));
	Interaction &intr = ss.intr;
	intr.wo			  = normalize(ctx.p - intr.p);

	ls.intr = intr;
	ls.pdf	= ss.pdf;
	ls.L	= L(intr.p, intr.n, intr.uv, intr.wo);
	return ls;
}

}

KRR_NAMESPACE_END