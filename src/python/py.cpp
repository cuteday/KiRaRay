#include "py.h"
#include "common.h"

#include "render/wavefront/integrator.h"
#include "render/passes/accumulate.h"
#include "render/passes/tonemapping.h"
#include "render/path/pathtracer.h"
#include "main/renderer.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

void run(const char scene_file[], const char env_file[] = nullptr) {
	gpContext = Context::SharedPtr(new Context());
	RenderApp app(KRR_PROJECT_NAME, { 1920, 1080 },
				  { RenderPass::SharedPtr(new WavefrontPathTracer()),
					RenderPass::SharedPtr(new AccumulatePass()),
					RenderPass::SharedPtr(new ToneMappingPass()) });
	Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
	if (env_file) scene->addInfiniteLight(InfiniteLight(env_file));
	AssimpImporter importer;
	Log(Info, "Scene importing.");
	importer.import(scene_file, scene);
	Log(Info, "Scene loaded.");
	app.setScene(scene);
	app.run();
}

PYBIND11_MODULE(pykrr, m) { 
	m.doc() = "KiRaRay python binding!";

	m.def("run", &run, 
		"Run KiRaRay renderer with default configuration", 
		"scene"_a, "env"_a=nullptr);
}


KRR_NAMESPACE_END