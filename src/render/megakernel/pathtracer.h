#pragma once

#include "window.h"
#include "scene.h"
#include "camera.h"
#include "file.h"

#include "device/optix.h"
#include "device/buffer.h"
#include "device/context.h"
#include "path.h"

NAMESPACE_BEGIN(krr)

class MegakernelPathTracer: public RenderPass{
public:
	using SharedPtr = std::shared_ptr<MegakernelPathTracer>;
	KRR_REGISTER_PASS_DEC(MegakernelPathTracer);
	MegakernelPathTracer() = default;

	void renderUI() override;
	void render(RenderContext *context) override;
	void resize(const Vector2i &size) override;
	void setScene(Scene::SharedPtr scene) override;
	string getName() const override { return "MegakernelPathTracer"; }

protected:
	void compilePrograms();

	OptixBackend::SharedPtr optixBackend;
	LaunchParameters<MegakernelPathTracer> launchParams;

	friend void to_json(json &j, const MegakernelPathTracer &p) {
		j = json{
			{ "nee", p.launchParams.NEE },
			{ "max_depth", p.launchParams.maxDepth },
			{ "rr", p.launchParams.probRR },
		};
	}

	friend void from_json(const json &j, MegakernelPathTracer &p) {
		p.launchParams.NEE = j.value("nee", true);
		p.launchParams.maxDepth = j.value("max_depth", 10);
		p.launchParams.probRR = j.value("rr", 0.8);
	}

private:
	bool needsRecompile{false};
};

NAMESPACE_END(krr)