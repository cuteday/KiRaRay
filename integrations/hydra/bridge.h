#pragma once

#include "settings.h"
#include "main/renderer.h"
#include "scene/interop.h"
#include "integrations/materials/material.h"
#include <condition_variable>
#include <mutex>

namespace krr::hydra {

struct MeshRecord {
	interop::MeshInput input;
	std::vector<std::string> materials;
	std::vector<Matrix4f> transforms{Matrix4f::Identity()};
	bool visible{true};
};

struct SceneInput {
	std::map<std::string, MeshRecord> meshes;
	std::map<std::string, interop::MaterialNetwork> materials;
	std::map<std::string, interop::LightInput> lights;
	std::map<std::string, std::vector<std::string>> diagnostics;
	std::map<std::string, std::string> opaqueMixBranches;
	double emissionLuminanceScale{1.0};
	bool blenderScene{};
	uint64_t geometryVersion{}, transformVersion{}, materialVersion{};
};

struct RenderState {
	std::mutex mutex;
	std::condition_variable released;
	SceneInput scene;
	CameraState camera;
	WavefrontSettings wavefront;
	Vector2i size{0, 0};
	std::string assetRoot, graphicsApi{"vulkan"}, statusPath;
	std::shared_ptr<const RenderSession::Snapshot> image;
	uint64_t version{1}, imageVersion{}, seed{};
	uint32_t samples{64};
	int priority{1};
	bool alive{true}, active{}, paused{}, converged{}, ready{}, depthRequested{};
	std::string error;

	void changed() {
		++version;
		converged = false;
		ready	  = false;
		error.clear();
	}
};

std::shared_ptr<RenderState> createState();
void releaseState(const std::shared_ptr<RenderState> &state);
void wakeRenderer();

} // namespace krr::hydra
