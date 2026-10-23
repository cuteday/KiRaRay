#include <iostream>
#include <stdexcept>

#include "core/camera.h"

using namespace krr;

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

int main() {
	try {
		rt::CameraData camera;
		camera.transform = Transformation(Matrix4f(Matrix4f::Identity()));
		camera.externalProjection = true;
		camera.projection = perspective(1.f, 1.f, .1f, 100.f);
		camera.nearClip = 0.f;
		camera.inverseProjection = camera.projection.inverse();
		rt::CameraSample center{Vector2f(.5f), Vector2f(.5f), 0.f};
		auto ray = camera.getRay({0, 0}, {1, 1}, center);
		require(ray.origin.isZero() && ray.dir.isApprox(Vector3f{0, 0, -1}), "Perspective center ray is incorrect");
		camera.projection(0, 2) = .3f;
		camera.inverseProjection = camera.projection.inverse();
		ray = camera.getRay({0, 0}, {1, 1}, center);
		require(ray.dir[0] > 0.f && ray.dir[2] < 0.f, "Off-axis projection shift was ignored");
		camera.projection = orthogonal(-1.f, 1.f, -1.f, 1.f, .1f, 100.f);
		camera.inverseProjection = camera.projection.inverse();
		camera.orthographic = true;
		auto left = camera.getRay({0, 0}, {2, 1}, center);
		auto right = camera.getRay({1, 0}, {2, 1}, center);
		require(left.dir.isApprox(right.dir) && left.origin[0] < right.origin[0], "Orthographic rays are incorrect");
		camera.orthographic = false;
		camera.projection = perspective(1.f, 1.f, .1f, 100.f);
		camera.projection.row(2) = 2.f * camera.projection.row(2) - camera.projection.row(3);
		camera.nearClip = -1.f;
		camera.inverseProjection = camera.projection.inverse();
		ray = camera.getRay({0, 0}, {1, 1}, center);
		require(ray.dir.isApprox(Vector3f{0, 0, -1}), "OpenGL projection convention is incorrect");
		Matrix4f shifted = perspective(1.f, 2.f, .1f, 100.f);
		shifted(0, 2) = .3f;
		camera.setProjection(shifted, 0.f, false);
		require(std::abs(camera.aspectRatio - 2.f) < 1e-6f, "Projection source aspect is incorrect");
		camera.setAspectRatio(1.f);
		require(camera.projection.row(0).isApprox(2.f * shifted.row(0)) &&
			camera.projection.row(1).isApprox(shifted.row(1)),
			"Resolution conform did not preserve vertical FOV and horizontal film offset");
		require((camera.projection * camera.inverseProjection).isApprox(Matrix4f::Identity()),
			"Resolution conform did not refresh inverse projection");
		camera.setAspectRatio(2.f);
		require(camera.projection.isApprox(shifted), "Projection aspect round-trip drifted");
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
