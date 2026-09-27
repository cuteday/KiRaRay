#include <iostream>
#include <stdexcept>

#include "scene/interop.h"

using namespace krr;

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

template <typename F> void rejects(F operation) {
	try { operation(); } catch (const std::invalid_argument &) { return; }
	throw std::runtime_error("Invalid mesh input was accepted");
}

float area(const Mesh &mesh) {
	float result = 0.f;
	for (const auto &triangle : mesh.indices)
		result += .5f * (mesh.positions[triangle[1]] - mesh.positions[triangle[0]]).cross(
			mesh.positions[triangle[2]] - mesh.positions[triangle[0]]).norm();
	return result;
}

Scene::SharedPtr makeScene() {
	auto scene = std::make_shared<Scene>();
	scene->getSceneGraph()->setRoot(std::make_shared<SceneGraphNode>("root"));
	scene->getSceneGraph()->setScene(scene);
	return scene;
}

int main() {
	try {
		interop::MeshInput input;
		input.points = {Vector3f{0, 0, 0}, Vector3f{2, 0, 0}, Vector3f{2, 2, 0}, Vector3f{1, 1, 0}, Vector3f{0, 2, 0}};
		input.faceCounts = {5};
		input.faceIndices = {0, 1, 2, 3, 4};
		input.materials = {std::make_shared<Material>("first"), std::make_shared<Material>("second")};
		auto concave = interop::makeMeshes(input);
		require(concave.size() == 1 && concave[0]->indices.size() == 3 && std::abs(area(*concave[0]) - 3.f) < 1e-6f,
			"Concave polygon triangulation changed its area");
		input.leftHanded = true;
		auto reversed = interop::makeMeshes(input);
		require(reversed[0]->normals[0][2] < 0.f, "Left-handed winding did not reverse generated normals");
		input.leftHanded = false;
		input.points = {Vector3f{0, 0, 0}, Vector3f{1, 0, 0}, Vector3f{1, 1, 0}, Vector3f{0, 1, 0}};
		input.faceCounts = {3, 3};
		input.faceIndices = {0, 1, 2, 0, 2, 3};
		input.texcoords.interpolation = interop::Interpolation::FaceVarying;
		input.texcoords.values = {Vector2f{0, 0}, Vector2f{1, 0}, Vector2f{1, 1}, Vector2f{0, 1}, Vector2f{.25f, .75f}};
		input.texcoords.indices = {0, 1, 2, 4, 2, 3};
		input.normals.interpolation = interop::Interpolation::Uniform;
		input.normals.values = {Vector3f{0, 0, 2}, Vector3f{0, 2, 0}};
		auto seams = interop::makeMeshes(input);
		require(seams[0]->positions[0] == seams[0]->positions[3] &&
			seams[0]->texcoords[0] != seams[0]->texcoords[3], "Indexed face-varying UV seam was merged");
		require(seams[0]->normals[0].isApprox(Vector3f::UnitZ()) &&
			seams[0]->normals[3].isApprox(Vector3f::UnitY()), "Uniform normals were not preserved and normalized");
		auto mirrored = input;
		mirrored.normals.values = {Vector3f::UnitZ(),Vector3f::UnitZ()};
		mirrored.texcoords.indices.clear();
		mirrored.texcoords.values = {Vector2f{0,0},Vector2f{-1,0},Vector2f{-1,1},Vector2f{0,0},Vector2f{-1,1},Vector2f{0,1}};
		auto tangentMesh = interop::makeMeshes(mirrored)[0];
		for (const auto &tangent : tangentMesh->tangents)
			require(tangent.isApprox(-Vector3f::UnitX(),1e-5f),"Mirrored UV tangent orientation changed");
		auto smooth = mirrored;
		smooth.texcoords.values = {Vector2f{0,0},Vector2f{1,0},Vector2f{1,1},
			Vector2f{0,0},Vector2f{1,1},Vector2f{0,2}};
		auto smoothMesh = interop::makeMeshes(smooth)[0];
		require(std::abs(smoothMesh->tangents[0].y()) > .05f,
			"Skewed UV fixture did not average tangents across the shared edge");
		smooth.faceMaterials = {0, 1};
		auto smoothParts = interop::makeMeshes(smooth);
		for (int face = 0; face < 2; ++face)
			for (int corner = 0; corner < 3; ++corner)
				require(smoothParts[face]->tangents[corner].isApprox(
					smoothMesh->tangents[3 * face + corner], 1e-5f),
					"Material partition changed a smooth normal-map tangent");
		auto firstSubmesh = smooth;
		firstSubmesh.faceCounts = {3};
		firstSubmesh.faceIndices = {0, 1, 2};
		firstSubmesh.faceMaterials.clear();
		firstSubmesh.texcoords.values.resize(3);
		auto secondSubmesh = firstSubmesh;
		secondSubmesh.faceIndices = {0, 2, 3};
		secondSubmesh.texcoords.values.assign(smooth.texcoords.values.begin() + 3,
			smooth.texcoords.values.end());
		std::vector<Mesh::SharedPtr> separateMeshes{
			interop::makeMeshes(firstSubmesh)[0], interop::makeMeshes(secondSubmesh)[0]};
		require(!separateMeshes[0]->tangents[0].isApprox(smoothMesh->tangents[0], 1e-5f),
			"Separate mesh fixture did not expose the material-boundary tangent seam");
		interop::generateTangents(separateMeshes);
		for (int face = 0; face < 2; ++face)
			for (int corner = 0; corner < 3; ++corner)
				require(separateMeshes[face]->tangents[corner].isApprox(
					smoothMesh->tangents[3 * face + corner], 1e-5f),
					"Pooled submesh tangents differ from the original mesh");
		input.faceMaterials = {0, 1};
		auto parts = interop::makeMeshes(input);
		require(parts.size() == 2 && parts[0]->material == input.materials[0] && parts[1]->material == input.materials[1],
			"Face material assignments were not partitioned");
		auto invalid = input;
		invalid.faceIndices[0] = 100;
		rejects([&] { interop::makeMeshes(invalid); });
		invalid = input;
		invalid.texcoords.indices[0] = -1;
		rejects([&] { interop::makeMeshes(invalid); });
		invalid = input;
		invalid.faceMaterials[0] = 2;
		rejects([&] { interop::makeMeshes(invalid); });
		invalid = input;
		invalid.normals.values[0][0] = std::numeric_limits<float>::quiet_NaN();
		rejects([&] { interop::makeMeshes(invalid); });
		auto scene = makeScene();
		Affine3f affine = Affine3f::Identity();
		affine.matrix()(0, 0) = -2.f;
		affine.matrix()(0, 1) = .3f;
		affine.matrix()(1, 1) = 3.f;
		affine.translation() = Vector3f{1, 2, 3};
		auto first = interop::attachMeshes(scene, parts, affine, "first");
		interop::attachMeshes(scene, parts, Affine3f::Identity(), "second");
		scene->getSceneGraph()->update(1);
		require(scene->getMeshes().size() == 2 && scene->getMeshInstances().size() == 4,
			"Instances did not share mesh geometry");
		require(first->getGlobalTransform().matrix().isApprox(affine.matrix()), "Affine transform lost reflection, scale or shear");
		auto copy = makeScene();
		for (const auto &mesh : parts) copy->addMesh(mesh);
		auto cloned = copy->getSceneGraph()->attach(copy->getSceneGraph()->getRoot(), first);
		copy->getSceneGraph()->update(2);
		require(cloned->getGlobalTransform().matrix().isApprox(affine.matrix()) && copy->getMeshInstances().size() == 2,
			"Cloned instance group lost its affine transform or topology");
		first->setLocalTransform(Affine3f::Identity());
		scene->getSceneGraph()->update(3);
		scene->getSceneGraph()->update(4);
		require(scene->getSceneGraph()->getLastUpdateRecord().frameIndex == 3,
			"Transform propagation produced an extra update on the next quiet frame");
		for (auto type : {interop::LightInput::Type::Rectangle, interop::LightInput::Type::Disk, interop::LightInput::Type::Sphere}) {
			auto lights = makeScene();
			interop::LightInput light;
			light.type = type;
			light.width = 2.f;
			light.height = 3.f;
			light.radius = 1.f;
			light.normalize = true;
			light.intensity = 5.f;
			interop::attachLight(lights, light);
			float totalArea = 0.f;
			for (auto &mesh : lights->getMeshes()) {
				totalArea += area(*mesh);
				require(!mesh->cameraVisible, "Analytic area light geometry is visible to camera rays");
			}
			float expected = type == interop::LightInput::Type::Rectangle ? 6.f :
				type == interop::LightInput::Type::Disk ? M_PI : 4.f * M_PI;
			require(std::abs(totalArea - expected) / expected < .02f, "Area light geometry has incorrect surface area");
			float emission = lights->getMeshes()[0]->Le[0];
			require(std::abs(emission * totalArea - 5.f) < 1e-5f, "Normalized area light power changed with geometry area");
			for (const auto &mesh : lights->getMeshes()) for (const auto &triangle : mesh->indices) {
				Vector3f center = (mesh->positions[triangle[0]] + mesh->positions[triangle[1]] + mesh->positions[triangle[2]]) / 3.f;
				Vector3f normal = mesh->normals[triangle[0]];
				require(type == interop::LightInput::Type::Sphere ? normal.dot(center) > 0 : normal.z() < 0,
					"Area light emitted from the wrong side");
			}
		}
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
