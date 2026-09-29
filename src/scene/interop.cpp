#include "interop.h"
#include "ext/mikktspace/mikktspace.h"

#include <algorithm>
#include <numeric>
#include <set>

NAMESPACE_BEGIN(krr)
namespace interop {
namespace {

struct TangentFace {
	Mesh *mesh;
	Vector3i indices;
};

const TangentFace &tangentFace(const SMikkTSpaceContext *context, int face) {
	return (*static_cast<const std::vector<TangentFace> *>(context->m_pUserData))[face];
}

}

void generateTangents(const std::vector<Mesh::SharedPtr> &meshes) {
	std::vector<TangentFace> faces;
	for (const auto &mesh : meshes) {
		if (mesh->texcoords.empty()) continue;
		mesh->tangents.resize(mesh->positions.size());
		for (const auto &triangle : mesh->indices) faces.push_back({mesh.get(), triangle});
	}
	if (faces.empty()) return;
	SMikkTSpaceInterface callbacks{};
	callbacks.m_getNumFaces = [](const SMikkTSpaceContext *c) {
		return int(static_cast<const std::vector<TangentFace> *>(c->m_pUserData)->size());
	};
	callbacks.m_getNumVerticesOfFace = [](const SMikkTSpaceContext *, int) { return 3; };
	callbacks.m_getPosition = [](const SMikkTSpaceContext *c, float *out, int face, int corner) {
		const auto &f = tangentFace(c, face);
		std::copy_n(f.mesh->positions[f.indices[corner]].data(), 3, out);
	};
	callbacks.m_getNormal = [](const SMikkTSpaceContext *c, float *out, int face, int corner) {
		const auto &f = tangentFace(c, face);
		std::copy_n(f.mesh->normals[f.indices[corner]].data(), 3, out);
	};
	callbacks.m_getTexCoord = [](const SMikkTSpaceContext *c, float *out, int face, int corner) {
		const auto &f = tangentFace(c, face);
		std::copy_n(f.mesh->texcoords[f.indices[corner]].data(), 2, out);
	};
	callbacks.m_setTSpaceBasic = [](const SMikkTSpaceContext *c, const float *tangent, float, int face, int corner) {
		const auto &f = tangentFace(c, face);
		f.mesh->tangents[f.indices[corner]] = Vector3f(tangent[0], tangent[1], tangent[2]);
	};
	SMikkTSpaceContext context{&callbacks, &faces};
	if (!genTangSpaceDefault(&context))
		throw std::runtime_error("Cannot generate material tangents for " + meshes.front()->getName());
}

namespace {

template <typename T> T value(const Primvar<T> &primvar, int face, int corner, int vertex) {
	int index = 0;
	switch (primvar.interpolation) {
	case Interpolation::Constant: break;
	case Interpolation::Uniform: index = face; break;
	case Interpolation::Vertex: index = vertex; break;
	case Interpolation::FaceVarying: index = corner; break;
	}
	if (!primvar.indices.empty()) {
		if (index < 0 || size_t(index) >= primvar.indices.size())
			throw std::invalid_argument("Primvar index array does not match mesh topology");
		index = primvar.indices[index];
	}
	if (index < 0 || size_t(index) >= primvar.values.size())
		throw std::invalid_argument("Primvar value array does not match mesh topology");
	if (!primvar.values[index].allFinite())
		throw std::invalid_argument("Mesh primvar contains a nonfinite value");
	return primvar.values[index];
}

float cross2(const Vector2f &a, const Vector2f &b, const Vector2f &c) {
	return (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
}

std::vector<Vector3i> triangulate(const std::vector<Vector3f> &points) {
	if (points.size() == 3) return {{0, 1, 2}};
	Vector3f normal = Vector3f::Zero();
	for (size_t i = 0; i < points.size(); ++i)
		normal += points[i].cross(points[(i + 1) % points.size()]);
	Eigen::Index axis;
	normal.cwiseAbs().maxCoeff(&axis);
	std::vector<Vector2f> projected;
	for (const auto &point : points)
		projected.emplace_back(point[(axis + 1) % 3], point[(axis + 2) % 3]);
	float area = 0.f;
	for (size_t i = 0; i < projected.size(); ++i) {
		const auto &a = projected[i], &b = projected[(i + 1) % projected.size()];
		area += a.x() * b.y() - b.x() * a.y();
	}
	if (area == 0.f) throw std::invalid_argument("Cannot triangulate a degenerate polygon");
	const float sign = area > 0.f ? 1.f : -1.f;
	std::vector<int> remaining(points.size());
	std::iota(remaining.begin(), remaining.end(), 0);
	std::vector<Vector3i> result;
	while (remaining.size() > 3) {
		bool clipped = false;
		for (size_t i = 0; i < remaining.size(); ++i) {
			int a = remaining[(i + remaining.size() - 1) % remaining.size()];
			int b = remaining[i], c = remaining[(i + 1) % remaining.size()];
			if (sign * cross2(projected[a], projected[b], projected[c]) <= 0.f) continue;
			bool contains = false;
			for (int other : remaining) {
				if (other == a || other == b || other == c) continue;
				if (sign * cross2(projected[a], projected[b], projected[other]) >= 0.f &&
					sign * cross2(projected[b], projected[c], projected[other]) >= 0.f &&
					sign * cross2(projected[c], projected[a], projected[other]) >= 0.f) {
					contains = true;
					break;
				}
			}
			if (contains) continue;
			result.emplace_back(a, b, c);
			remaining.erase(remaining.begin() + i);
			clipped = true;
			break;
		}
		if (!clipped) throw std::invalid_argument("Cannot triangulate polygon: degenerate or self-intersecting edges");
	}
	result.emplace_back(remaining[0], remaining[1], remaining[2]);
	return result;
}

}

Material::SharedPtr errorMaterial(const std::string &name) {
	auto material = std::make_shared<Material>();
	material->setName(name + " [unsupported]");
	material->mBsdfType = MaterialType::Diffuse;
	material->mShadingModel = Material::ShadingModel::MetallicRoughness;
	material->mMaterialParams.diffuse = RGBA{1.f, 0.f, 1.f, 1.f};
	material->setConstantTexture(Material::TextureType::Emissive, RGBA{0.25f, 0.f, 0.25f, 1.f});
	return material;
}

std::vector<Mesh::SharedPtr> makeMeshes(const MeshInput &input) {
	if (!input.faceMaterials.empty() && input.faceMaterials.size() != input.faceCounts.size())
		throw std::invalid_argument("Face material assignments do not match mesh topology");
	std::set<int> holes(input.holeFaces.begin(), input.holeFaces.end());
	for (int face : holes)
		if (face < 0 || size_t(face) >= input.faceCounts.size())
			throw std::invalid_argument("Mesh hole index is out of range");
	std::map<int, Mesh::SharedPtr> parts;
	size_t offset = 0;
	for (size_t face = 0; face < input.faceCounts.size(); ++face) {
		int count = input.faceCounts[face];
		if (count < 3 || size_t(count) > input.faceIndices.size() - offset)
			throw std::invalid_argument("Invalid polygon vertex count");
		std::vector<Vector3f> points;
		for (int corner = 0; corner < count; ++corner) {
			int vertex = input.faceIndices[offset + corner];
			if (vertex < 0 || size_t(vertex) >= input.points.size() || !input.points[vertex].allFinite())
				throw std::invalid_argument("Invalid mesh point or vertex index");
			points.push_back(input.points[vertex]);
		}
		if (!holes.count(int(face))) {
			int slot = input.faceMaterials.empty() ? 0 : input.faceMaterials[face];
			if (slot < 0 || size_t(slot) >= input.materials.size())
				throw std::invalid_argument("Mesh material slot is out of range");
			auto &mesh = parts[slot];
			if (!mesh) {
				mesh = std::make_shared<Mesh>();
				mesh->setName(input.name + "/" + std::to_string(slot));
				mesh->setMaterial(input.materials[slot]);
			}
			for (Vector3i triangle : triangulate(points)) {
				if (input.leftHanded) std::swap(triangle[1], triangle[2]);
				Vector3f normal = (points[triangle[1]] - points[triangle[0]]).cross(
					points[triangle[2]] - points[triangle[0]]);
				if (normal.squaredNorm() == 0.f) continue;
				normal.normalize();
				int start = int(mesh->positions.size());
				for (int corner : {triangle[0], triangle[1], triangle[2]}) {
					int vertex = input.faceIndices[offset + corner];
					mesh->positions.push_back(points[corner]);
					Vector3f n = input.normals.values.empty() ? normal :
						value(input.normals, int(face), int(offset) + corner, vertex);
					mesh->normals.push_back(n.squaredNorm() > 0.f ? n.normalized().eval() : normal);
					if (!input.texcoords.values.empty())
						mesh->texcoords.push_back(value(input.texcoords, int(face), int(offset) + corner, vertex));
				}
				mesh->indices.emplace_back(start, start + 1, start + 2);
			}
		}
		offset += count;
	}
	if (offset != input.faceIndices.size())
		throw std::invalid_argument("Unused entries in polygon vertex indices");
	std::vector<Mesh::SharedPtr> result;
	for (auto &[slot, mesh] : parts) {
		if (mesh->indices.empty()) continue;
		mesh->computeBoundingBox();
		result.push_back(mesh);
	}
	generateTangents(result);
	return result;
}

SceneGraphNode::SharedPtr attachMeshes(Scene::SharedPtr scene,
	const std::vector<Mesh::SharedPtr> &meshes, const Affine3f &transform,
	const std::string &name, SceneGraphNode::SharedPtr parent) {
	auto graph = scene->getSceneGraph();
	auto node = std::make_shared<SceneGraphNode>(name);
	node->setLocalTransform(transform);
	graph->attach(parent ? parent : graph->getRoot(), node);
	for (const auto &mesh : meshes) {
		if (std::find(scene->getMeshes().begin(), scene->getMeshes().end(), mesh) == scene->getMeshes().end())
			graph->addMesh(mesh);
		auto material = mesh->getMaterial();
		if (material && std::find(scene->getMaterials().begin(), scene->getMaterials().end(), material) == scene->getMaterials().end())
			graph->attachLeaf(graph->getRoot(), material, material->getName());
		graph->attachLeaf(node, std::make_shared<MeshInstance>(mesh), mesh->getName());
	}
	return node;
}

SceneGraphNode::SharedPtr attachLight(Scene::SharedPtr scene, const LightInput &input,
	SceneGraphNode::SharedPtr parent) {
	const float scale = input.intensity * std::exp2(input.exposure);
	if (!std::isfinite(scale) || scale < 0.f || !input.color.allFinite() || input.color.minCoeff() < 0.f)
		throw std::invalid_argument("Light intensity and color must be finite and nonnegative");
	if (!input.transform.matrix().allFinite()) throw std::invalid_argument("Light transform must be finite");
	if ((input.type == LightInput::Type::Rectangle && (!(input.width > 0) || !(input.height > 0))) ||
		((input.type == LightInput::Type::Disk || input.type == LightInput::Type::Sphere) && !(input.radius > 0)))
		throw std::invalid_argument("Area light dimensions must be positive");
	auto graph = scene->getSceneGraph();
	SceneLight::SharedPtr light;
	if (input.type == LightInput::Type::Point || input.type == LightInput::Type::Spot) {
		float strength = scale * (input.normalize ? 0.25f : 1.f);
		if (input.type == LightInput::Type::Point)
			light = std::make_shared<PointLight>(input.color, strength);
		else
			light = std::make_shared<SpotLight>(input.color, strength,
				input.coneAngle * (1.f - input.coneSoftness), input.coneAngle);
	} else if (input.type == LightInput::Type::Distant) {
		float irradiance = scale;
		if (!input.normalize && input.distantAngle > 0.f) {
			float angle = std::min(input.distantAngle,360.f) * M_PI / 360.f;
			float size = std::sin(angle) * std::sin(angle);
			irradiance *= M_PI * (angle <= .5f * M_PI ? size : 2.f - size);
		}
		if (input.distantAngle > 0.f)
			Log(Warning,"Light %s: using a directional source with the distant light's integrated illuminance",input.name.c_str());
		light = std::make_shared<DirectionalLight>(input.color, irradiance);
	} else if (input.type == LightInput::Type::Dome) {
		auto dome = std::make_shared<InfiniteLight>(input.color, scale);
		if (!input.texture.empty()) {
			auto texture = Texture::createFromFile(input.texture);
			if (!texture->hasImage()) throw std::runtime_error("Cannot read environment texture: " + input.texture);
			auto source = texture->getImage();
			auto tinted = std::make_shared<Image>(source->getSize(),Image::Format::RGBAfloat);
			auto target = reinterpret_cast<float *>(tinted->data());
			for (int pixel = 0; pixel < source->getSize().prod(); ++pixel) {
				for (int channel = 0; channel < 3; ++channel) {
					float value;
					if (source->getFormat() == Image::Format::RGBAfloat)
						value = reinterpret_cast<const float *>(source->data())[pixel*4+channel];
					else {
						value = source->data()[pixel*4+channel]/255.f;
						value = value <= .04045f ? value/12.92f : std::pow((value+.055f)/1.055f,2.4f);
					}
					target[pixel*4+channel] = value*input.color[channel];
				}
				target[pixel*4+3] = 1.f;
			}
			texture->mImage = tinted;
			dome->setTexture(texture);
		} else dome->setTexture(std::make_shared<Texture>(RGBA{input.color[0],input.color[1],input.color[2],1.f}));
		light = dome;
	}
	if (light) {
		auto node = graph->attachLeaf(parent ? parent : graph->getRoot(), light, input.name);
		node->setLocalTransform(input.transform);
		return node;
	}
	MeshInput mesh;
	mesh.name = input.name;
	float area = 0.f;
	if (input.type == LightInput::Type::Rectangle) {
		float x = input.width * 0.5f, y = input.height * 0.5f;
		mesh.points = {{-x, -y, 0}, {-x, y, 0}, {x, y, 0}, {x, -y, 0}};
		mesh.faceCounts = {4};
		mesh.faceIndices = {0, 1, 2, 3};
	} else if (input.type == LightInput::Type::Disk) {
		for (int i = 0; i < 64; ++i) {
			float angle = -2.f * M_PI * i / 64.f;
			mesh.points.emplace_back(input.radius * std::cos(angle), input.radius * std::sin(angle), 0.f);
			mesh.faceIndices.push_back(i);
		}
		mesh.faceCounts = {64};
	} else if (input.type == LightInput::Type::Sphere) {
		for (int y = 0; y <= 16; ++y) {
			float theta = M_PI * y / 16.f;
			for (int x = 0; x < 32; ++x) {
				float phi = 2.f * M_PI * x / 32.f;
				mesh.points.emplace_back(input.radius * std::sin(theta) * std::cos(phi),
					input.radius * std::sin(theta) * std::sin(phi), input.radius * std::cos(theta));
			}
		}
		for (int y = 0; y < 16; ++y) for (int x = 0; x < 32; ++x) {
			int a = y * 32 + x, b = y * 32 + (x + 1) % 32;
			if (y != 0) { mesh.faceCounts.push_back(3); mesh.faceIndices.insert(mesh.faceIndices.end(), {a, a + 32, b}); }
			if (y != 15) { mesh.faceCounts.push_back(3); mesh.faceIndices.insert(mesh.faceIndices.end(), {b, a + 32, b + 32}); }
		}
	}
	for (auto &point : mesh.points) point = input.transform * point;
	mesh.leftHanded = input.transform.linear().determinant() < 0.f;
	auto material = std::make_shared<Material>();
	material->setName(input.name);
	material->mBsdfType = MaterialType::Diffuse;
	material->mMaterialParams.diffuse = RGBA{0.f, 0.f, 0.f, 1.f};
	mesh.materials = {material};
	auto parts = makeMeshes(mesh);
	for (const auto &part : parts) for (const auto &triangle : part->indices)
		area += 0.5f * (part->positions[triangle[1]] - part->positions[triangle[0]]).cross(
			part->positions[triangle[2]] - part->positions[triangle[0]]).norm();
	if (area <= 0.f) throw std::invalid_argument("Area light must have positive area");
	RGB emission = input.color * (input.normalize ? scale / area : scale);
	for (auto &part : parts) {
		part->Le = emission;
		part->cameraVisible = false;
	}
	return attachMeshes(scene, parts, Affine3f::Identity(), input.name, parent);
}

}
NAMESPACE_END(krr)
