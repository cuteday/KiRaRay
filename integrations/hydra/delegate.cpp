#include "bridge.h"

#include <pxr/pxr.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quath.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/light.h>
#include <pxr/imaging/hd/material.h>
#include <pxr/imaging/hd/mesh.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/repr.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdLux/blackbody.h>
#include <fstream>

PXR_NAMESPACE_OPEN_SCOPE

namespace {
using State = std::shared_ptr<krr::hydra::RenderState>;

void fail(const State &state, const std::exception &error) {
	std::lock_guard lock(state->mutex);
	state->error	 = error.what();
	state->converged = true;
	if (!state->statusPath.empty())
		std::ofstream(state->statusPath) << json{{"error", state->error}};
	krr::hydra::wakeRenderer();
}

krr::Matrix4f matrix(const GfMatrix4d &input) {
	krr::Matrix4f result;
	for (int row = 0; row < 4; ++row)
		for (int col = 0; col < 4; ++col) result(row, col) = float(input[col][row]);
	return result;
}

json value(const VtValue &input) {
	if (input.IsEmpty()) return {};
	if (input.IsHolding<float>()) return input.UncheckedGet<float>();
	if (input.IsHolding<double>()) return input.UncheckedGet<double>();
	if (input.IsHolding<int>()) return input.UncheckedGet<int>();
	if (input.IsHolding<long>()) return input.UncheckedGet<long>();
	if (input.IsHolding<int64_t>()) return input.UncheckedGet<int64_t>();
	if (input.IsHolding<uint64_t>()) return input.UncheckedGet<uint64_t>();
	if (input.IsHolding<bool>()) return input.UncheckedGet<bool>();
	if (input.IsHolding<std::string>()) return input.UncheckedGet<std::string>();
	if (input.IsHolding<TfToken>()) return input.UncheckedGet<TfToken>().GetString();
	if (input.IsHolding<SdfAssetPath>()) {
		const auto &path = input.UncheckedGet<SdfAssetPath>();
		return path.GetResolvedPath().empty() ? path.GetAssetPath() : path.GetResolvedPath();
	}
	if (input.IsHolding<GfVec2f>()) {
		auto v = input.UncheckedGet<GfVec2f>();
		return {v[0], v[1]};
	}
	if (input.IsHolding<GfVec3f>()) {
		auto v = input.UncheckedGet<GfVec3f>();
		return {v[0], v[1], v[2]};
	}
	if (input.IsHolding<GfVec4f>()) {
		auto v = input.UncheckedGet<GfVec4f>();
		return {v[0], v[1], v[2], v[3]};
	}
	throw std::runtime_error("Unsupported Hydra value type: " + input.GetTypeName());
}

krr::interop::Interpolation interpolation(HdInterpolation mode) {
	switch (mode) {
		case HdInterpolationConstant:
			return krr::interop::Interpolation::Constant;
		case HdInterpolationUniform:
			return krr::interop::Interpolation::Uniform;
		case HdInterpolationFaceVarying:
			return krr::interop::Interpolation::FaceVarying;
		default:
			return krr::interop::Interpolation::Vertex;
	}
}

template <typename T>
bool samePrimvar(const krr::interop::Primvar<T> &a, const krr::interop::Primvar<T> &b) {
	return a.interpolation == b.interpolation && a.indices == b.indices && a.values == b.values;
}

bool sameGeometry(const krr::hydra::MeshRecord &a, const krr::hydra::MeshRecord &b) {
	return a.visible == b.visible && a.materials == b.materials &&
		   a.input.points == b.input.points && a.input.faceCounts == b.input.faceCounts &&
		   a.input.faceIndices == b.input.faceIndices && a.input.holeFaces == b.input.holeFaces &&
		   a.input.leftHanded == b.input.leftHanded &&
		   a.input.faceMaterials == b.input.faceMaterials &&
		   samePrimvar(a.input.normals, b.input.normals) &&
		   samePrimvar(a.input.texcoords, b.input.texcoords);
}

bool sameLight(const krr::interop::LightInput &a, const krr::interop::LightInput &b) {
	return a.type == b.type && a.transform.matrix() == b.transform.matrix() &&
		   (a.color == b.color).all() && a.intensity == b.intensity && a.exposure == b.exposure &&
		   a.width == b.width && a.height == b.height && a.radius == b.radius &&
		   a.coneAngle == b.coneAngle && a.coneSoftness == b.coneSoftness &&
		   a.distantAngle == b.distantAngle && a.normalize == b.normalize && a.texture == b.texture;
}

template <typename T>
T lightValue(HdSceneDelegate *delegate, const SdfPath &id, const char *name, T fallback) {
	auto result = delegate->GetLightParamValue(id, TfToken(name));
	return result.IsHolding<T>() ? result.UncheckedGet<T>() : fallback;
}

std::vector<krr::Matrix4f> instanceTransforms(HdSceneDelegate *delegate, const SdfPath &instancer,
											  const SdfPath &prototype, int level = 0) {
	if (instancer.IsEmpty()) return {krr::Matrix4f::Identity()};
	if (level > 32) throw std::runtime_error("Hydra instancing exceeds 32 levels");
	auto indices	  = delegate->GetInstanceIndices(instancer, prototype);
	auto values		  = delegate->Get(instancer, HdInstancerTokens->instanceTransforms);
	auto translations = delegate->Get(instancer, HdInstancerTokens->instanceTranslations);
	auto scales		  = delegate->Get(instancer, HdInstancerTokens->instanceScales);
	auto rotations	  = delegate->Get(instancer, HdInstancerTokens->instanceRotations);
	const auto base	  = matrix(delegate->GetInstancerTransform(instancer));
	std::vector<krr::Matrix4f> result;
	for (int index : indices) {
		krr::Matrix4f local = krr::Matrix4f::Identity();
		if (values.IsHolding<VtMatrix4dArray>()) {
			const auto &array = values.UncheckedGet<VtMatrix4dArray>();
			if (index >= 0 && size_t(index) < array.size()) local = matrix(array[index]);
		}
		if (scales.IsHolding<VtVec3fArray>()) {
			const auto &array = scales.UncheckedGet<VtVec3fArray>();
			if (index >= 0 && size_t(index) < array.size()) {
				krr::Matrix4f scale = krr::Matrix4f::Identity();
				for (int i = 0; i < 3; ++i) scale(i, i) = array[index][i];
				local = scale * local;
			}
		}
		if (rotations.IsHolding<VtQuathArray>()) {
			const auto &array = rotations.UncheckedGet<VtQuathArray>();
			if (index >= 0 && size_t(index) < array.size()) {
				GfMatrix4d rotation(1.0);
				rotation.SetRotate(GfQuatd(array[index]));
				local = matrix(rotation) * local;
			}
		}
		if (translations.IsHolding<VtVec3fArray>()) {
			const auto &array = translations.UncheckedGet<VtVec3fArray>();
			if (index >= 0 && size_t(index) < array.size())
				for (int i = 0; i < 3; ++i) local(i, 3) += array[index][i];
		}
		for (const auto &parent : instanceTransforms(delegate, delegate->GetInstancerId(instancer),
													 instancer, level + 1))
			result.push_back(parent * base * local);
	}
	return result;
}

class KrrMesh final : public HdMesh {
public:
	KrrMesh(const SdfPath &id, State state) : HdMesh(id), mState(std::move(state)) {}
	~KrrMesh() override {
		std::lock_guard lock(mState->mutex);
		mState->scene.meshes.erase(GetId().GetString());
		++mState->scene.geometryVersion;
		mState->changed();
	}
	HdDirtyBits GetInitialDirtyBitsMask() const override { return HdChangeTracker::AllDirty; }
	void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirty,
			  const TfToken &) override try {
		const auto id				 = GetId();
		const bool visibilityChanged = (*dirty & HdChangeTracker::DirtyVisibility) != 0;
		_UpdateInstancer(delegate, dirty);
		_UpdateVisibility(delegate, dirty);
		std::lock_guard lock(mState->mutex);
		auto &record			   = mState->scene.meshes[id.GetString()];
		const HdDirtyBits geometry = HdChangeTracker::DirtyPoints | HdChangeTracker::DirtyTopology |
									 HdChangeTracker::DirtyNormals | HdChangeTracker::DirtyPrimvar |
									 HdChangeTracker::DirtyMaterialId |
									 HdChangeTracker::DirtyVisibility;
		if ((*dirty & geometry) || visibilityChanged || record.input.points.empty()) {
			const auto previous = record;
			record.input		= {};
			record.input.name	= id.GetString();
			auto points			= delegate->Get(id, HdTokens->points);
			if (points.IsHolding<VtVec3fArray>())
				for (const auto &p : points.UncheckedGet<VtVec3fArray>())
					record.input.points.emplace_back(p[0], p[1], p[2]);
			auto topology = delegate->GetMeshTopology(id);
			record.input.faceCounts.assign(topology.GetFaceVertexCounts().begin(),
										   topology.GetFaceVertexCounts().end());
			record.input.faceIndices.assign(topology.GetFaceVertexIndices().begin(),
											topology.GetFaceVertexIndices().end());
			record.input.holeFaces.assign(topology.GetHoleIndices().begin(),
										  topology.GetHoleIndices().end());
			record.input.leftHanded = topology.GetOrientation() == HdTokens->leftHanded;
			record.materials		= {delegate->GetMaterialId(id).GetString()};
			record.input.faceMaterials.assign(record.input.faceCounts.size(), 0);
			for (const auto &subset : topology.GetGeomSubsets()) {
				if (subset.type != HdGeomSubset::TypeFaceSet) continue;
				int slot = int(record.materials.size());
				record.materials.push_back(subset.materialId.GetString());
				for (int face : subset.indices)
					if (face >= 0 && size_t(face) < record.input.faceMaterials.size())
						record.input.faceMaterials[face] = slot;
			}
			for (int mode = HdInterpolationConstant; mode < HdInterpolationCount; ++mode) {
				for (const auto &descriptor :
					 delegate->GetPrimvarDescriptors(id, HdInterpolation(mode))) {
					bool normal = descriptor.name == HdTokens->normals;
					bool uv		= descriptor.name == TfToken("st") ||
							  (descriptor.role == HdPrimvarRoleTokens->textureCoordinate &&
							   record.input.texcoords.values.empty());
					if (!normal && !uv) continue;
					VtIntArray indices;
					auto data = delegate->GetIndexedPrimvar(id, descriptor.name, &indices);
					if (normal && data.IsHolding<VtVec3fArray>()) {
						auto &output		 = record.input.normals;
						output.interpolation = interpolation(HdInterpolation(mode));
						output.indices.assign(indices.begin(), indices.end());
						for (const auto &v : data.UncheckedGet<VtVec3fArray>())
							output.values.emplace_back(v[0], v[1], v[2]);
					}
					if (uv && data.IsHolding<VtVec2fArray>()) {
						auto &output		 = record.input.texcoords;
						output.interpolation = interpolation(HdInterpolation(mode));
						output.indices.assign(indices.begin(), indices.end());
						for (const auto &v : data.UncheckedGet<VtVec2fArray>())
							output.values.emplace_back(v[0], v[1]);
					}
				}
			}
			record.visible = delegate->GetVisible(id);
			if (!sameGeometry(previous, record)) ++mState->scene.geometryVersion;
		}
		auto transforms = instanceTransforms(delegate, delegate->GetInstancerId(id), id);
		for (auto &transform : transforms) transform *= matrix(delegate->GetTransform(id));
		if (transforms.size() != record.transforms.size()) ++mState->scene.geometryVersion;
		record.transforms = std::move(transforms);
		++mState->scene.transformVersion;
		mState->changed();
		*dirty = HdChangeTracker::Clean;
	} catch (const std::exception &error) {
		fail(mState, error);
	}

protected:
	HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override { return bits; }
	void _InitRepr(const TfToken &token, HdDirtyBits *) override {
		if (std::find_if(_reprs.begin(), _reprs.end(), _ReprComparator(token)) == _reprs.end())
			_reprs.emplace_back(token, std::make_shared<HdRepr>());
	}

private:
	State mState;
};

class KrrMaterial final : public HdMaterial {
public:
	KrrMaterial(const SdfPath &id, State state) : HdMaterial(id), mState(std::move(state)) {}
	~KrrMaterial() override {
		std::lock_guard lock(mState->mutex);
		mState->scene.materials.erase(GetId().GetString());
		++mState->scene.materialVersion;
		mState->changed();
	}
	HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
	void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirty) override try {
		krr::interop::MaterialNetwork output;
		output.name	  = GetId().GetString();
		auto resource = delegate->GetMaterialResource(GetId());
		HdMaterialNetwork2 network;
		if (resource.IsHolding<HdMaterialNetworkMap>())
			network = HdConvertToHdMaterialNetwork2(resource.UncheckedGet<HdMaterialNetworkMap>());
		else if (resource.IsHolding<HdMaterialNetwork2>())
			network = resource.UncheckedGet<HdMaterialNetwork2>();
		else
			output.diagnostics.push_back("Hydra material has no supported surface network");
		for (const auto &[path, source] : network.nodes) {
			auto &node		= output.nodes[path.GetString()];
			node.identifier = source.nodeTypeId.GetString();
			for (const auto &[name, parameter] : source.parameters) {
				auto &input = node.inputs[name.GetString()];
				input.type	= parameter.GetTypeName();
				try {
					input.value = value(parameter);
					auto colorSpace =
						source.parameters.find(TfToken("colorSpace:" + name.GetString()));
					if (colorSpace != source.parameters.end())
						input.colorSpace = value(colorSpace->second).get<std::string>();
				} catch (const std::exception &error) {
					input.error = error.what();
				}
			}
			for (const auto &[name, connections] : source.inputConnections) {
				if (connections.empty()) continue;
				auto &input = node.inputs[name.GetString()];
				input.error.clear();
				if (connections.size() != 1)
					input.error = "Array-valued material connections are unsupported";
				input.node	 = connections[0].upstreamNode.GetString();
				input.output = connections[0].upstreamOutputName.GetString();
			}
		}
		auto terminal = network.terminals.find(HdMaterialTerminalTokens->surface);
		if (terminal != network.terminals.end())
			output.terminal = terminal->second.upstreamNode.GetString();
		std::lock_guard lock(mState->mutex);
		mState->scene.materials[output.name] = std::move(output);
		++mState->scene.materialVersion;
		mState->changed();
		*dirty = Clean;
	} catch (const std::exception &error) {
		fail(mState, error);
	}

private:
	State mState;
};

class KrrLight final : public HdLight {
public:
	KrrLight(const SdfPath &id, const TfToken &type, State state) :
		HdLight(id), mType(type), mState(std::move(state)) {}
	~KrrLight() override {
		std::lock_guard lock(mState->mutex);
		mState->scene.lights.erase(GetId().GetString());
		++mState->scene.geometryVersion;
		mState->changed();
	}
	HdDirtyBits GetInitialDirtyBitsMask() const override { return AllDirty; }
	void Sync(HdSceneDelegate *delegate, HdRenderParam *, HdDirtyBits *dirty) override try {
		krr::interop::LightInput input;
		input.name		= GetId().GetString();
		input.transform = krr::Affine3f(matrix(delegate->GetTransform(GetId())));
		using Type		= krr::interop::LightInput::Type;
		if (mType == HdPrimTypeTokens->distantLight)
			input.type = Type::Distant;
		else if (mType == HdPrimTypeTokens->rectLight)
			input.type = Type::Rectangle;
		else if (mType == HdPrimTypeTokens->diskLight)
			input.type = Type::Disk;
		else if (mType == HdPrimTypeTokens->domeLight)
			input.type = Type::Dome;
		else
			input.type = Type::Sphere;
		auto color = lightValue(delegate, GetId(), "color", GfVec3f(1.f));
		if (lightValue(delegate, GetId(), "enableColorTemperature", false))
			color = GfCompMult(color, UsdLuxBlackbodyTemperatureAsRgb(lightValue(
										  delegate, GetId(), "colorTemperature", 6500.f)));
		input.color		   = krr::RGB(color[0], color[1], color[2]);
		input.intensity	   = lightValue(delegate, GetId(), "intensity", 1.f);
		input.exposure	   = lightValue(delegate, GetId(), "exposure", 0.f);
		input.normalize	   = lightValue(delegate, GetId(), "normalize", false);
		input.width		   = lightValue(delegate, GetId(), "width", 1.f);
		input.height	   = lightValue(delegate, GetId(), "height", 1.f);
		input.radius	   = lightValue(delegate, GetId(), "radius", 0.5f);
		input.distantAngle = lightValue(delegate, GetId(), "angle", 0.f);
		input.coneAngle	   = lightValue(delegate, GetId(), "inputs:shaping:cone:angle",
										lightValue(delegate, GetId(), "shaping:cone:angle", 180.f));
		input.coneSoftness =
			lightValue(delegate, GetId(), "inputs:shaping:cone:softness",
					   lightValue(delegate, GetId(), "shaping:cone:softness", 0.f));
		if (input.type == Type::Sphere) {
			if (input.coneAngle < 180.f)
				input.type = Type::Spot;
			else if (input.radius <= 0.f || lightValue(delegate, GetId(), "treatAsPoint", false))
				input.type = Type::Point;
		}
		auto asset = delegate->GetLightParamValue(GetId(), TfToken("texture:file"));
		if (asset.IsHolding<SdfAssetPath>()) input.texture = value(asset).get<std::string>();
		std::lock_guard lock(mState->mutex);
		auto previous = mState->scene.lights.find(input.name);
		if (previous != mState->scene.lights.end() && sameLight(previous->second, input)) {
			*dirty = Clean;
			return;
		}
		mState->scene.lights[input.name] = std::move(input);
		++mState->scene.geometryVersion;
		mState->changed();
		*dirty = Clean;
	} catch (const std::exception &error) {
		fail(mState, error);
	}

private:
	TfToken mType;
	State mState;
};

class KrrBuffer final : public HdRenderBuffer {
public:
	explicit KrrBuffer(const SdfPath &id) : HdRenderBuffer(id) {}
	bool Allocate(const GfVec3i &size, HdFormat format, bool multiSampled) override {
		if (size[0] <= 0 || size[1] <= 0 || size[2] != 1 || multiSampled ||
			(format != HdFormatFloat32Vec4 && format != HdFormatFloat32))
			return false;
		mSize	= size;
		mFormat = format;
		mPixels.assign(size_t(size[0]) * size[1] * channels(), 0.f);
		mConverged = false;
		mImage.reset();
		mDepth.reset();
		return true;
	}
	unsigned int GetWidth() const override { return mSize[0]; }
	unsigned int GetHeight() const override { return mSize[1]; }
	unsigned int GetDepth() const override { return mSize[2]; }
	HdFormat GetFormat() const override { return mFormat; }
	bool IsMultiSampled() const override { return false; }
	void *Map() override {
		++mMappings;
		return mPixels.data();
	}
	void Unmap() override { --mMappings; }
	bool IsMapped() const override { return mMappings != 0; }
	void Resolve() override {}
	bool IsConverged() const override { return mConverged; }
	void update(const std::shared_ptr<const krr::RenderSession::Snapshot> &image,
		const TfToken &aov, uint64_t version, bool converged) {
		mConverged = converged;
		if (!image || image->size[0] != mSize[0] || image->size[1] != mSize[1]) return;
		if (version == mImageVersion && aov == mAov &&
			(channels() == 4 ? mImage.lock() == image : mDepth.lock() == image->depth))
			return;
		const std::vector<float> *depth = nullptr;
		if (channels() == 1) {
			if (!image->depth) return;
			depth = aov == TfToken("linearDepth") ? &image->depth->linear : &image->depth->projected;
		}
		for (int y = 0; y < mSize[1]; ++y)
			for (int x = 0; x < mSize[0]; ++x) {
				size_t source = size_t(mSize[1] - y - 1) * mSize[0] + x;
				size_t target = size_t(y) * mSize[0] + x;
				if (channels() == 4) {
					std::copy_n(image->image.data() + source * 3, 3, mPixels.data() + target * 4);
					mPixels[target * 4 + 3] = 1.f;
				} else if (source < depth->size())
					mPixels[target] = (*depth)[source];
			}
		mImage = image;
		mDepth = image->depth;
		mImageVersion = version;
		mAov = aov;
	}

protected:
	void _Deallocate() override {
		mPixels.clear();
		mSize = GfVec3i(0);
		mImage.reset();
		mDepth.reset();
	}

private:
	size_t channels() const { return mFormat == HdFormatFloat32Vec4 ? 4 : 1; }
	GfVec3i mSize{0};
	HdFormat mFormat{HdFormatInvalid};
	std::vector<float> mPixels;
	unsigned int mMappings{};
	bool mConverged{};
	std::weak_ptr<const krr::RenderSession::Snapshot> mImage;
	std::weak_ptr<const krr::RenderSession::DepthSnapshot> mDepth;
	uint64_t mImageVersion{};
	TfToken mAov;
};

class KrrPass final : public HdRenderPass {
public:
	KrrPass(HdRenderIndex *index, const HdRprimCollection &collection, State state) :
		HdRenderPass(index, collection), mState(std::move(state)) {}
	bool IsConverged() const override {
		std::lock_guard lock(mState->mutex);
		return !mState->error.empty() || (mConverged && mCopiedVersion == mState->version);
	}

protected:
	void _Execute(const HdRenderPassStateSharedPtr &pass, const TfTokenVector &) override try {
		std::lock_guard lock(mState->mutex);
		auto projection = matrix(pass->GetProjectionMatrix());
		auto camera		= matrix(pass->GetWorldToViewMatrix().GetInverse());
		if (!mState->camera.projection.isApprox(projection, 1e-7f) ||
			!mState->camera.cameraToWorld.isApprox(camera, 1e-7f)) {
			mState->camera.projection	 = projection;
			mState->camera.cameraToWorld = camera;
			mState->camera.orthographic	 = std::abs(projection(3, 3) - 1.f) < 1e-6f;
			mState->changed();
		}
		const auto &bindings = pass->GetAovBindings();
		bool depthRequested = std::any_of(bindings.begin(), bindings.end(), [](const auto &binding) {
			return binding.aovName == HdAovTokens->depth || binding.aovName == TfToken("linearDepth");
		});
		if (depthRequested && !mState->depthRequested) mState->changed();
		mState->depthRequested = depthRequested;
		for (const auto &binding : bindings) {
			auto *buffer = binding.renderBuffer;
			if (!buffer)
				buffer = static_cast<HdRenderBuffer *>(GetRenderIndex()->GetBprim(
					HdPrimTypeTokens->renderBuffer, binding.renderBufferId));
			if (auto *target = dynamic_cast<KrrBuffer *>(buffer)) {
				krr::Vector2i size{int(target->GetWidth()), int(target->GetHeight())};
				if (mState->size != size) {
					mState->size = size;
					mState->changed();
				}
				target->update(mState->imageVersion == mState->version ? mState->image : nullptr,
					binding.aovName, mState->imageVersion, mState->converged);
			}
		}
		mConverged	   = mState->converged;
		mCopiedVersion = mState->imageVersion;
		mState->ready  = true;
		krr::hydra::wakeRenderer();
	} catch (const std::exception &error) {
		fail(mState, error);
	}

private:
	State mState;
	bool mConverged{};
	uint64_t mCopiedVersion{};
};

class KrrDelegate final : public HdRenderDelegate {
public:
	KrrDelegate() :
		mState(krr::hydra::createState()), mRegistry(std::make_shared<HdResourceRegistry>()) {}
	~KrrDelegate() override { krr::hydra::releaseState(mState); }
	const TfTokenVector &GetSupportedRprimTypes() const override {
		static const TfTokenVector types{HdPrimTypeTokens->mesh};
		return types;
	}
	const TfTokenVector &GetSupportedSprimTypes() const override {
		static const TfTokenVector types{
			HdPrimTypeTokens->camera,		HdPrimTypeTokens->material,
			HdPrimTypeTokens->distantLight, HdPrimTypeTokens->sphereLight,
			HdPrimTypeTokens->rectLight,	HdPrimTypeTokens->diskLight,
			HdPrimTypeTokens->domeLight};
		return types;
	}
	const TfTokenVector &GetSupportedBprimTypes() const override {
		static const TfTokenVector types{HdPrimTypeTokens->renderBuffer};
		return types;
	}
	HdResourceRegistrySharedPtr GetResourceRegistry() const override { return mRegistry; }
	HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex *index,
										   const HdRprimCollection &collection) override {
		return std::make_shared<KrrPass>(index, collection, mState);
	}
	HdInstancer *CreateInstancer(HdSceneDelegate *delegate, const SdfPath &id) override {
		return new HdInstancer(delegate, id);
	}
	void DestroyInstancer(HdInstancer *instancer) override { delete instancer; }
	HdRprim *CreateRprim(const TfToken &type, const SdfPath &id) override {
		return type == HdPrimTypeTokens->mesh ? new KrrMesh(id, mState) : nullptr;
	}
	void DestroyRprim(HdRprim *prim) override { delete prim; }
	HdSprim *CreateSprim(const TfToken &type, const SdfPath &id) override {
		if (type == HdPrimTypeTokens->camera) return new HdCamera(id);
		if (type == HdPrimTypeTokens->material) return new KrrMaterial(id, mState);
		return new KrrLight(id, type, mState);
	}
	HdSprim *CreateFallbackSprim(const TfToken &type) override {
		return CreateSprim(type, SdfPath());
	}
	void DestroySprim(HdSprim *prim) override { delete prim; }
	HdBprim *CreateBprim(const TfToken &type, const SdfPath &id) override {
		return type == HdPrimTypeTokens->renderBuffer ? new KrrBuffer(id) : nullptr;
	}
	HdBprim *CreateFallbackBprim(const TfToken &type) override {
		return CreateBprim(type, SdfPath());
	}
	void DestroyBprim(HdBprim *prim) override { delete prim; }
	void CommitResources(HdChangeTracker *) override {}
	TfTokenVector GetMaterialRenderContexts() const override {
		return {TfToken("mtlx"), TfToken()};
	}
	TfToken GetMaterialNetworkSelector() const override { return TfToken("mtlx"); }
	HdAovDescriptor GetDefaultAovDescriptor(const TfToken &name) const override {
		if (name == HdAovTokens->color) return {HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.f))};
		if (name == HdAovTokens->depth || name == TfToken("linearDepth"))
			return {HdFormatFloat32, false, VtValue(1.f)};
		return {};
	}
	void SetRenderSetting(const TfToken &key, const VtValue &input) override try {
		if (GetRenderSetting(key) == input) return;
		HdRenderDelegate::SetRenderSetting(key, input);
		std::lock_guard lock(mState->mutex);
		const auto name = key.GetString();
		const auto data = value(input);
		if (name == "krr:paused") {
			mState->paused = data.is_boolean() ? data.get<bool>() : data.get<int>() != 0;
			krr::hydra::wakeRenderer();
			return;
		}
		if (name == "krr:samples")
			mState->samples = std::max(1, data.get<int>());
		else if (name == "krr:seed")
			mState->seed = data.get<uint64_t>();
		else if (name == "krr:priority")
			mState->priority = data.get<int>();
		else if (name == "krr:assetRoot")
			mState->assetRoot = data.get<std::string>();
		else if (name == "krr:graphicsApi")
			mState->graphicsApi = data.get<std::string>();
		else if (name == "krr:statusPath")
			mState->statusPath = data.get<std::string>();
		else if (name == "krr:blenderScene") {
			mState->scene.blenderScene = data.get<bool>();
			++mState->scene.geometryVersion;
		} else if (name == "krr:emissionLuminanceScale") {
			double scale = data.get<double>();
			if (!std::isfinite(scale) || scale <= 0)
				throw std::runtime_error("Emission luminance scale must be positive and finite");
			mState->scene.emissionLuminanceScale = scale;
			++mState->scene.materialVersion;
		} else if (name == "krr:diagnostics") {
			mState->scene.diagnostics =
				json::parse(data.get<std::string>()).get<decltype(mState->scene.diagnostics)>();
			++mState->scene.materialVersion;
		} else
			return;
		mState->changed();
	} catch (const std::exception &error) {
		fail(mState, error);
	}
	VtDictionary GetRenderStats() const override {
		std::lock_guard lock(mState->mutex);
		double progress = mState->image && mState->imageVersion == mState->version ?
			100.0 * mState->image->completedFrames / mState->samples : 0.0;
		return {{"percentDone", VtValue(std::min(100.0, progress))},
				{"krr:error", VtValue(mState->error)}};
	}

private:
	State mState;
	HdResourceRegistrySharedPtr mRegistry;
};

} // namespace

class HdKiRaRayRendererPlugin final : public HdRendererPlugin {
public:
	bool IsSupported(const HdRendererCreateArgs &, std::string *) const override { return true; }
	HdRenderDelegate *CreateRenderDelegate() override { return new KrrDelegate; }
	HdRenderDelegate *CreateRenderDelegate(const HdRenderSettingsMap &settings) override {
		auto *delegate = new KrrDelegate;
		for (const auto &[key, value] : settings) delegate->SetRenderSetting(key, value);
		return delegate;
	}
	void DeleteRenderDelegate(HdRenderDelegate *delegate) override { delete delegate; }
};

TF_REGISTRY_FUNCTION(TfType) { HdRendererPluginRegistry::Define<HdKiRaRayRendererPlugin>(); }

PXR_NAMESPACE_CLOSE_SCOPE
