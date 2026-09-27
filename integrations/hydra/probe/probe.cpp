#include <pxr/pxr.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/type.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/instancer.h>
#include <pxr/imaging/hd/renderBuffer.h>
#include <pxr/imaging/hd/renderDelegate.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPass.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/rendererPlugin.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/resourceRegistry.h>
#include <pxr/imaging/hd/rprim.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <cstdio>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

class KrrProbeBuffer final : public HdRenderBuffer {
public:
    explicit KrrProbeBuffer(const SdfPath& id) : HdRenderBuffer(id) {}

    bool Allocate(const GfVec3i& dimensions, HdFormat format, bool multiSampled) override {
        if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[2] != 1 ||
            (format != HdFormatFloat32Vec4 && format != HdFormatFloat32) || multiSampled)
            return false;
        mDimensions = dimensions;
        mFormat = format;
        mPixels.assign(size_t(dimensions[0]) * dimensions[1] * channels(), 0.f);
        mConverged = false;
        return true;
    }
    unsigned int GetWidth() const override { return mDimensions[0]; }
    unsigned int GetHeight() const override { return mDimensions[1]; }
    unsigned int GetDepth() const override { return mDimensions[2]; }
    HdFormat GetFormat() const override { return mFormat; }
    bool IsMultiSampled() const override { return false; }
    void* Map() override { ++mMappings; return mPixels.data(); }
    void Unmap() override { --mMappings; }
    bool IsMapped() const override { return mMappings != 0; }
    void Resolve() override {}
    bool IsConverged() const override { return mConverged; }

    void fill(const TfToken& aov) {
        for (unsigned int y = 0; y < GetHeight(); ++y) {
            for (unsigned int x = 0; x < GetWidth(); ++x) {
                float* pixel = mPixels.data() + (size_t(y) * GetWidth() + x) * channels();
                if (channels() == 4) {
                    pixel[0] = float(x) / std::max(1u, GetWidth() - 1);
                    pixel[1] = float(y) / std::max(1u, GetHeight() - 1);
                    pixel[2] = 0.25f;
                    pixel[3] = 1.f;
                } else {
                    pixel[0] = 0.5f;
                }
            }
        }
        mConverged = true;
        std::printf("KRR_HYDRA_PROBE_AOV %s %u %u\n", aov.GetText(), GetWidth(), GetHeight());
        std::fflush(stdout);
    }

protected:
    void _Deallocate() override { mPixels.clear(); mDimensions = GfVec3i(0); }

private:
    unsigned int channels() const { return mFormat == HdFormatFloat32Vec4 ? 4 : 1; }
    GfVec3i mDimensions{0};
    HdFormat mFormat{HdFormatInvalid};
    std::vector<float> mPixels;
    unsigned int mMappings{0};
    bool mConverged{false};
};

class KrrProbePass final : public HdRenderPass {
public:
    KrrProbePass(HdRenderIndex* index, const HdRprimCollection& collection)
        : HdRenderPass(index, collection) {}
    bool IsConverged() const override { return mConverged; }

protected:
    void _Execute(const HdRenderPassStateSharedPtr& state, const TfTokenVector&) override {
        for (const auto& binding : state->GetAovBindings()) {
            auto* buffer = binding.renderBuffer;
            if (!buffer)
                buffer = static_cast<HdRenderBuffer*>(GetRenderIndex()->GetBprim(
                    HdPrimTypeTokens->renderBuffer, binding.renderBufferId));
            if (auto* probeBuffer = dynamic_cast<KrrProbeBuffer*>(buffer))
                probeBuffer->fill(binding.aovName);
        }
        mConverged = true;
    }

private:
    bool mConverged{false};
};

class KrrProbeDelegate final : public HdRenderDelegate {
public:
    KrrProbeDelegate() : mRegistry(std::make_shared<HdResourceRegistry>()) {}
    const TfTokenVector& GetSupportedRprimTypes() const override {
        static const TfTokenVector types;
        return types;
    }
    const TfTokenVector& GetSupportedSprimTypes() const override {
        static const TfTokenVector types{HdPrimTypeTokens->camera};
        return types;
    }
    const TfTokenVector& GetSupportedBprimTypes() const override {
        static const TfTokenVector types{HdPrimTypeTokens->renderBuffer};
        return types;
    }
    HdResourceRegistrySharedPtr GetResourceRegistry() const override { return mRegistry; }
    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex* index, const HdRprimCollection& collection) override {
        return std::make_shared<KrrProbePass>(index, collection);
    }
    HdInstancer* CreateInstancer(HdSceneDelegate* delegate, const SdfPath& id) override {
        return new HdInstancer(delegate, id);
    }
    void DestroyInstancer(HdInstancer* instancer) override { delete instancer; }
    HdRprim* CreateRprim(const TfToken&, const SdfPath&) override { return nullptr; }
    void DestroyRprim(HdRprim* prim) override { delete prim; }
    HdSprim* CreateSprim(const TfToken& type, const SdfPath& id) override {
        return type == HdPrimTypeTokens->camera ? new HdCamera(id) : nullptr;
    }
    HdSprim* CreateFallbackSprim(const TfToken& type) override { return CreateSprim(type, SdfPath()); }
    void DestroySprim(HdSprim* prim) override { delete prim; }
    HdBprim* CreateBprim(const TfToken& type, const SdfPath& id) override {
        return type == HdPrimTypeTokens->renderBuffer ? new KrrProbeBuffer(id) : nullptr;
    }
    HdBprim* CreateFallbackBprim(const TfToken& type) override { return CreateBprim(type, SdfPath()); }
    void DestroyBprim(HdBprim* prim) override { delete prim; }
    void CommitResources(HdChangeTracker*) override {}
    HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const override {
        if (name == HdAovTokens->color)
            return {HdFormatFloat32Vec4, false, VtValue(GfVec4f(0.f))};
        if (name == HdAovTokens->depth)
            return {HdFormatFloat32, false, VtValue(1.f)};
        return {};
    }

private:
    HdResourceRegistrySharedPtr mRegistry;
};

class HdKrrProbeRendererPlugin final : public HdRendererPlugin {
public:
    bool IsSupported(const HdRendererCreateArgs&, std::string*) const override { return true; }
    HdRenderDelegate* CreateRenderDelegate() override { return new KrrProbeDelegate; }
    HdRenderDelegate* CreateRenderDelegate(const HdRenderSettingsMap&) override { return new KrrProbeDelegate; }
    void DeleteRenderDelegate(HdRenderDelegate* delegate) override { delete delegate; }
};

TF_REGISTRY_FUNCTION(TfType) {
    HdRendererPluginRegistry::Define<HdKrrProbeRendererPlugin>();
}

PXR_NAMESPACE_CLOSE_SCOPE
