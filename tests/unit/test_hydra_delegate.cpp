#include <pxr/pxr.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/imaging/hd/renderDelegate.h>

#include <iostream>
#include <stdexcept>

PXR_NAMESPACE_USING_DIRECTIVE

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
	try {
		if (argc != 2) throw std::invalid_argument("Expected the Hydra plugin directory");
		PlugRegistry::GetInstance().RegisterPlugins(std::string(argv[1]) + "/plugInfo.json");
		auto delegate = HdRendererPluginRegistry::GetInstance().CreateRenderDelegate(
			TfToken("HdKiRaRayRendererPlugin"));
		require(bool(delegate), "Could not create the KiRaRay delegate");
		auto error = [&] {
			const auto stats = delegate->GetRenderStats();
			const auto found = stats.find("krr:error");
			require(found != stats.end() && found->second.IsHolding<std::string>(),
				"The delegate did not expose its error status");
			return found->second.Get<std::string>();
		};
		auto retry = [&](const char *name, const VtValue &accepted, const VtValue &invalid) {
			const TfToken key(name);
			delegate->SetRenderSetting(key, accepted);
			require(error().empty(), "Valid wavefront setting was rejected");
			delegate->SetRenderSetting(key, invalid);
			require(!error().empty(), "Invalid wavefront setting did not report an error");
			require(delegate->GetRenderSetting(key) == accepted, "Invalid value replaced the accepted setting");
			delegate->SetRenderSetting(key, accepted);
			require(error().empty(), "Retrying the accepted setting did not clear the error");
		};
		retry("krr:maxDepth", VtValue(10), VtValue(-1));
		retry("krr:nee", VtValue(true), VtValue(1));
		retry("krr:rr", VtValue(0.8), VtValue(0.0));
		retry("krr:rr", VtValue(0.8), VtValue(1.00000001));
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		return 1;
	}
}
