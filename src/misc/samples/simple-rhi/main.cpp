#include <common.h>
#include <logger.h>
#include <nvrhi/vulkan.h>

#include "SineWaveSimulation.h"
#include "devicemanager.h"

KRR_NAMESPACE_BEGIN

class WaveRenderer: public IRenderPass {
public:
	using IRenderPass::IRenderPass;
	
	void Initialize() {
	
	}

	void Animate(float seconds) override {

	}

	void BackBufferResizing() override {

	}

	void Render(nvrhi::IFramebuffer* framebuffer) override {

	}

private:
	nvrhi::ShaderHandle m_vertexShader, m_pixelShader;
	nvrhi::BufferHandle m_heightBuffer, m_xyBuffer, m_indexBuffer;
	nvrhi::InputLayoutHandle m_inputLayout;
	nvrhi::BindingLayoutHandle m_bindingLayout;
	nvrhi::BindingSetHandle m_bindingSets[4];
	nvrhi::GraphicsPipelineHandle m_pipeline;
	nvrhi::CommandListHandle m_commandList;
};

extern "C" int main(int argc, const char *argv[]) {
	Log(Info, "Hello from simple-rhi!");
	
	DeviceManager *deviceManager = DeviceManager::Create(nvrhi::GraphicsAPI::VULKAN);

	DeviceCreationParameters deviceParams	= {};
	deviceParams.enableDebugRuntime			= true;
	deviceParams.enableNvrhiValidationLayer = true;

	if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, "Sine Wave")) {
		logFatal("Cannot initialize a graphics device with the requested parameters");
		return 1;
	}

	{ 
		WaveRenderer app(deviceManager);
		app.Initialize();

		deviceManager->AddRenderPassToBack(&app);
		deviceManager->RunMessageLoop();
		deviceManager->RemoveRenderPass(&app);
	}

	deviceManager->Shutdown();
	delete deviceManager;
	
	exit(EXIT_SUCCESS);
}

KRR_NAMESPACE_END