#include <iostream>

#include "device/context.h"
#include "window.h"

using namespace krr;

int main() {
	DeviceManager device;
	try {
		Context::ensureInitialized();
		DeviceCreationParameters params;
		params.backBufferWidth = 3;
		params.backBufferHeight = 2;
		params.enableDebugRuntime = false;
		params.enableNvrhiValidationLayer = false;
		if (!device.createHeadlessDevice(params))
			throw std::runtime_error("Could not create the offscreen device.");
		if (device.getWindow() || device.getBackBufferCount() != 0)
			throw std::runtime_error("Headless initialization created presentation resources.");

		auto *context = device.getRenderContext();
		context->clear();
		auto target = context->getColorTexture()->getCudaRenderTarget();
		for (int frame = 0; frame < 64; ++frame) {
			context->sychronizeCuda();
			GPUParallelFor(6, [=] KRR_DEVICE(int i) mutable {
				int x = i % 3, y = i / 3;
				target.write(RGBA(float(x + 1), float(y + 4), float(i + frame), 1.f), i);
			}, context->getCudaStream());
			context->sychronizeVulkan();
		}
		auto pixels = context->readback();
		for (int y = 0; y < 2; ++y) {
			for (int x = 0; x < 3; ++x) {
				size_t offset = (y * 3 + x) * 3;
				int sourceY = 1 - y;
				if (pixels[offset] != float(x + 1) ||
					pixels[offset + 1] != float(sourceY + 4) ||
					pixels[offset + 2] != float(sourceY * 3 + x + 63))
					throw std::runtime_error("Readback changed channels or image orientation.");
			}
		}
		bool fatalRaised = false;
		try {
			device.getDevice()->getMessageCallback()->message(nvrhi::MessageSeverity::Fatal,
				"Injected fatal callback for error propagation test");
		} catch (const std::runtime_error &error) {
			fatalRaised = std::string(error.what()) == "Injected fatal callback for error propagation test";
		}
		if (!fatalRaised) throw std::runtime_error("NVRHI fatal messages must raise an exception.");
		device.shutdown();
		device.shutdown();
		return EXIT_SUCCESS;
	} catch (const std::exception &error) {
		std::cerr << error.what() << std::endl;
		device.shutdown();
		return EXIT_FAILURE;
	}
}
