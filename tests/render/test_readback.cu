#include <iostream>

#include "device/context.h"
#include "graphics/device.h"
#include "graphics/rendercontext.h"

using namespace krr;

void checkReadback(RenderContext *context, Vector2i size) {
	context->resize(size);
	context->clear();
	auto target = context->getColorTexture()->getCudaRenderTarget();
	auto *depthTexture = context->getRenderTarget()->getDepthTexture();
	auto depth = depthTexture ? depthTexture->getCudaRenderTarget() : CudaRenderTarget{};
	for (int frame = 0; frame < 64; ++frame) {
		context->beginCuda();
		GPUParallelFor(size[0] * size[1], [=] KRR_DEVICE(int i) mutable {
			int x = i % target.width, y = i / target.width;
			const float previous = depth ? depth.read(i)[0] : target.read(i)[2];
			const float accumulated = previous + float(i + 1);
			if (depth) depth.write(RGBA(accumulated, 0.f, 0.f, 1.f), i);
			target.write(RGBA(float(x + 1), float(y + 4), accumulated, 1.f), i);
		}, context->getCudaStream());
		context->endCuda();
	}
	auto pixels = context->readback();
	for (int y = 0; y < size[1]; ++y) {
		for (int x = 0; x < size[0]; ++x) {
			size_t offset = (y * size[0] + x) * 3;
			int sourceY = size[1] - 1 - y;
			if (pixels[offset] != float(x + 1) ||
				pixels[offset + 1] != float(sourceY + 4) ||
				pixels[offset + 2] != float(64 * (sourceY * size[0] + x + 1)))
				throw std::runtime_error("Readback changed channels, orientation, or accumulated values.");
		}
	}
}

void checkBuffer(RenderContext *context) {
	auto *device = context->getDevice();
	nvrhi::BufferDesc desc;
	desc.byteSize = 65 * sizeof(unsigned);
	desc.debugName = "Shared buffer test";
	CudaBufferMapping mapping(device, desc);
	auto *buffer = mapping.getBuffer();
	context->addSharedBuffer(buffer);
	try {
		auto command = device->createCommandList();
		std::vector<unsigned> initial(65, 7);
		command->open();
		command->writeBuffer(buffer, initial.data(), desc.byteSize);
		command->close();
		device->executeCommandList(command);
		auto *values = static_cast<unsigned *>(mapping.getPointer());
		for (int frame = 0; frame < 32; ++frame) {
			context->beginCuda();
			GPUParallelFor(65, [=] KRR_DEVICE(int i) { values[i] += unsigned(i + 1); },
				context->getCudaStream());
			context->endCuda();
		}
		nvrhi::BufferDesc readbackDesc;
		readbackDesc.byteSize = desc.byteSize;
		readbackDesc.cpuAccess = nvrhi::CpuAccessMode::Read;
		auto readback = device->createBuffer(readbackDesc);
		command->open();
		command->beginTrackingBufferState(readback, nvrhi::ResourceStates::CopyDest);
		command->copyBuffer(readback, 0, buffer, 0, desc.byteSize);
		command->close();
		device->executeCommandList(command);
		device->waitForIdle();
		const auto *data = static_cast<const unsigned *>(device->mapBuffer(readback, nvrhi::CpuAccessMode::Read));
		if (!data) throw std::runtime_error("Could not read the shared buffer");
		std::vector<unsigned> result(data, data + 65);
		device->unmapBuffer(readback);
		for (unsigned i = 0; i < result.size(); ++i)
			if (result[i] != 7 + 32 * (i + 1)) throw std::runtime_error("Shared buffer handoff changed data");
	} catch (...) {
		context->endCuda();
		cudaStreamSynchronize(context->getCudaStream());
		device->waitForIdle();
		context->removeSharedBuffer(buffer);
		throw;
	}
	context->removeSharedBuffer(buffer);
}

int main(int argc, char **argv) {
	DeviceManager device;
	try {
		Context::ensureInitialized();
		DeviceCreationParameters params;
		const std::string api = argc > 1 ? argv[1] : "vulkan";
		if (api != "vulkan" && api != "d3d12") throw std::invalid_argument("Unknown graphics API");
		params.graphicsApi = api == "d3d12" ? nvrhi::GraphicsAPI::D3D12 : nvrhi::GraphicsAPI::VULKAN;
		params.backBufferWidth = 3;
		params.backBufferHeight = 2;
		if (!device.createHeadlessDevice(params))
			throw std::runtime_error("Could not create the offscreen device.");
		if (device.getWindow() || device.getBackBufferCount() != 0)
			throw std::runtime_error("Headless initialization created presentation resources.");

		for (const Vector2i size : {Vector2i(3, 2), Vector2i(7, 5), Vector2i(3, 2)})
			checkReadback(device.getRenderContext(), size);
		device.getRenderContext()->getRenderTarget()->setDepthEnabled(true);
		checkReadback(device.getRenderContext(), Vector2i(3, 2));
		checkBuffer(device.getRenderContext());
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
