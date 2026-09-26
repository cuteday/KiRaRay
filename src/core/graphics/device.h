#pragma once
#include <nvrhi/nvrhi.h>
#include <list>
#include "input.h"
#include "renderpass.h"
#include "graphics/rendercontext.h"

NAMESPACE_BEGIN(krr)


class DeviceManager;
class GraphicsBackend;
class CommonRenderPasses;
class BindingCache;

struct DeviceCreationParameters {
	nvrhi::GraphicsAPI graphicsApi = nvrhi::GraphicsAPI::VULKAN;
	bool startMaximized				= false;
	bool startFullscreen			= false;
	int windowPosX					= -1; // -1 means use default placement
	int windowPosY					= -1;
	uint32_t backBufferWidth		= 1280;
	uint32_t backBufferHeight		= 720;
	uint32_t refreshRate			= 0;
	uint32_t swapChainBufferCount	= 2;
	nvrhi::Format swapChainFormat	= nvrhi::Format::RGBA8_UNORM;	// not srgb, since we render in linear HDR.
	uint32_t maxFramesInFlight		= 2;
	bool enableDebugRuntime			= true;
	bool enableNvrhiValidationLayer = true;
	bool vsyncEnabled				= false;
	bool enableComputeQueue			= true;
	bool enableCopyQueue			= true;
	bool enablePerMonitorDPI		= false;
};


class DeviceManager {
public:
	DeviceManager();
	virtual ~DeviceManager();

	[[nodiscard]] nvrhi::IDevice *getDevice() const { return mNvrhiDevice; }
	[[nodiscard]] nvrhi::GraphicsAPI getGraphicsAPI() const { return mDeviceParams.graphicsApi; }

	bool createWindowDeviceAndSwapChain(const DeviceCreationParameters &params,
										const char *windowTitle);
	bool createHeadlessDevice(const DeviceCreationParameters &params);

	void addRenderPassToFront(RenderPass::SharedPtr pController);
	void addRenderPassToBack(RenderPass::SharedPtr pController);
	void removeRenderPass(RenderPass::SharedPtr pController);

	void runMessageLoop();

	// returns the size of the window in screen coordinates
	Vector2i getFrameSize() const;
	void getFrameSize(int &width, int &height) const;

	// returns the screen coordinate to pixel coordinate scale factor
	void getDPIScaleInfo(float &x, float &y) const {
		x = mDPIScaleFactorX;
		y = mDPIScaleFactorY;
	}

protected:
	bool mwindowVisible = false;
	bool mHeadless = false;
	bool mGlfwInitialized = false;

	DeviceCreationParameters mDeviceParams;
	GLFWwindow *mWindow = nullptr;
	std::list<RenderPass::SharedPtr> mRenderPasses;
	// timestamp in seconds for the previous frame
	double mPreviousFrameTimestamp = .0;
	// current DPI scale info (updated when window moves)
	float mDPIScaleFactorX = 1.f;
	float mDPIScaleFactorY = 1.f;
	bool mRequestedVSync   = false;

	double mAverageFrameTime		  = .0;
	double mAverageTimeUpdateInterval = .5;
	double mFrameTimeSum			  = .0;
	int mNumberOfAccumulatedFrames	  = 0;
	uint32_t mFrameIndex = 0;
	uint64_t mSeed = 0;

	nvrhi::DeviceHandle mNvrhiDevice;
	nvrhi::CommandListHandle mCommandList;

	std::vector<nvrhi::FramebufferHandle> mSwapChainFramebuffers;
	std::unique_ptr<CommonRenderPasses> mHelperPass;
	std::unique_ptr<BindingCache> mBindingCache;
	RenderContext::SharedPtr mRenderContext;

	void updateWindowSize();

	virtual void backBufferResizing();
	virtual void backBufferResized();

	virtual void tick(double elapsedTime);
	virtual void render();
	virtual void updateAverageFrameTime(double elapsedTime);

	// device-specific methods
	virtual bool createDeviceAndSwapChain();
	virtual void destroyDeviceAndSwapChain();
	virtual void resizeSwapChain();
	virtual bool beginFrame();
	virtual void present();

public:
	[[nodiscard]] virtual const char *getRendererString() const { return mRendererString.c_str(); }
	const DeviceCreationParameters &getDeviceParams() const { return mDeviceParams; }
	[[nodiscard]] double getAverageFrameTimeSeconds() const { return mAverageFrameTime; }
	[[nodiscard]] double getPreviousFrameTimestamp() const { return mPreviousFrameTimestamp; }
	void setFrameTimeUpdateInterval(double seconds) { mAverageTimeUpdateInterval = seconds; }
	[[nodiscard]] bool isVsyncEnabled() const { return mDeviceParams.vsyncEnabled; }
	virtual void setVsyncEnabled(bool enabled) {
		mRequestedVSync = enabled; /* will be processed later */
	}

	virtual Vector2f getMouseScale() {
		Vector2i fbSize;
		getFrameSize(fbSize[0], fbSize[1]);
		return fbSize.cast<float>().cwiseInverse();
	}
	inline Vector2i getMousePos() const {
		double x, y;
		glfwGetCursorPos(mWindow, &x, &y);
		return {(int) x, (int) y};
	}

	virtual void onWindowClose() {}
	virtual void onWindowIconify(int iconified) {}
	virtual void onWindowFocus(int focused) {}
	virtual void onWindowRefresh() {}
	virtual void onWindowPosUpdate(int xpos, int ypos);
	virtual bool onMouseEvent(io::MouseEvent &mouseEvent);
	virtual bool onKeyEvent(io::KeyboardEvent &keyEvent);

	[[nodiscard]] GLFWwindow *getWindow() const { return mWindow; }
	[[nodiscard]] bool isHeadless() const { return mHeadless; }
	[[nodiscard]] size_t getFrameIndex() const { return mFrameIndex; }
	void setFrameIndex(uint32_t frameIndex) { mFrameIndex = frameIndex; }
	[[nodiscard]] uint64_t getSeed() const { return mSeed; }
	void setSeed(uint64_t seed) { mSeed = seed; }
	[[nodiscard]] RenderContext *getRenderContext() const { return mRenderContext.get(); }

	virtual nvrhi::ITexture *getCurrentBackBuffer() const;
	virtual nvrhi::ITexture *getBackBuffer(size_t index) const;
	virtual nvrhi::ITexture *getRenderImage() const;
	virtual size_t getCurrentBackBufferIndex() const;
	virtual size_t getBackBufferCount() const;

	void shutdown();
	void setWindowTitle(const char *title);

	struct PipelineCallbacks {
		std::function<void(DeviceManager &)> beforeFrame   = nullptr;
		std::function<void(DeviceManager &)> beforeTick = nullptr;
		std::function<void(DeviceManager &)> afterTick  = nullptr;
		std::function<void(DeviceManager &)> beforeRender  = nullptr;
		std::function<void(DeviceManager &)> afterRender   = nullptr;
		std::function<void(DeviceManager &)> beforePresent = nullptr;
		std::function<void(DeviceManager &)> afterPresent  = nullptr;
	} mcallbacks;

private:
	std::unique_ptr<GraphicsBackend> mBackend;
	bool mFrameAcquired = false;
	std::string mRendererString;
	std::string mWindowTitle;
};

NAMESPACE_END(krr)
