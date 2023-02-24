#pragma once
#include <stack>
#include <unordered_map>
#include <memory>
#include <queue>
#include <cuda_runtime.h>

#include "common.h"
#include "host/timer.h"
#include "device/timer.h"

KRR_NAMESPACE_BEGIN

class Profiler {
public:
	using SharedPtr = std::shared_ptr<Profiler>;

	enum Flags {
		None = 0x0,
		Internal = 0x1,
		Default = Internal 
	};

	struct Stats {
		float min;
		float max;
		float mean;
		float stdDev;

		static Stats compute(const float* data, size_t len);
	};

	class Event
	{
	public:
		const std::string getName() const { return mName; }

		float getCpuTime() const { return mCpuTime; }
		float getGpuTime() const { return mGpuTime; }

		float getCpuTimeAverage() const { return mCpuTimeAverage; }
		float getGpuTimeAverage() const { return mGpuTimeAverage; }

		Stats computeCpuTimeStats() const;
		Stats computeGpuTimeStats() const;

	private:
		Event(const std::string& name);

		void start(uint32_t frameIndex);
		void end(uint32_t frameIndex);
		void endFrame(uint32_t frameIndex);

		std::string mName;                              ///< Nested event name.

		float mCpuTime = 0.0;                           ///< CPU time (previous frame).
		float mGpuTime = 0.0;                           ///< GPU time (previous frame).

		float mCpuTimeAverage = -1.f;                   ///< Average CPU time (negative value to signify invalid).
		float mGpuTimeAverage = -1.f;                   ///< Average GPU time (negative value to signify invalid).

		std::vector<float> mCpuTimeHistory;             ///< CPU time history (round-robin, used for computing stats).
		std::vector<float> mGpuTimeHistory;             ///< GPU time history (round-robin, used for computing stats).
		size_t mHistoryWriteIndex = 0;                  ///< History write index.
		size_t mHistorySize = 0;                        ///< History size.

		uint32_t mTriggered = 0;                        ///< Keeping track of nested calls to start().

		struct FrameData {
			CpuTimer::TimePoint cpuStartTime;           ///< Last event CPU start time.
			float cpuTotalTime = 0.0;                   ///< Total accumulated CPU time.

			std::vector<GpuTimer> gpuTimers;
			int currentTimer{ 0 };
			int activeTimer{ -1 };
		};
		FrameData mFrameData[2];                        ///< Double-buffered frame data to avoid GPU flushes.

		friend class Profiler;
	};

	class Capture {
	public:
		using SharedPtr = std::shared_ptr<Capture>;

		struct Lane {
			std::string name;
			Stats stats;
			std::vector<float> records;
		};

		size_t getFrameCount() const { return mFrameCount; }
		const std::vector<Lane>& getLanes() const { return mLanes; }

	private:
		Capture(size_t reservedEvents, size_t reservedFrames);

		static SharedPtr create(size_t reservedEvents, size_t reservedFrames);
		void captureEvents(const std::vector<Event*>& events);
		void finalize();

		size_t mReservedFrames;
		size_t mFrameCount = 0;
		std::vector<Event*> mEvents;
		std::vector<Lane> mLanes;
		bool mFinalized = false;

		friend class Profiler;
	};
	
	/** Check if the profiler is enabled.
		\return Returns true if the profiler is enabled.
	*/
	bool isEnabled() const { return mEnabled; }

	/** Enable/disable the profiler.
		\param[in] enabled True to enable the profiler.
	*/
	void setEnabled(bool enabled) { mEnabled = enabled; }

	/** Check if the profiler is paused.
		\return Returns true if the profiler is paused.
	*/
	bool isPaused() const { return mPaused; }

	/** Pause/resume the profiler.
		\param[in] paused True to pause the profiler.
	*/
	void setPaused(bool paused) { mPaused = paused; }

	/** Start profile capture.
		\param[in] reservedFrames Number of frames to reserve memory for.
	*/
	void startCapture(size_t reservedFrames = 1024);

	/** End profile capture.
		\return Returns the captured data.
	*/
	Capture::SharedPtr endCapture();

	/** Check if the profiler is capturing.
		\return Return true if the profiler is capturing.
	*/
	bool isCapturing() const;

	/** Finish profiling for the entire frame.
		Note: Must be called once at the end of each frame.
	*/
	void endFrame();

	/** Start profiling a new event and update the events hierarchies.
		\param[in] name The event name.
		\param[in] flags The event flags.
	*/
	void startEvent(const std::string& name, Flags flags = Flags::Default);

	/** Finish profiling a new event and update the events hierarchies.
		\param[in] name The event name.
		\param[in] flags The event flags.
	*/
	void endEvent(const std::string& name, Flags flags = Flags::Default);

	/** Get the event, or create a new one if the event does not yet exist.
		This is a public interface to facilitate more complicated construction of event names and finegrained control over the profiled region.
		\param[in] name The event name.
		\return Returns a pointer to the event.
	*/
	Event* getEvent(const std::string& name);

	/** Get the profiler events (previous frame).
	*/
	const std::vector<Event*>& getEvents() const { return mLastFrameEvents; }

	/** Global profiler instance pointer.
	*/
	static const Profiler::SharedPtr& instancePtr();

	/** Global profiler instance.
	*/
	static Profiler& instance() { return *instancePtr(); }

private:
	/** Create a new event.
		\param[in] name The event name.
		\return Returns the new event.
	*/
	Event* createEvent(const std::string& name);

	/** Find an event that was previously created.
		\param[in] name The event name.
		\return Returns the event or nullptr if none was found.
	*/
	Event* findEvent(const std::string& name);

	bool mEnabled = false;
	bool mPaused = false;

	std::unordered_map<std::string, std::shared_ptr<Event>> mEvents; ///< Events by name.
	std::vector<Event*> mCurrentFrameEvents;            ///< Events registered for current frame.
	std::vector<Event*> mLastFrameEvents;               ///< Events from last frame.
	std::string mCurrentEventName;                      ///< Current nested event name.
	uint32_t mCurrentLevel = 0;                         ///< Current nesting level.
	uint32_t mFrameIndex = 0;                           ///< Current frame index.

	Capture::SharedPtr mpCapture;                       ///< Currently active capture.
};

/** Helper class for starting and ending profiling events using RAII.
	The constructor and destructor call Profiler::StartEvent() and Profiler::EndEvent().
	The PROFILE macro wraps creation of local ProfilerEvent objects when profiling is enabled,
	and does nothing when profiling is disabled, so should be used instead of directly creating ProfilerEvent objects.
*/
class ProfilerEvent {
public:
	ProfilerEvent(const std::string& name, Profiler::Flags flags = Profiler::Flags::Default)
		: mName(name)
		, mFlags(flags) { 
		Profiler::instance().startEvent(mName, mFlags); 
	}

	~ProfilerEvent() { Profiler::instance().endEvent(mName, mFlags); }

private:
	const std::string mName;
	Profiler::Flags mFlags;
};

#if KRR_ENABLE_PROFILE
#define PROFILE_ALL_FLAGS(_name) krr::ProfilerEvent _profileEvent##__LINE__(_name)
#define PROFILE_SOME_FLAGS(_name, _flags) krr::ProfilerEvent _profileEvent##__LINE__(_name, _flags)

#define GET_PROFILE(_1, _2, NAME, ...) NAME
#define PROFILE(...) GET_PROFILE(__VA_ARGS__, PROFILE_SOME_FLAGS, PROFILE_ALL_FLAGS)(__VA_ARGS__)
#else 
#define PROFILE(_name)
#endif

KRR_NAMESPACE_END