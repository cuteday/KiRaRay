#include "profiler.h"
#include "logger.h"



NAMESPACE_BEGIN(krr)

namespace {
	// With sigma = 0.98, then after 100 frames, a given value's contribution is down to ~1.7% of
	// the running average, which seems to provide a reasonable trade-off of temporal smoothing
	// versus setting in to a new value when something has changed.
	const float kSigma = 0.98f;

	// Size of the event history. The event history is keeping track of event times to allow
	// for computing statistics (min, max, mean, stddev) over the recent history.
	const size_t kMaxHistorySize = 512;
	const size_t kFrameTimeWindow = 10;
}

Profiler::Stats Profiler::Stats::compute(const float* data, size_t len){
	if (len == 0) return {};

	float min = std::numeric_limits<float>::max();
	float max = std::numeric_limits<float>::lowest();
	double sum = 0.0;
	double sum2 = 0.0;

	for (size_t i = 0; i < len; ++i) {
		float value = data[i];
		min = std::min(min, value);
		max = std::max(max, value);
		sum += value;
		sum2 += value * value;
	}

	double mean = sum / len;
	double mean2 = sum2 / len;
	double variance = mean2 - mean * mean;
	double stdDev = std::sqrt(variance);

	return { min, max, (float)mean, (float)stdDev };
}

// Profiler::Event
Profiler::Event::Event(const std::string& name)
	: mName(name)
	, mCpuTimeHistory(kMaxHistorySize, 0.f)
	, mGpuTimeHistory(kMaxHistorySize, 0.f)
{}

Profiler::Stats Profiler::Event::computeCpuTimeStats() const {
	return Stats::compute(mCpuTimeHistory.data(), mHistorySize);
}

Profiler::Stats Profiler::Event::computeGpuTimeStats() const {
	return Stats::compute(mGpuTimeHistory.data(), mHistorySize);
}

void Profiler::Event::start(uint32_t frameIndex) {
	if (++mTriggered > 1) {
		logWarning("Profiler event '" + mName + "' was triggered while it is already running. Nesting profiler events with the same name is disallowed and you should probably fix that. Ignoring the new call.");
		return;
	}

	auto& frameData = mFrameData[frameIndex % 2];

	// Update CPU time.
	frameData.cpuStartTime = CpuTimer::getCurrentTimePoint();

	// Update GPU time.
	assert(frameData.activeTimer == -1);
	assert(frameData.currentTimer <= frameData.gpuTimers.size());
	if (frameData.currentTimer == frameData.gpuTimers.size()) {
	    frameData.gpuTimers.push_back(GpuTimer());
	}
	frameData.activeTimer = frameData.currentTimer++;
	frameData.gpuTimers[frameData.activeTimer].begin();
}

void Profiler::Event::end(uint32_t frameIndex) {
	if (--mTriggered != 0) return;

	auto& frameData = mFrameData[frameIndex % 2];

	// Update CPU time.
	frameData.cpuTotalTime += (float)CpuTimer::calcDuration(frameData.cpuStartTime, CpuTimer::getCurrentTimePoint());

	// Update GPU time.
	assert(frameData.activeTimer >= 0);
	frameData.gpuTimers[frameData.activeTimer].end();
	frameData.activeTimer = -1;
}

void Profiler::Event::endFrame(uint32_t frameIndex) {
	// Update CPU/GPU time from last frame measurement.
	auto& frameData = mFrameData[(frameIndex + 1) % 2];
	mCpuTime = frameData.cpuTotalTime;
	mGpuTime = 0.f;
	for (size_t i = 0; i < frameData.currentTimer; ++i) 
	    mGpuTime += (float)frameData.gpuTimers[i].getElapsedTime();
	frameData.cpuTotalTime = 0.f;
	frameData.currentTimer = 0;

	// Update EMA.
	mCpuTimeAverage = mCpuTimeAverage < 0.f ? mCpuTime : (kSigma * mCpuTimeAverage + (1.f - kSigma) * mCpuTime);
	mGpuTimeAverage = mGpuTimeAverage < 0.f ? mGpuTime : (kSigma * mGpuTimeAverage + (1.f - kSigma) * mGpuTime);

	// Update history.
	mCpuTimeHistory[mHistoryWriteIndex] = mCpuTime;
	mGpuTimeHistory[mHistoryWriteIndex] = mGpuTime;
	mHistoryWriteIndex = (mHistoryWriteIndex + 1) % kMaxHistorySize;
	mHistorySize = std::min(mHistorySize + 1, kMaxHistorySize);

	mTriggered = 0;
}

Profiler::Capture::Capture(size_t reservedEvents, size_t reservedFrames)
	: mReservedFrames(reservedFrames) {
	// Speculativly allocate event record storage.
	mLanes.resize(reservedEvents * 2);
	for (auto& lane : mLanes) lane.records.reserve(reservedFrames);
}

Profiler::Capture::SharedPtr Profiler::Capture::create(size_t reservedEvents, size_t reservedFrames) {
	return SharedPtr(new Capture(reservedEvents, reservedFrames));
}

void Profiler::Capture::captureEvents(const std::vector<Event*>& events) {
	if (events.empty()) return;

	// Initialize on first capture.
	if (mEvents.empty()) {
		mEvents = events;
		mLanes.resize(mEvents.size() * 2);
		for (size_t i = 0; i < mEvents.size(); ++i) {
			auto& pEvent = mEvents[i];
			mLanes[i * 2].name = pEvent->getName() + "/cpuTime";
			mLanes[i * 2].records.reserve(mReservedFrames);
			mLanes[i * 2 + 1].name = pEvent->getName() + "/gpuTime";
			mLanes[i * 2 + 1].records.reserve(mReservedFrames);
		}
	}

	for (size_t i = 0; i < mEvents.size(); ++i) {
		auto& pEvent = mEvents[i];
		mLanes[i * 2].records.push_back(pEvent->getCpuTime());
		mLanes[i * 2 + 1].records.push_back(pEvent->getGpuTime());
	}

	++mFrameCount;
}

void Profiler::Capture::finalize() {
	assert(!mFinalized);
	for (auto& lane : mLanes) {
		lane.stats = Stats::compute(lane.records.data(), lane.records.size());
	}
	mFinalized = true;
}

// Profiler

void Profiler::startEvent(const std::string& name, Flags flags) {
	if (mEnabled && (flags & Flags::Internal) ) {
		// '/' is used as a "path delimiter", so it cannot be used in the event name.
		if (name.find('/') != std::string::npos) {
			logWarning("Profiler event names must not contain '/'. Ignoring this profiler event.");
			return;
		}

		mCurrentEventName = mCurrentEventName + "/" + name;

		Event* pEvent = getEvent(mCurrentEventName);
		assert(pEvent != nullptr);
		if (!mPaused) pEvent->start(mFrameIndex);

		if (std::find(mCurrentFrameEvents.begin(), mCurrentFrameEvents.end(), pEvent) == mCurrentFrameEvents.end()) {
			mCurrentFrameEvents.push_back(pEvent);
		}
	}
}

void Profiler::endEvent(const std::string& name, Flags flags) {
	if (mEnabled && (flags & Flags::Internal)) {
		// '/' is used as a "path delimiter", so it cannot be used in the event name.
		if (name.find('/') != std::string::npos) return;

		Event* pEvent = getEvent(mCurrentEventName);
		assert(pEvent != nullptr);
		if (!mPaused) pEvent->end(mFrameIndex);

		mCurrentEventName.erase(mCurrentEventName.find_last_of("/"));
	}
}

Profiler::Event* Profiler::getEvent(const std::string& name) {
	auto event = findEvent(name);
	return event ? event : createEvent(name);
}

void Profiler::endFrame() {
	if (mPaused) return;

	for (Event* pEvent : mCurrentFrameEvents) {
		pEvent->endFrame(mFrameIndex);
	}

	if (mCapture) mCapture->captureEvents(mCurrentFrameEvents);

	mLastFrameEvents = std::move(mCurrentFrameEvents);
	++mFrameIndex;
}

void Profiler::startCapture(size_t reservedFrames) {
	setEnabled(true);
	mCapture = Capture::create(mLastFrameEvents.size(), reservedFrames);
}

Profiler::Capture::SharedPtr Profiler::endCapture() {
	Capture::SharedPtr pCapture;
	std::swap(pCapture, mCapture);
	if (pCapture) pCapture->finalize();
	return pCapture;
}

bool Profiler::isCapturing() const {
	return mCapture != nullptr;
}

const Profiler::SharedPtr& Profiler::instancePtr() {
	static Profiler::SharedPtr pInstance;
	if (!pInstance) pInstance = std::make_shared<Profiler>();
	return pInstance;
}

Profiler::Event* Profiler::createEvent(const std::string& name) {
	auto pEvent = std::shared_ptr<Event>(new Event(name));
	mEvents.emplace(name, pEvent);
	return pEvent.get();
}

Profiler::Event* Profiler::findEvent(const std::string& name) {
	auto event = mEvents.find(name);
	return (event == mEvents.end()) ? nullptr : event->second.get();
}

NAMESPACE_END(krr)