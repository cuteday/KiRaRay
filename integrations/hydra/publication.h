#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace krr::hydra {

class PublicationSchedule {
public:
	using Clock = std::chrono::steady_clock;
	void reset() { *this = {}; }
	double intervalMs() const {
		return std::clamp(mSnapshotMs * snapshotCostMultiplier, minIntervalMs, maxIntervalMs);
	}
	bool due(uint64_t frames, uint64_t requested, Clock::time_point now) const {
		return frames > mSubmittedFrames &&
			(!mSubmittedFrames || frames >= requested ||
			 (elapsed(now - mSubmittedAt) && publicationDue(frames, requested, now)));
	}
	bool publicationDue(uint64_t frames, uint64_t requested, Clock::time_point now) const {
		return frames > mPublishedFrames &&
			(!mPublishedFrames || frames >= requested || elapsed(now - mPublishedAt));
	}
	void submitted(uint64_t frames, Clock::time_point now) {
		mSubmittedFrames = frames;
		mSubmittedAt = now;
	}
	bool needsWait(uint64_t frames) const {
		return frames > mSynchronizedFrames && frames - mSynchronizedFrames >= maxPendingFrames;
	}
	void synchronized(uint64_t frames) { mSynchronizedFrames = std::max(frames, mSynchronizedFrames); }
	void published(uint64_t frames, Clock::time_point now, double snapshotMs) {
		if (mPublishedFrames) {
			mSnapshotMs = mHasSnapshotCost ? (mSnapshotMs + snapshotMs) * 0.5 : snapshotMs;
			mHasSnapshotCost = true;
		}
		mPublishedFrames = frames;
		mPublishedAt = now;
		synchronized(frames);
	}

private:
	static constexpr double minIntervalMs = 75.0, maxIntervalMs = 500.0;
	static constexpr double snapshotCostMultiplier = 10.0;
	static constexpr uint64_t maxPendingFrames = 4;

	bool elapsed(Clock::duration duration) const {
		return duration >= std::chrono::duration<double, std::milli>(intervalMs());
	}
	uint64_t mSubmittedFrames{}, mPublishedFrames{}, mSynchronizedFrames{};
	Clock::time_point mSubmittedAt{}, mPublishedAt{};
	double mSnapshotMs{};
	bool mHasSnapshotCost{};
};

} // namespace krr::hydra
