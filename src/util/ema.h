/* Exponential moving average */
#pragma once
#include <chrono>

#include "common.h"

NAMESPACE_BEGIN(krr)

class Ema {
public:
	enum class Type {
		Time,
		Step
	};

	Ema(Type type, float half_life)
	: mType{type}, mDecay{std::pow(0.5f, 1.0f / half_life)}, mCreationTime{std::chrono::steady_clock::now()} {}

	int64_t currentProgress() {
		if (mType == Type::Time) {
			auto now = std::chrono::steady_clock::now();
			return std::chrono::duration_cast<std::chrono::milliseconds>(now - mCreationTime).count();
		} else {
			return mLastProgress + 1;
		}
	}

	void update(float val) {
		int64_t cur = currentProgress();
		int64_t elapsed = cur - mLastProgress;
		mLastProgress = cur;

		float decay = std::pow(mDecay, elapsed);
		mVal = val;
		mEmaVal = decay * mEmaVal + (1.0f - decay) * val;
	}

	void set(float val) {
		mLastProgress = currentProgress();
		mVal = mEmaVal = val;
	}

	float val() const {
		return mVal;
	}

	float emaVal() const {
		return mEmaVal;
	}

private:
	float mVal = 0.0f;
	float mEmaVal = 0.0f;
	Type mType;
	float mDecay;

	int64_t mLastProgress = 0;
	std::chrono::time_point<std::chrono::steady_clock> mCreationTime;
};

NAMESPACE_END(krr)