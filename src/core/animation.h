#pragma once
#include <optional>

#include "common.h"
#include "krrmath/math.h"

KRR_NAMESPACE_BEGIN
namespace anime {

enum class AnimationAttribute {
	Undefined = 0,
	Scaling,
	Rotation,
	Translation,
};

struct Keyframe {
	float time		   = 0.f;
	Array4f value	   = 0.f;
	Array4f inTangent  = 0.f;
	Array4f outTangent = 0.f;
};

enum class InterpolationMode {
	Step,
	Linear,
	Slerp,
	CatmullRomSpline,
	HermiteSpline,
	Count
};

Array4f interpolate(InterpolationMode mode, const Keyframe &a, const Keyframe &b, const Keyframe &c,
					const Keyframe &d, float t1, float dt);

class Sampler {
public:
	using SharedPtr = std::shared_ptr<Sampler>;
	Sampler()		   = default;
	virtual ~Sampler() = default;

	std::optional<Array4f> evaluate(float time, bool extrapolateLastValues = false) const;
	void addKeyframe(const Keyframe& keyframe);

	InterpolationMode getInterpolationMode() const { return mMode; }
	void setInterpolationMode(InterpolationMode mode) { mMode = mode; }

	float getStartTime() const;
	float getEndTime() const;

protected:
	std::vector<Keyframe> mKeyframes;
	InterpolationMode mMode = InterpolationMode::Step;
};

class Sequence {
public:
	using SharedPtr = std::shared_ptr<Sequence>;
	Sequence()			= default;
	virtual ~Sequence() = default;

	float getDuration() const { return mDuration; }
	Sampler::SharedPtr getTrack(const std::string &name);
	std::optional<Array4f> evaluate(const std::string &name, float time, bool extrapolateLastValues = false);
	void addTrack(const std::string &name, Sampler::SharedPtr track);

protected:
	std::unordered_map<std::string, Sampler::SharedPtr> mTracks;
	float mDuration = 0.f;
};

}
KRR_NAMESPACE_END