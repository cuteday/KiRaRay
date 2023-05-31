#include "animation.h"
#include "logger.h"

KRR_NAMESPACE_BEGIN
namespace anime {

Array4f interpolate(const InterpolationMode mode, const Keyframe &a, const Keyframe &b,
					const Keyframe &c, const Keyframe &d, const float t, const float dt) {
	switch (mode) {
		case InterpolationMode::Step:
			return b.value;

		case InterpolationMode::Linear:
			return lerp(b.value, c.value, t);

		case InterpolationMode::Slerp: {
			Quaternionf qb(b.value.w(), b.value.x(), b.value.y(), b.value.z());
			Quaternionf qc(c.value.w(), c.value.x(), c.value.y(), c.value.z());
			Quaternionf qr = qb.slerp(t, qc);
			return Array4f(qr.x(), qr.y(), qr.z(), qr.w());
		}

		case InterpolationMode::CatmullRomSpline: {
			// https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_the_unit_interval_with_matched_derivatives_at_endpoints
			// a = p[n-1], b = p[n], c = p[n+1], d = p[n+2]
			Array4f i = -a.value + 3.f * b.value - 3.f * c.value + d.value;
			Array4f j = 2.f * a.value - 5.f * b.value + 4.f * c.value - d.value;
			Array4f k = -a.value + c.value;
			return 0.5f * ((i * t + j) * t + k) * t + b.value;
		}

		case InterpolationMode::HermiteSpline: {
			// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-c-spline-interpolation
			const float t2 = t * t;
			const float t3 = t2 * t;
			return (2.f * t3 - 3.f * t2 + 1.f) * b.value + (t3 - 2.f * t2 + t) * b.outTangent * dt +
				   (-2.f * t3 + 3.f * t2) * c.value + (t3 - t2) * c.inTangent * dt;
		}

		default:
			assert(!"Unknown interpolation mode");
			return b.value;
	}
}

std::optional<Array4f> Sampler::evaluate(float time, bool extrapolateLastValues) const {
	const size_t count = mKeyframes.size();

	if (count == 0) return std::optional<Array4f>();

	if (time <= mKeyframes.front().time) return std::optional(mKeyframes.front().value);

	if (mKeyframes.size() == 1 || time >= mKeyframes.back().time) {
		if (extrapolateLastValues)
			return std::optional(mKeyframes.back().value);
		else
			return std::optional<Array4f>();
	}

	for (size_t offset = 0; offset < count; offset++) {
		const float tb = mKeyframes[offset].time;
		const float tc = mKeyframes[offset + 1].time;

		if (tb <= time && time < tc) {
			const Keyframe &b = mKeyframes[offset];
			const Keyframe &c = mKeyframes[offset + 1];
			const Keyframe &a = (offset > 0) ? mKeyframes[offset - 1] : b;
			const Keyframe &d = (offset < count - 2) ? mKeyframes[offset + 2] : c;
			const float dt	  = tc - tb;
			const float u	  = (time - tb) / dt;

			Array4f y = interpolate(mMode, a, b, c, d, u, dt);

			return std::optional(y);
		}
	}

	// shouldn't get here if the keyframes are properly ordered in time
	return std::optional<Array4f>();
}

void Sampler::addKeyframe(const Keyframe &keyframe) { mKeyframes.push_back(keyframe); }

float Sampler::getStartTime() const {
	if (!mKeyframes.empty()) return mKeyframes.front().time;
	return 0.f;
}

float Sampler::getEndTime() const {
	if (!mKeyframes.empty()) return mKeyframes.back().time;
	return 0.f;
}

Sampler::SharedPtr Sequence::getTrack(const std::string &name) {
	if (!mTracks.count(name)) {
		Log(Error, "The track %s does not exist in the sequence", name.c_str());
		return nullptr;
	}
	return mTracks[name];
}

std::optional<Array4f> Sequence::evaluate(const std::string &name, float time,
										  bool extrapolateLastValues) {
	auto track = getTrack(name);
	return track->evaluate(time, extrapolateLastValues);
}

void Sequence::addTrack(const std::string &name, Sampler::SharedPtr track) {
	mTracks[name] = track;
	mDuration	  = max(mDuration, track->getEndTime());
}

} // namespace anime
KRR_NAMESPACE_END