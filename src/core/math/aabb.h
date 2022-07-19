#pragma once

#include "common.h"
#include "math/vec.h"

KRR_NAMESPACE_BEGIN
namespace math {
	
struct BoundingBox {
	KRR_CALLABLE BoundingBox() {}

	KRR_CALLABLE BoundingBox(const vec3f &a, const vec3f &b) : min{ a }, max{ b } {}

	KRR_CALLABLE void enlarge(const BoundingBox &other) {
		min = min.cwiseMin(other.min);
		max = max.cwiseMax(other.max);
	}

	KRR_CALLABLE void enlarge(const vec3f &point) {
		min = min.cwiseMin(point);
		max = max.cwiseMax(point);
	}

	KRR_CALLABLE void inflate(float amount) {
		min -= vec3f::Constant(amount);
		max += vec3f::Constant(amount);
	}

	KRR_CALLABLE vec3f diag() const { return max - min; }

	KRR_CALLABLE vec3f relative_pos(const vec3f &pos) const {
		return (pos - min).cwiseQuotient(diag());
	}

	KRR_CALLABLE vec3f center() const { return 0.5f * (max + min); }

	KRR_CALLABLE BoundingBox intersection(const BoundingBox &other) const {
		BoundingBox result = *this;
		result.min		   = result.min.cwiseMax(other.min);
		result.max		   = result.max.cwiseMin(other.max);
		return result;
	}

	KRR_CALLABLE bool intersects(const BoundingBox &other) const { return !intersection(other).is_empty(); }

	KRR_CALLABLE vec2f ray_intersect(const vec3f &pos, const vec3f &dir) const {
		float tmin = (min.x() - pos.x()) / dir.x();
		float tmax = (max.x() - pos.x()) / dir.x();

		if (tmin > tmax) {
			tcnn::host_device_swap(tmin, tmax);
		}

		float tymin = (min.y() - pos.y()) / dir.y();
		float tymax = (max.y() - pos.y()) / dir.y();

		if (tymin > tymax) {
			tcnn::host_device_swap(tymin, tymax);
		}

		if (tmin > tymax || tymin > tmax) {
			return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		}

		if (tymin > tmin) {
			tmin = tymin;
		}

		if (tymax < tmax) {
			tmax = tymax;
		}

		float tzmin = (min.z() - pos.z()) / dir.z();
		float tzmax = (max.z() - pos.z()) / dir.z();

		if (tzmin > tzmax) {
			tcnn::host_device_swap(tzmin, tzmax);
		}

		if (tmin > tzmax || tzmin > tmax) {
			return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
		}

		if (tzmin > tmin) {
			tmin = tzmin;
		}

		if (tzmax < tmax) {
			tmax = tzmax;
		}

		return { tmin, tmax };
	}

	KRR_CALLABLE bool is_empty() const { return (max.array() < min.array()).any(); }

	KRR_CALLABLE bool contains(const vec3f &p) const {
		return p.x() >= min.x() && p.x() <= max.x() && p.y() >= min.y() && p.y() <= max.y() && p.z() >= min.z() &&
			   p.z() <= max.z();
	}

	/// Calculate the squared point-AABB distance
	KRR_CALLABLE float distance(const vec3f &p) const { return sqrt(distance_sq(p)); }

	KRR_CALLABLE float distance_sq(const vec3f &p) const {
		return (min - p).cwiseMax(p - max).cwiseMax(0.0f).squaredNorm();
	}

	KRR_CALLABLE float signed_distance(const vec3f &p) const {
		vec3f q = (p - min).cwiseAbs() - diag();
		return q.cwiseMax(0.0f).norm() + std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
	}

	KRR_CALLABLE void get_vertices(vec3f v[8]) const {
		v[0] = { min.x(), min.y(), min.z() };
		v[1] = { min.x(), min.y(), max.z() };
		v[2] = { min.x(), max.y(), min.z() };
		v[3] = { min.x(), max.y(), max.z() };
		v[4] = { max.x(), min.y(), min.z() };
		v[5] = { max.x(), min.y(), max.z() };
		v[6] = { max.x(), max.y(), min.z() };
		v[7] = { max.x(), max.y(), max.z() };
	}

	vec3f min = vec3f::Constant(std::numeric_limits<float>::infinity());
	vec3f max = vec3f::Constant(-std::numeric_limits<float>::infinity());
};
	
}
KRR_NAMESPACE_END