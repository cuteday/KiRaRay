#pragma once
#include "common.h"
#include "file.h"
#include "texture.h"
#include "krrmath/math.h"
#include "device/buffer.h"

NAMESPACE_BEGIN(krr)

class Film {
public:
	using SharedPtr = std::shared_ptr<Film>;
	using Pixel = RGBA;
	using WeightedPixel = struct {Pixel pixel; float weight;};

	Film() = default;
	~Film() = default;
	KRR_HOST Film(size_t res_x, size_t res_y) {
		m_size = { (int) res_x, (int) res_y };
		m_data.resize(res_x * res_y);
		reset();
	}
	
	KRR_HOST Film(const Vector2f size) :
		Film(size[0], size[1]) {}

	KRR_CALLABLE WeightedPixel *data() { return m_data.data(); }

	KRR_CALLABLE Vector2i size() { return m_size; }

	KRR_HOST void reset(const Pixel &value = {}) {
		m_data.for_each([value] KRR_DEVICE(const WeightedPixel &c) -> WeightedPixel 
			{ return { value, 0 }; });
	}

	KRR_HOST TypedBuffer<WeightedPixel> &getInternalBuffer() { return m_data; }

	KRR_HOST void clear() { 
		m_size = {};
		m_data.clear();
	}

	KRR_HOST void resize(const Vector2i& size) {
		m_size = size;
		m_data.resize(size[0] * size[1]);
		reset();
	}

	KRR_CALLABLE void put(const Pixel &pixel, const size_t offset) {
		m_data[offset].pixel += pixel;
		m_data[offset].weight += 1.f;
	};

	KRR_CALLABLE void put(const Pixel &pixel, const Vector2i &pos) {
		size_t idx	= pos[0] + pos[1] * m_size[0];
		put(pixel, idx);
	}

	KRR_CALLABLE Pixel getPixel(const size_t offset) {
		const WeightedPixel &pixel = m_data[offset];
		return pixel.pixel / pixel.weight;
	}

	KRR_CALLABLE Pixel getPixel(const Vector2i &pos) {
		size_t idx = pos[0] + pos[1] * m_size[0];
		return getPixel(idx);
	}

	KRR_HOST void save(const fs::path& filepath) {
		size_t n_pixels = m_size[0] * m_size[1]; 
		CUDABuffer tmp(n_pixels * sizeof(Pixel));
		Pixel *pixels_device = reinterpret_cast<Pixel *>(tmp.data());
#ifdef __NVCC__
		thrust::transform(thrust::device, m_data.data(), m_data.data() + n_pixels, pixels_device,
						  [] KRR_DEVICE(const WeightedPixel &d) -> Pixel { return d.pixel / d.weight; });
#endif
		Image frame(m_size, Image::Format::RGBAfloat, false);
		tmp.copy_to_host(frame.data(), n_pixels * sizeof(RGBA));
		frame.saveImage(filepath, true);
	}

private:
	TypedBuffer<WeightedPixel> m_data;
	Vector2i m_size{};
};

NAMESPACE_END(krr)