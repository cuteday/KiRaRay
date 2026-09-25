#include <iostream>
#include <stdexcept>

#include "core/sampler.h"

using namespace krr;

void require(bool condition, const char *message) {
	if (!condition) throw std::runtime_error(message);
}

template <typename T> void testSampler() {
	T first, repeat, otherSeed, otherPixel, otherSample;
	first.setPixelSample(Vector2ui(11, 29), 7, 123);
	repeat.setPixelSample(Vector2ui(11, 29), 7, 123);
	otherSeed.setPixelSample(Vector2ui(11, 29), 7, 124);
	otherPixel.setPixelSample(Vector2ui(12, 29), 7, 123);
	otherSample.setPixelSample(Vector2ui(11, 29), 8, 123);
	bool seedDiffers = false, pixelDiffers = false, sampleDiffers = false;
	for (int i = 0; i < 4096; ++i) {
		float value = first.get1D();
		require(value >= 0 && value < 1, "Sampler output is outside [0, 1)");
		require(value == repeat.get1D(), "Sampler sequence is not repeatable");
		seedDiffers |= value != otherSeed.get1D();
		pixelDiffers |= value != otherPixel.get1D();
		sampleDiffers |= value != otherSample.get1D();
	}
	require(seedDiffers && pixelDiffers && sampleDiffers, "Sampler streams are not distinct");
	T defaults, zeroSeed;
	defaults.setPixelSample(Vector2ui(3, 5), 13);
	zeroSeed.setPixelSample(Vector2ui(3, 5), 13, 0);
	for (int i = 0; i < 32; ++i)
		require(defaults.get1D() == zeroSeed.get1D(), "Seed zero changed the default sequence");
}

int main() {
	try {
		testSampler<PCGSampler>();
		testSampler<LCGSampler>();
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what() << '\n';
		return 1;
	}
}
