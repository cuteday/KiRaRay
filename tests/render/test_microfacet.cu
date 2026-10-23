#include <iostream>
#include "device/context.h"
#include "render/bsdf.h"

using namespace krr;

constexpr int resolution = 512;
constexpr int sampleCount = resolution * resolution;

struct Result {
	double ndf{}, density{}, expectedX{}, expectedZ{}, sampledX{}, sampledZ{};
	double sampledEnergy{}, integratedEnergy{};
	int accepted{}, invalid{};
};

KRR_CALLABLE bool finite(Vector3f v) {
	return isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]);
}

template<class Distribution>
__global__ void measureDistribution(Vector3f wo, Result *output) {
	Distribution distribution(.35f, .65f);
	Result result;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sampleCount; i += blockDim.x * gridDim.x) {
		float z = (i / resolution + .5f) / resolution;
		float phi = M_2PI * (i % resolution + .5f) / resolution;
		Vector3f wh(sqrtf(1 - z * z) * cosf(phi), sqrtf(1 - z * z) * sinf(phi), z);
		float ndf = distribution.D(wh);
		result.ndf += ndf * z * M_2PI / sampleCount;
		if (wo[2] < 0) wh = -wh;
		float pdf = dot(wo, wh) > 0 ? distribution.Pdf(wo, wh) : 0;
		if (!isfinite(ndf) || ndf < 0 || !isfinite(pdf) || pdf < 0) ++result.invalid;
		result.density += pdf * M_2PI / sampleCount;
		result.expectedX += wh[0] * pdf * M_2PI / sampleCount;
		result.expectedZ += wh[2] * pdf * M_2PI / sampleCount;
		PCGSampler random;
		random.setSeed(i + 27, 1);
		Sampler sampler(&random);
		Vector3f sampled = distribution.Sample(wo, sampler.get2D());
		float sampledPdf = distribution.Pdf(wo, sampled);
		if (!finite(sampled) || abs(sampled.squaredNorm() - 1) > 2e-5f || dot(wo, sampled) <= 0 ||
			!isfinite(sampledPdf) || sampledPdf <= 0) ++result.invalid;
		result.sampledX += sampled[0] / sampleCount;
		result.sampledZ += sampled[2] / sampleCount;
	}
	atomicAdd(&output->ndf, result.ndf);
	atomicAdd(&output->density, result.density);
	atomicAdd(&output->expectedX, result.expectedX);
	atomicAdd(&output->expectedZ, result.expectedZ);
	atomicAdd(&output->sampledX, result.sampledX);
	atomicAdd(&output->sampledZ, result.sampledZ);
	atomicAdd(&output->invalid, result.invalid);
}

template<class BSDF>
__global__ void measureBsdf(BSDF bsdf, Vector3f wo, MicrofacetType type, Result *output) {
	Result result;
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < sampleCount; i += blockDim.x * gridDim.x) {
		PCGSampler random;
		random.setSeed(i + 29, 3);
		Sampler sampler(&random);
		auto sample = bsdf.sample(wo, sampler, TransportMode::Importance, type);
		if (!isfinite(sample.pdf) || sample.pdf < 0 || !sample.f.isFinite().all() || sample.f.minCoeff() < 0)
			++result.invalid;
		if (sample.pdf > 0) {
			++result.accepted;
			auto eval = bsdf.eval(wo, sample.wi, TransportMode::Importance, type);
			if (!finite(sample.wi) || !isfinite(sample.pdf) || !sample.f.isFinite().all() ||
				abs(eval.pdf - sample.pdf) > 2e-3f * max(1.f, sample.pdf) ||
				(eval.f - sample.f).abs().maxCoeff() > 2e-3f * max(1.f, sample.f.maxCoeff()))
				++result.invalid;
			result.sampledEnergy += sample.f.mean() * abs(sample.wi[2]) / (sample.pdf * sampleCount);
		}
		float z = 2 * (i / resolution + .5f) / resolution - 1;
		float phi = M_2PI * (i % resolution + .5f) / resolution;
		Vector3f wi(sqrtf(1 - z * z) * cosf(phi), sqrtf(1 - z * z) * sinf(phi), z);
		auto eval = bsdf.eval(wo, wi, TransportMode::Importance, type);
		if (!isfinite(eval.pdf) || eval.pdf < 0 || !eval.f.isFinite().all() || eval.f.minCoeff() < 0)
			++result.invalid;
		result.density += eval.pdf * (2 * M_2PI) / sampleCount;
		result.integratedEnergy += eval.f.mean() * abs(z) * (2 * M_2PI) / sampleCount;
	}
	atomicAdd(&output->accepted, result.accepted);
	atomicAdd(&output->invalid, result.invalid);
	atomicAdd(&output->density, result.density);
	atomicAdd(&output->sampledEnergy, result.sampledEnergy);
	atomicAdd(&output->integratedEnergy, result.integratedEnergy);
}

KRR_CALLABLE void checkBoundaries(Result *output) {
	BeckmannMicrofacetDistribution distribution(.2f, .8f);
	const Vector2f samples[] = {Vector2f(0, 0), Vector2f(0, 1), Vector2f(1, 0), Vector2f(1, 1), Vector2f(.5f, .5f)};
	const Vector3f directions[] = {Vector3f(0, 0, 1), Vector3f(1, 0, 0), normalize(Vector3f(1, 1, 1e-7f)),
		normalize(Vector3f(1, 1, -1e-7f))};
	for (const auto &wo : directions) {
		for (const auto &u : samples) {
			Vector3f wh = distribution.Sample(wo, u);
			if (!finite(wh) || dot(wo, wh) <= 0 || !isfinite(distribution.Pdf(wo, wh)) ||
				!isfinite(distribution.G1(wo)) || !isfinite(distribution.G(wo, wh))) {
				printf("Boundary cos=%g u=(%g,%g) normal=(%g,%g,%g) dot=%g PDF=%g G1=%g G=%g\n",
					wo[2], u[0], u[1], wh[0], wh[1], wh[2], dot(wo, wh), distribution.Pdf(wo, wh),
					distribution.G1(wo), distribution.G(wo, wh));
				++output->invalid;
			}
		}
	}
	if (distribution.D(Vector3f(1, 0, 0)) != 0 || distribution.G1(Vector3f(1, 0, 0)) != 0) ++output->invalid;
	const float cosines[] = {0, 1e-6f, .01f, .5f, .9f, .9998f};
	const float quantiles[] = {1e-6f, 1e-5f, .001f, .1f, .5f, .9f, .999f, 1 - 1e-6f};
	for (float c : cosines) {
		float s = sqrtf(1 - c * c), cot = c / s;
		Vector3f wo = normalize(Vector3f(s / .2f, 0, c));
		for (float u : quantiles) {
			Vector3f wh = distribution.Sample(wo, {u, .3f});
			double slope = -double(wh[0]) / (wh[2] * .2);
			double normalization = c * (1 + erf(double(cot))) + s * exp(-double(cot) * cot) / sqrt(double(M_PI));
			double cdf = (c * (1 + erf(slope)) + s * exp(-slope * slope) / sqrt(double(M_PI))) / normalization;
			if (!isfinite(cdf) || abs(cdf - u) > 8e-6) {
				printf("CDF cos=%g quantile=%g actual=%g slope=%g\n", c, u, cdf, slope);
				++output->invalid;
			}
		}
	}
	for (MicrofacetType type : {MicrofacetType::GGX, MicrofacetType::Beckmann}) {
		ConductorBsdf mirror(Spectrum(.8f), 0, 0);
		DielectricBsdf glass(Spectrum(.8f), 1.5f, 0, 0);
		DielectricBsdf matched(Spectrum(.8f), 1, .2f, .8f);
		PCGSampler random;
		random.setSeed(31, 5);
		Sampler sampler(&random);
		Vector3f wo = normalize(Vector3f(.95f, 0, -.1f));
		for (int i = 0; i < 64; ++i) {
			auto reflection = mirror.sample(wo, sampler, TransportMode::Importance, type);
			auto tir = glass.sample(wo, sampler, TransportMode::Importance, type);
			auto null = matched.sample(wo, sampler, TransportMode::Importance, type);
			if (!reflection.isDelta() || reflection.pdf != 1 || !SameHemisphere(wo, reflection.wi) ||
				!tir.isDelta() || tir.pdf != 1 || !SameHemisphere(wo, tir.wi) ||
				!null.isDelta() || null.pdf != 1 || (null.wi + wo).norm() > 1e-6f ||
				!reflection.f.isFinite().all() || !tir.f.isFinite().all() || !null.f.isFinite().all()) ++output->invalid;
		}
	}
}

__global__ void boundaries(Result *output) { checkBoundaries(output); }

void synchronize() {
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
	Result *result = nullptr;
	try {
		Result host;
		checkBoundaries(&host);
		if (host.invalid) throw std::runtime_error("CPU microfacet boundary/delta/TIR check failed");
		Context::ensureInitialized();
		CUDA_CHECK(cudaMallocManaged(&result, sizeof(Result)));
		*result = {};
		boundaries<<<1, 1>>>(result);
		synchronize();
		if (result->invalid) throw std::runtime_error("Microfacet boundary/delta/TIR check failed");
		const Vector3f directions[] = {Vector3f(0, 0, 1), normalize(Vector3f(.7f, .2f, .4f)),
			normalize(Vector3f(1, .3f, .01f)), normalize(Vector3f(.7f, .2f, -.4f))};
		for (MicrofacetType type : {MicrofacetType::GGX, MicrofacetType::Beckmann}) {
			for (const auto &wo : directions) {
				*result = {};
				if (type == MicrofacetType::GGX) measureDistribution<GGXMicrofacetDistribution><<<64, 128>>>(wo, result);
				else measureDistribution<BeckmannMicrofacetDistribution><<<64, 128>>>(wo, result);
				synchronize();
				std::cerr << "Distribution " << int(type) << " cos " << wo[2] << ": NDF " << result->ndf
					<< " PDF " << result->density << " invalid " << result->invalid << '\n';
				if (result->invalid || abs(result->ndf - 1) > .002 || abs(result->density - 1) > .002 ||
					abs(result->sampledX - result->expectedX) > .006 || abs(result->sampledZ - result->expectedZ) > .006)
					throw std::runtime_error("Microfacet normalization or visible-normal sampling failed");
			}
			for (int model = 0; model < 2; ++model) {
				for (float side : {-1.f, 1.f}) {
					*result = {};
					Vector3f wo = normalize(Vector3f(.6f, .2f, side));
					if (model == 0) measureBsdf<<<64, 128>>>(ConductorBsdf(Spectrum(.8f), .35f, .65f), wo, type, result);
					else measureBsdf<<<64, 128>>>(DielectricBsdf(Spectrum(.8f), 1.5f, .35f, .65f), wo, type, result);
					synchronize();
					std::cerr << "BSDF " << model << " distribution " << int(type) << " side " << side
						<< ": energy " << result->sampledEnergy << " integral " << result->integratedEnergy
						<< " PDF " << result->density << " accepted " << result->accepted << " invalid " << result->invalid << '\n';
					if (result->invalid || !result->accepted || !std::isfinite(result->sampledEnergy) ||
						result->sampledEnergy < 0 || result->sampledEnergy > .81 ||
						abs(result->sampledEnergy - result->integratedEnergy) > .02 ||
						abs(double(result->accepted) / sampleCount - result->density) > .01)
						throw std::runtime_error("Microfacet BSDF sampling/evaluation failed");
				}
			}
		}
		cudaFree(result);
		std::cerr << "Microfacet distribution GPU validation passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		cudaFree(result);
		return 1;
	}
}
