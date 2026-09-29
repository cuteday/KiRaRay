#include <iostream>
#include "device/context.h"
#include "material/description.h"
#include "texture.h"
#include "render/bsdf.h"

using namespace krr;

constexpr int sampleCount = 262144;

struct Result {
	double sampled{}, integrated{}, density{}, expectedDeltaEnergy{};
	int accepted{}, delta{}, invalid{};
};

__global__ void measure(rt::MaterialData material, Result *output, int test) {
	Result result;
	SurfaceInteraction intr;
	intr.material = &material;
	intr.n = Vector3f(0, 0, 1);
	intr.tangent = Vector3f(1, 0, 0);
	intr.bitangent = Vector3f(0, 1, 0);
	intr.uv = {.25f, .75f};
	intr.lambda = SampledWavelengths::sampleUniform(.37f);
	intr.sd.bsdfType = MaterialType::Composite;
	MaterialContext context;
	context.uv = MaterialValue(.25f, .75f, 0);
	prepareCompositeFlags(intr.sd, material.mProgram, context);
	CompositeBsdf bsdf;
	bsdf.setup(intr);
	Vector3f wo = normalize(Vector3f(.3f, .1f, test == 9 || test == 12 ? -1.f : 1.f));
	TransportMode mode = test == 9 || test >= 11 ? TransportMode::Radiance : TransportMode::Importance;
	bool deltaCase = test == 3 || test == 7 || test >= 10;
	if (deltaCase && threadIdx.x == 0) {
		Spectrum diffuse = Spectrum::fromRGB(RGB(.2f), SpectrumType::RGBBounded,
			intr.lambda, *material.getColorSpace());
		Spectrum delta = Spectrum::fromRGB(RGB(.8f), SpectrumType::RGBBounded,
			intr.lambda, *material.getColorSpace());
		if (test == 3 || test == 10) {
			Spectrum reflectance = delta.cwiseMax(0.f).cwiseMin(.9999f);
			Spectrum extinction = 2 * reflectance.sqrt() / (Spectrum(1) - reflectance).sqrt();
			delta = FrComplex(fabsf(wo[2]), Spectrum(1), extinction);
		} else if (test >= 11) {
			float fresnel = FrDielectric(wo[2], 1.5f), eta = wo[2] < 0 ? 1 / 1.5f : 1.5f;
			delta *= fresnel + (1 - fresnel) / (eta * eta);
		}
		float a = test == 10 ? 1.f : .25f, b = test == 10 ? 1.f : .75f;
		output->expectedDeltaEnergy = (a * diffuse + b * delta).mean();
	}
	PCGSampler random;
	random.setSeed(1234 + threadIdx.x, 9);
	Sampler sampler(&random);
	constexpr int samples = sampleCount;
	for (int index = threadIdx.x; index < samples; index += blockDim.x) {
		auto sample = bsdf.sample(wo, sampler, mode);
		if (sample.pdf > 0) {
			++result.accepted;
			if (!sample.f.isFinite().all() || !isfinite(sample.pdf) ||
				sample.f.minCoeff() < 0 || fabsf(sample.wi.norm() - 1) > 1e-4f)
				++result.invalid;
			result.sampled += (sample.f * fabsf(sample.wi[2]) / sample.pdf).mean() / samples;
			if (sample.isDelta()) {
				++result.delta;
			} else {
				auto value = bsdf.eval(wo, sample.wi, mode);
				if (fabsf(sample.pdf - value.pdf) > 3e-4f * fmaxf(1, value.pdf) ||
					(sample.f - value.f).abs().maxCoeff() > 3e-4f * fmaxf(1, value.f.maxCoeff()))
					++result.invalid;
			}
		}
		float z = 1 - 2 * sampler.get1D(), phi = 2 * M_PI * sampler.get1D();
		float radius = sqrtf(fmaxf(0, 1 - z * z));
		Vector3f wi(radius * cosf(phi), radius * sinf(phi), z);
		auto value = bsdf.eval(wo, wi, mode);
		result.integrated += value.f.mean() * fabsf(z) * (4 * M_PI) / samples;
		result.density += value.pdf * (4 * M_PI) / samples;
		if (test <= 1 || test == 8) {
			float a = test == 1 ? 1 : .25f, b = test == 1 ? 1 : .75f;
			Spectrum expected = a * Spectrum::fromRGB(RGB(.2f), SpectrumType::RGBBounded,
				intr.lambda, *material.getColorSpace()) +
				b * Spectrum::fromRGB(RGB(.8f), SpectrumType::RGBBounded,
					intr.lambda, *material.getColorSpace());
			expected *= z > 0 ? M_INV_PI : 0;
			if ((value.f - expected).abs().maxCoeff() > 1e-5f ||
				fabsf(value.pdf - fmaxf(z, 0) * M_INV_PI) > 1e-5f) ++result.invalid;
		}
	}
	if (deltaCase && !(bsdf.flags() & BSDF_DELTA)) ++result.invalid;
	if (!(bsdf.flags() & BSDF_SMOOTH)) ++result.invalid;
	atomicAdd(&output->sampled, result.sampled);
	atomicAdd(&output->integrated, result.integrated);
	atomicAdd(&output->density, result.density);
	atomicAdd(&output->accepted, result.accepted);
	atomicAdd(&output->delta, result.delta);
	atomicAdd(&output->invalid, result.invalid);
}

std::shared_ptr<MaterialDescription> description(int test) {
	using P = MaterialParameter;
	auto result = std::make_shared<MaterialDescription>();
	result->model = MaterialModel::Composite;
	for (int index = 0; index < 2; ++index) {
		MaterialComponent component;
		component.model = MaterialModel::Diffuse;
		if (index == 1) {
			if (test == 2 || test == 3 || test == 10) component.model = MaterialModel::Conductor;
			if (test == 4 || test == 7 || test == 9 || test >= 11) component.model = MaterialModel::Dielectric;
			if (test == 6) component.model = MaterialModel::OpenPBR;
		}
		float weight = test == 1 || test == 10 ? 1 : index == 0 ? .25f : .75f;
		component.set(P::Weight, result->constant(MaterialValue(weight)));
		component.set(P::BaseColor, result->constant(MaterialValue(index ? .8f : .2f), MaterialValueType::Color3));
		component.set(P::SpecularRoughness, result->constant(MaterialValue(test == 3 || test >= 10 ? 0 : .65f)));
		component.set(P::SpecularIor, result->constant(MaterialValue(test == 7 ? 1 : 1.5f)));
		if (test == 5) component.set(P::Normal, result->constant(
			MaterialValue(index ? -.3f : .4f, 0, index ? .9539392f : .9165151f), MaterialValueType::Vector3));
		result->components.push_back(component);
	}
	if (test == 8) {
		MaterialNode uv;
		uv.op = MaterialOp::UV;
		uv.type = MaterialValueType::Vector2;
		int id = result->add(uv);
		MaterialNode channel;
		channel.op = MaterialOp::Extract;
		channel.inputs[0] = id;
		id = result->add(channel);
		result->components[0].set(P::Weight, id);
		MaterialNode inverse;
		inverse.op = MaterialOp::Subtract;
		inverse.inputs[0] = result->constant(MaterialValue(1));
		inverse.inputs[1] = id;
		result->components[1].set(P::Weight, result->add(inverse));
	}
	return result;
}

__global__ void cubic(rt::TextureData texture, float *output) {
	int i = threadIdx.x;
	float x = 1.1f + i * .23f;
	RGBA value = texture.evaluate({(x + .5f) / 4, 2.f / 4});
	output[i] = fabsf(value[0] - (x + 3.f));
}

void checkCubic() {
	auto texture = std::make_shared<Texture>();
	texture->mImage = std::make_shared<Image>(Vector2i(4, 4), Image::Format::RGBAfloat, false);
	auto *pixels = reinterpret_cast<float *>(texture->mImage->data());
	for (int y = 0; y < 4; ++y)
		for (int x = 0; x < 4; ++x)
			for (int c = 0; c < 4; ++c) pixels[(y * 4 + x) * 4 + c] = x + 2.f * y;
	MaterialTexture sampling;
	sampling.filter = MaterialFilter::Cubic;
	rt::TextureData data;
	float *errors = nullptr;
	try {
		data.initializeFromHost(texture, &sampling);
		CUDA_CHECK(cudaMallocManaged(&errors, 4 * sizeof(float)));
		cubic<<<1, 4>>>(data, errors);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		for (int i = 0; i < 4; ++i)
			if (!std::isfinite(errors[i]) || errors[i] > 2e-5f)
				throw std::runtime_error("Cubic image sampling failed linear reconstruction at " +
					std::to_string(i) + ": " + std::to_string(errors[i]));
		cudaFree(errors);
		data.release();
	} catch (...) {
		cudaFree(errors);
		data.release();
		throw;
	}
}

int main() {
	Result *result = nullptr;
	auto material = std::make_shared<Material>();
	auto blob = std::make_shared<Blob>(sizeof(rt::MaterialData));
	auto *device = new (blob->data()) rt::MaterialData();
	try {
		Context::ensureInitialized();
		checkCubic();
		CUDA_CHECK(cudaMallocManaged(&result, sizeof(Result)));
		for (int test = 0; test < 13; ++test) {
			material->setDescription(description(test));
			device->getObjectData(material, blob, test == 0);
			*result = {};
			measure<<<1, 128>>>(*device, result, test);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());
			std::cerr << test << ": energy " << result->sampled << " integral " << result->integrated
				<< " density " << result->density << " accepted " << result->accepted << " delta " << result->delta
				<< " invalid " << result->invalid << '\n';
			if (result->invalid || !result->accepted || !std::isfinite(result->sampled))
				throw std::runtime_error("Composite sampling/evaluation mismatch");
			bool deltaCase = test == 3 || test == 7 || test >= 10;
			if (!deltaCase && fabs(result->sampled - result->integrated) > .035)
				throw std::runtime_error("Composite sample energy differs from integration");
			if (deltaCase && fabs(result->sampled - result->expectedDeltaEnergy) > .01)
				throw std::runtime_error("Composite delta throughput differs from analytic weighted energy");
			if (fabs(double(result->accepted - result->delta) / sampleCount - result->density) > .025)
				throw std::runtime_error("Composite PDF differs from sampling distribution");
			if (deltaCase && fabs(double(result->delta) / sampleCount - (test == 10 ? .5 : .75)) > .01)
				throw std::runtime_error("Composite delta selection probability is incorrect");
		}
		device->releaseProgram();
		cudaFree(result);
		std::cerr << "Composite material GPU validation passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		device->releaseProgram();
		cudaFree(result);
		return 1;
	}
}
