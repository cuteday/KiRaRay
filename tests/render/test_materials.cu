#include <iostream>
#include "device/context.h"
#include "material/description.h"
#include "texture.h"
#include "render/bsdf.h"
#include "sampler.h"

using namespace krr;

__device__ bool checkAuthoredFlags(rt::MaterialProgramData program, MaterialContext context) {
	auto values = program.evaluate(context);
	BSDFData expected, actual;
	expected.bsdfType = actual.bsdfType = MaterialType::OpenPBR;
	expected.metallic = values[MaterialParameter::Metalness][0];
	expected.specularTransmission = values[MaterialParameter::TransmissionWeight][0];
	prepareAuthoredFlags(actual, program, context);
	return actual.metallic == expected.metallic &&
		actual.specularTransmission == expected.specularTransmission &&
		actual.getBsdfType() == expected.getBsdfType();
}

__global__ void evaluateGraph(rt::MaterialProgramData program, MaterialValues *results) {
	int pixel = threadIdx.x;
	MaterialContext context;
	context.uv		   = {pixel % 2 ? .75f : .25f, pixel / 2 ? .75f : .25f, 0};
	results[pixel]	   = program.evaluate(context);
	results[pixel + 4] = program.evaluate(context, rt::MaterialEvaluation::Opacity);
	results[pixel + 8] = program.evaluate(context, rt::MaterialEvaluation::Emission);
	bool flagsMatch = checkAuthoredFlags(program, context) &&
		checkAuthoredFlags(program, MaterialContext{});
	program.kind = MaterialProgramKind::Constant;
	program.defaults[MaterialParameter::Metalness] = MaterialValue(.3f);
	program.defaults[MaterialParameter::TransmissionWeight] = MaterialValue(.6f);
	results[pixel + 12][MaterialParameter::Opacity] =
		MaterialValue(flagsMatch && checkAuthoredFlags(program, context));
}

__global__ void evaluateSimpleEmission(rt::MaterialProgramData program, MaterialValues *output) {
	MaterialSimpleBinding bindings[] = {{MaterialParameter::BaseColor, 65535, -1, false},
										{MaterialParameter::Opacity, 65535, 3, false},
										{MaterialParameter::EmissionColor, 0, -1, true}};
	program.simple					 = bindings;
	program.simpleCount				 = 3;
	MaterialContext context;
	context.uv = MaterialValue(.25f, .25f, 0);
	*output	   = program.evaluate(context, rt::MaterialEvaluation::Emission);
}

__global__ void evaluateEmitter(rt::DiffuseAreaLight light, rt::DiffuseAreaLight legacy,
								rt::MaterialData *material, float *results) {
	SampledWavelengths lambda = SampledWavelengths::sampleUniform(.27f);
	Vector3f p(.25f, .25f, 0.f), n(0.f, 0.f, 1.f);
	Vector2f uv(.25f, .25f);
	RGB color;
	for (int channel = 0; channel < 3; ++channel) {
		float encoded  = (128 >> channel) / 255.f;
		color[channel] = powf((encoded + .055f) / 1.055f, 2.4f);
	}
	Spectrum expected =
		Spectrum::fromRGB(color, SpectrumType::RGBIlluminant, lambda, *material->getColorSpace()) *
		.002f;
	results[0] = (light.L(p, n, uv, n, lambda) - expected).abs().maxCoeff();
	results[1] = (light.L(p, n, uv, n, lambda, true) - expected * (64.f / 255)).abs().maxCoeff();
	results[2] = light.L(p, n, uv, -n, lambda).abs().maxCoeff();
	material->mProgram.defaults[MaterialParameter::ThinWalled] = MaterialValue(1);
	results[3]				 = (light.L(p, n, uv, -n, lambda) - expected).abs().maxCoeff();
	material->mProgram.model = MaterialModel::PreviewSurface;
	results[4] = (light.L(p, n, uv, n, lambda) - expected * 1000).abs().maxCoeff() / 1000;
	rt::Light legacyHandle(&legacy);
	results[5] = (legacyHandle.L<false>(p, n, uv, n, lambda) - legacyHandle.L(p, n, uv, n, lambda))
					 .abs()
					 .maxCoeff();
	rt::LightSampleContext context{Vector3f(.25f, .25f, 1.f), -n};
	Vector2f sample(.3f, .4f);
	results[6] = (legacyHandle.sampleLi<false>(sample, context, lambda).L -
				  legacyHandle.sampleLi(sample, context, lambda).L)
					 .abs()
					 .maxCoeff();
}

void checkEmitter(const rt::MaterialData &source) {
	struct Fixture {
		rt::MeshData mesh;
		rt::InstanceData instance;
		Triangle triangle;
		rt::MaterialData material;
		float results[7]{};
	};
	Fixture *fixture = nullptr;
	CUDA_CHECK(cudaMallocManaged(&fixture, sizeof(Fixture)));
	new (fixture) Fixture();
	auto release = [&] {
		fixture->mesh.positions.clear();
		fixture->mesh.normals.clear();
		fixture->mesh.texcoords.clear();
		fixture->mesh.indices.clear();
		cudaFree(fixture);
	};
	try {
		fixture->material = source;
		fixture->mesh.positions.alloc_and_copy_from_host(
			std::vector<Vector3f>{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}});
		fixture->mesh.normals.alloc_and_copy_from_host(
			std::vector<Vector3f>(3, Vector3f(0.f, 0.f, 1.f)));
		fixture->mesh.texcoords.alloc_and_copy_from_host(
			std::vector<Vector2f>{{0.f, 0.f}, {1.f, 0.f}, {0.f, 1.f}});
		fixture->mesh.indices.alloc_and_copy_from_host(std::vector<Vector3i>{{0, 1, 2}});
		fixture->instance.mesh = &fixture->mesh;
		fixture->triangle	   = Triangle(0, &fixture->instance);
		rt::DiffuseAreaLight light(Shape(&fixture->triangle), &fixture->material);
		rt::DiffuseAreaLight legacy(Shape(&fixture->triangle), RGB(.2f, .4f, .6f));
		evaluateEmitter<<<1, 1>>>(light, legacy, &fixture->material, fixture->results);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		for (float value : fixture->results)
			if (!std::isfinite(value) || value > 1e-5f)
				throw std::runtime_error(
					"Material emission units, texture, opacity or sidedness mismatch");
		release();
	} catch (...) {
		cudaDeviceSynchronize();
		release();
		throw;
	}
}

void checkPrograms() {
	auto texture		 = std::make_shared<Texture>();
	texture->mImage		 = std::make_shared<Image>(Vector2i(2, 2), Image::Format::RGBAuchar, true);
	const uchar pixels[] = {128, 64, 32, 64, 0, 255, 0, 128, 0, 0, 255, 192, 255, 255, 255, 255};
	memcpy(texture->mImage->data(), pixels, sizeof(pixels));
	auto description = std::make_shared<MaterialDescription>();
	MaterialTexture binding;
	binding.texture = texture;
	binding.filter	= MaterialFilter::Closest;
	description->textures.push_back(binding);
	auto add = [&](MaterialOp op, MaterialValueType type, std::initializer_list<int> inputs,
				   int auxiliary = 0) {
		MaterialNode node;
		node.op		   = op;
		node.type	   = type;
		node.auxiliary = auxiliary;
		std::copy(inputs.begin(), inputs.end(), node.inputs.begin());
		return description->add(node);
	};
	int uv	  = add(MaterialOp::UV, MaterialValueType::Vector2, {});
	int image = add(MaterialOp::Image, MaterialValueType::Color4, {uv});
	int color = add(MaterialOp::Convert, MaterialValueType::Color3, {image}, 4);
	int alpha = add(MaterialOp::Extract, MaterialValueType::Float, {image}, 3);
	int metal = add(MaterialOp::Extract, MaterialValueType::Float, {image}, 1);
	description->set(MaterialParameter::BaseColor, color);
	description->set(MaterialParameter::EmissionColor, color);
	description->set(MaterialParameter::EmissionLuminance, description->constant(MaterialValue(2)));
	description->set(MaterialParameter::Opacity, alpha);
	description->set(MaterialParameter::Metalness, metal);
	description->set(MaterialParameter::TransmissionWeight, alpha);
	auto material		   = std::make_shared<Material>();
	auto blob			   = std::make_shared<Blob>(sizeof(rt::MaterialData));
	auto *device		   = new (blob->data()) rt::MaterialData();
	MaterialValues *output = nullptr;
	CUDA_CHECK(cudaMallocManaged(&output, 16 * sizeof(MaterialValues)));
	try {
		for (int iteration = 0; iteration < 5; ++iteration) {
			if (iteration == 1) {
				MaterialNode uniform;
				uniform.op	  = MaterialOp::Uniform;
				uniform.value = MaterialValue(1);
				int weight	  = description->add(uniform);
				description->set(MaterialParameter::Metalness,
					add(MaterialOp::Multiply, MaterialValueType::Float, {metal, weight}));
				description->set(MaterialParameter::TransmissionWeight,
					add(MaterialOp::Multiply, MaterialValueType::Float, {alpha, weight}));
				int dynamicColor =
					add(MaterialOp::Multiply, MaterialValueType::Color3, {color, weight});
				description->set(MaterialParameter::BaseColor, dynamicColor);
				description->set(MaterialParameter::EmissionColor, dynamicColor);
				int n = add(MaterialOp::Normal, MaterialValueType::Vector3, {});
				int t = add(MaterialOp::Tangent, MaterialValueType::Vector3, {});
				int b = add(MaterialOp::Bitangent, MaterialValueType::Vector3, {});
				int normalColor =
					description->constant(MaterialValue(.75f, .5f, 1), MaterialValueType::Color3);
				description->set(MaterialParameter::Normal,
								 add(MaterialOp::NormalMap, MaterialValueType::Vector3,
									 {normalColor, weight, n, t, b}));
			}
			if (iteration == 2)
				description->set(MaterialParameter::Opacity,
								 description->constant(MaterialValue(1)));
			if (iteration == 3) {
				texture->mImage =
					std::make_shared<Image>(Vector2i(2, 2), Image::Format::RGBAfloat, true);
				auto *values = reinterpret_cast<float *>(texture->mImage->data());
				for (int i = 0; i < 16; ++i) values[i] = pixels[i] / 255.f;
				description->set(MaterialParameter::Opacity, alpha);
			}
			if (iteration == 4) {
				description->outputs.fill(-1);
				description->set(MaterialParameter::BaseColor, color);
				description->set(MaterialParameter::EmissionColor, color);
				int expanded = add(MaterialOp::Convert, MaterialValueType::Color4, {color});
				description->set(MaterialParameter::Opacity,
								 add(MaterialOp::Extract, MaterialValueType::Float, {expanded}, 3));
			}
			material->setDescription(description);
			device->getObjectData(material, blob, iteration == 0);
			if ((iteration == 0) != (device->mProgram.kind == MaterialProgramKind::Simple))
				throw std::runtime_error("Incorrect material fast-path selection");
			if (iteration == 2 && device->mProgram.opacity.count != 0)
				throw std::runtime_error("Constant opacity retained texture evaluation");
			if (iteration == 4 && device->mProgram.classification.count != 0)
				throw std::runtime_error("Constant BSDF flags retained surface evaluation");
			evaluateGraph<<<1, 4>>>(device->mProgram, output);
			CUDA_CHECK(cudaGetLastError());
			CUDA_CHECK(cudaDeviceSynchronize());
			for (int pixel = 0; pixel < 4; ++pixel) {
				if (output[pixel + 12][MaterialParameter::Opacity][0] != 1)
					throw std::runtime_error("Authored BSDF flags disagree with the full evaluator");
				for (int channel = 0; channel < 3; ++channel) {
					float encoded  = pixels[pixel * 4 + channel] / 255.f;
					float expected = encoded <= .04045f ? encoded / 12.92f
														: powf((encoded + .055f) / 1.055f, 2.4f);
					if (fabsf(output[pixel][MaterialParameter::BaseColor][channel] - expected) >
							.003f ||
						fabsf(output[pixel + 8][MaterialParameter::EmissionColor][channel] -
							  expected) > .003f)
						throw std::runtime_error(
							"Material image color/channel/orientation mismatch");
				}
				float expectedAlpha =
					iteration == 2 || iteration == 4 ? 1 : pixels[pixel * 4 + 3] / 255.f;
				if (fabsf(output[pixel + 4][MaterialParameter::Opacity][0] - expectedAlpha) > 1e-5f)
					throw std::runtime_error("Material opacity slice mismatch");
				if (iteration > 0 && iteration < 4 &&
					(fabsf(output[pixel][MaterialParameter::Normal][0] - .4472136f) > 1e-5f ||
					 fabsf(output[pixel][MaterialParameter::Normal][2] - .8944272f) > 1e-5f))
					throw std::runtime_error("Material tangent normal mismatch");
			}
			if (iteration == 0) checkEmitter(*device);
			if (iteration == 0) {
				auto compiled = compileMaterial(*description);
				for (const auto &binding : compiled.simple)
					if (binding.emission != (binding.parameter == MaterialParameter::EmissionColor))
						throw std::runtime_error(
							"Simple material emission dependencies are incorrect");
				evaluateSimpleEmission<<<1, 1>>>(device->mProgram, output);
				CUDA_CHECK(cudaGetLastError());
				CUDA_CHECK(cudaDeviceSynchronize());
				if (output[0][MaterialParameter::BaseColor][0] !=
					device->mProgram.defaults[MaterialParameter::BaseColor][0])
					throw std::runtime_error("Emission evaluated an unrelated surface binding");
			}
		}
		device->releaseProgram();
		cudaFree(output);
	} catch (...) {
		cudaDeviceSynchronize();
		device->releaseProgram();
		cudaFree(output);
		throw;
	}
}

struct Result {
	double sampled{0}, integrated{0}, pdfIntegral{0};
	double expected{0};
	int valid{0}, invalid{0}, transmitted{0};
	float maximum{0};
};

__device__ bool matchesEvaluation(BSDFEval actual, Spectrum value, float pdf) {
	return actual.f.isFinite().all() && isfinite(actual.pdf) &&
		(actual.f - value).abs().maxCoeff() <= 1e-5f * fmaxf(1, value.abs().maxCoeff()) &&
		fabsf(actual.pdf - pdf) <= 1e-5f * fmaxf(1, pdf);
}

__device__ bool checkCombinedBsdf(const MaterialValues &values, Vector3f wo,
	const SampledWavelengths &lambda, const RGBColorSpace *colorSpace, MaterialModel kind) {
	rt::MaterialData material;
	material.mProgram.enabled = true;
	material.mProgram.model = kind;
	material.mProgram.defaults = values;
	material.mProgram.authoredMask = (1u << MaterialParameterCount) - 1;
	material.mColorSpace = colorSpace;
	SurfaceInteraction intr;
	intr.material = &material;
	intr.lambda = lambda;
	intr.sd.bsdfType = kind == MaterialModel::OpenPBR
		? MaterialType::OpenPBR : MaterialType::PreviewSurface;
	for (int orientation = 0; orientation < 2; ++orientation) {
		Frame frame(orientation == 0 ? Vector3f(0.f, 0.f, 1.f) :
			normalize(Vector3f(.3f, -.2f, 1.f)));
		intr.n = frame.N;
		intr.tangent = frame.T;
		intr.bitangent = frame.B;
		OpenPbrBsdf bsdf;
		bsdf.setup(intr);
		BSDF variant(intr);
		BxDF tagged(&bsdf);
		for (int direction = 0; direction < 32; ++direction) {
			float z = 2 * (direction + .5f) / 32 - 1;
			float phi = 2 * M_PI * fmodf(direction * .61803398875f + .25f, 1.f);
			Vector3f wi(sqrtf(1 - z * z) * cosf(phi), sqrtf(1 - z * z) * sinf(phi), z);
			if (direction == 0) wi = Vector3f(1.f, 0.f, 0.f);
			if (direction == 1) wi = Vector3f(-1.f, 0.f, 0.f);
			Vector3f outgoing = direction == 2 ? Vector3f(1.f, 0.f, 0.f) : wo;
			for (int modeIndex = 0; modeIndex < 2; ++modeIndex) {
				auto mode = modeIndex == 0 ? TransportMode::Radiance : TransportMode::Importance;
				Spectrum expected = bsdf.f(outgoing, wi, mode);
				float density = bsdf.pdf(outgoing, wi, mode);
				if (!matchesEvaluation(bsdf.eval(outgoing, wi, mode), expected, density)) return false;
				const auto &model = bsdf.prepare(outgoing);
				Vector3f world = intr.toWorld(wi);
				if (!matchesEvaluation(model.evalCosPdf(world, mode),
					model.evalCos(world, mode), model.pdf(world))) return false;
				if (direction < 4 &&
					(!matchesEvaluation(variant.eval(outgoing, wi, mode), expected, density) ||
					 !matchesEvaluation(tagged.eval(outgoing, wi, mode), expected, density) ||
					 !matchesEvaluation(BxDF::eval(intr, outgoing, wi, mode), expected, density)))
					return false;
			}
		}
	}
	return true;
}

__device__ bool checkPreparedBsdf(const MaterialValues &values, Vector3f wo,
	const SampledWavelengths &lambda, const RGBColorSpace *colorSpace, MaterialModel kind) {
	rt::MaterialData material;
	material.mProgram.enabled = true;
	material.mProgram.model = kind;
	material.mProgram.defaults = values;
	material.mProgram.authoredMask = (1u << MaterialParameterCount) - 1;
	material.mColorSpace = colorSpace;
	SurfaceInteraction intr;
	intr.material = &material;
	intr.n = Vector3f(0.f, 0.f, 1.f);
	intr.tangent = Vector3f(1.f, 0.f, 0.f);
	intr.bitangent = Vector3f(0.f, 1.f, 0.f);
	intr.lambda = lambda;
	intr.sd.bsdfType = kind == MaterialModel::OpenPBR
		? MaterialType::OpenPBR : MaterialType::PreviewSurface;
	OpenPbrBsdf cached;
	cached.setup(intr);
	PCGSampler reusedSampler, freshSampler, referenceSampler;
	reusedSampler.setSeed(217, 13);
	freshSampler.setSeed(217, 13);
	referenceSampler.setSeed(217, 13);
	Sampler reused(&reusedSampler), fresh(&freshSampler), reference(&referenceSampler);
	MaterialContext context;
	for (int change = 0; change < 3; ++change) {
		if (change == 1) wo = normalize(Vector3f(-.3f, .2f, .7f));
		if (change == 2) {
			material.mProgram.defaults[MaterialParameter::BaseColor] = MaterialValue(.2f, .5f, .8f);
			material.mProgram.defaults[MaterialParameter::CoatWeight] = MaterialValue(.35f);
			cached.setup(intr);
		}
		openpbr::Model model(material.mProgram.defaults, context, wo, lambda, *colorSpace, kind);
		for (int sample = 0; sample < 64; ++sample) {
			OpenPbrBsdf freshBsdf;
			freshBsdf.setup(intr);
			auto a = cached.sample(wo, reused, TransportMode::Importance);
			auto b = freshBsdf.sample(wo, fresh, TransportMode::Importance);
			// Reproduce the separate evaluation/PDF sampling path with the same random draws.
			Vector3f direction = model.sample(reference.get1D(), reference.get2D());
			BSDFSample separate;
			if (direction.squaredNorm() != 0) {
				Vector3f wi = intr.toLocal(direction);
				float density = model.pdf(direction);
				if (density > 0 && wi[2] != 0)
					separate = {model.evalCos(direction, TransportMode::Importance) / fabsf(wi[2]),
						wi, density, wi[2] * wo[2] > 0 ? BSDF_GLOSSY_REFLECTION : BSDF_GLOSSY_TRANSMISSION};
			}
			PCGSampler reusedCheck = reusedSampler, freshCheck = freshSampler,
				referenceCheck = referenceSampler;
			for (int draw = 0; draw < 3; ++draw) {
				float next = reusedCheck.get1D();
				if (next != freshCheck.get1D() || next != referenceCheck.get1D()) return false;
			}
			// Independently inlining Model::sample can change unit directions by a few ULPs.
			if ((a.pdf > 0) != (separate.pdf > 0) ||
				!matchesEvaluation({a.f, a.pdf}, separate.f, separate.pdf) ||
				(a.pdf > 0 && (a.flags != separate.flags ||
				 (a.wi - separate.wi).cwiseAbs().maxCoeff() > 1e-6f))) return false;
			if (a.pdf != b.pdf || (a.pdf > 0 && (a.flags != b.flags ||
				(a.f - b.f).abs().maxCoeff() != 0 || (a.wi - b.wi).cwiseAbs().maxCoeff() != 0)))
				return false;
			if (a.pdf == 0) continue;
			float pdf = cached.pdf(wo, a.wi);
			if (fabsf(pdf - model.pdf(a.wi)) > 1e-5f * fmaxf(1, pdf)) return false;
			for (int modeIndex = 0; modeIndex < 2; ++modeIndex) {
				auto mode = modeIndex == 0 ? TransportMode::Radiance : TransportMode::Importance;
				Spectrum value = cached.f(wo, a.wi, mode) * fabsf(a.wi[2]);
				if (!matchesEvaluation(cached.eval(wo, a.wi, mode),
					cached.f(wo, a.wi, mode), cached.pdf(wo, a.wi, mode))) return false;
				if ((value - model.evalCos(a.wi, mode)).abs().maxCoeff() >
					1e-5f * fmaxf(1, value.abs().maxCoeff())) return false;
			}
		}
	}
	return true;
}

__global__ void measure(const MaterialValues *cases, Result *results, int count,
						const RGBColorSpace *colorSpace) {
	int index = blockIdx.x;
	if (index >= count) return;
	MaterialContext context;
	SampledWavelengths lambda = SampledWavelengths::sampleUniform(.27f);
	Vector3f wo =
		index == 10 ? normalize(Vector3f(.95f, 0.f, -.2f)) : normalize(Vector3f(.4f, .1f, 1.f));
	if (index == 13) wo = normalize(Vector3f(.98f, .05f, .15f));
	if (index == 14) wo = normalize(Vector3f(.9f, .1f, .3f));
	if (index == 15) wo = Vector3f(0.f, 0.f, 1.f);
	openpbr::Model model(cases[index], context, wo, lambda, *colorSpace, MaterialModel::OpenPBR);
	PCGSampler sampler;
	sampler.setSeed(12345 + index, threadIdx.x + 1);
	Result result;
	if (materialEmissionIntensity(1000, MaterialModel::OpenPBR) != 1 ||
		materialEmissionIntensity(2, MaterialModel::PreviewSurface) != 2)
		++result.invalid;
	if (threadIdx.x == 0 &&
		(!checkPreparedBsdf(cases[index], wo, lambda, colorSpace, MaterialModel::OpenPBR) ||
		 !checkPreparedBsdf(cases[index], wo, lambda, colorSpace, MaterialModel::PreviewSurface) ||
		 !checkCombinedBsdf(cases[index], wo, lambda, colorSpace, MaterialModel::OpenPBR) ||
		 !checkCombinedBsdf(cases[index], wo, lambda, colorSpace, MaterialModel::PreviewSurface)))
		++result.invalid;
	if (index >= 9 && index <= 12 && threadIdx.x == 0) {
		rt::MaterialData material;
		material.mProgram.enabled	   = true;
		material.mProgram.defaults	   = cases[index];
		material.mProgram.authoredMask = (1u << MaterialParameterCount) - 1;
		material.mColorSpace		   = colorSpace;
		SurfaceInteraction intr;
		intr.material	 = &material;
		intr.n			 = Vector3f(0.f, 0.f, 1.f);
		intr.tangent	 = Vector3f(1.f, 0.f, 0.f);
		intr.bitangent	 = Vector3f(0.f, 1.f, 0.f);
		intr.lambda		 = lambda;
		intr.sd.bsdfType = MaterialType::OpenPBR;
		OpenPbrBsdf bsdf;
		bsdf.setup(intr);
		Sampler wrapped(&sampler);
		for (int sample = 0; sample < 64; ++sample) {
			auto value = bsdf.sample(wo, wrapped, TransportMode::Importance);
			if (value.pdf == 0) continue;
			float expectedPdf = bsdf.pdf(wo, value.wi, TransportMode::Importance);
			if (value.isDelta() || !value.isGlossy() ||
				fabsf(expectedPdf - value.pdf) > 1e-4f * value.pdf)
				++result.invalid;
		}
	}
	if (index >= 13) {
		float f = openpbr::fresnel(1.5f, wo[2]), r = 2 * f / (1 + f);
		MaterialValue c = cases[index][MaterialParameter::TransmissionColor];
		Spectrum tint =
			Spectrum::fromRGB(RGB(c[0], c[1], c[2]), SpectrumType::RGBBounded, lambda, *colorSpace);
		float cosRefracted = sqrtf(1 - (1 - wo[2] * wo[2]) / 2.25f);
		result.expected	   = r + (1 - r) * tint.pow(1 / cosRefracted).mean();
	}
	constexpr int samples = 131072;
	for (int sample = threadIdx.x; sample < samples; sample += blockDim.x) {
		Vector3f wi = model.sample(sampler.get1D(), sampler.get2D());
		if (wi.squaredNorm() == 0) continue;
		float pdf	   = model.pdf(wi);
		Spectrum value = model.evalCos(wi, TransportMode::Importance);
		if (!isfinite(pdf) || pdf <= 0 || !value.isFinite().all() || value.minCoeff() < 0 ||
			fabsf(wi.norm() - 1) > 1e-4f) {
			++result.invalid;
			continue;
		}
		++result.valid;
		if (wi[2] * wo[2] < 0) ++result.transmitted;
		float weight   = value.mean() / pdf;
		result.maximum = fmaxf(result.maximum, weight);
		result.sampled += weight / samples;
	}
	constexpr int rows = 256, columns = 512;
	for (int cell = threadIdx.x; cell < rows * columns; cell += blockDim.x) {
		int row = cell / columns, column = cell % columns;
		float z = 2 * (row + .5f) / rows - 1, phi = 2 * M_PI * (column + .5f) / columns;
		Vector3f wi(sqrtf(1 - z * z) * cosf(phi), sqrtf(1 - z * z) * sinf(phi), z);
		result.integrated +=
			model.evalCos(wi, TransportMode::Importance).mean() * (4 * M_PI / (rows * columns));
		result.pdfIntegral += model.pdf(wi) * (4 * M_PI / (rows * columns));
	}
	atomicAdd(&results[index].sampled, result.sampled);
	atomicAdd(&results[index].integrated, result.integrated);
	atomicAdd(&results[index].pdfIntegral, result.pdfIntegral);
	atomicAdd(&results[index].expected, result.expected / blockDim.x);
	atomicAdd(&results[index].valid, result.valid);
	atomicAdd(&results[index].invalid, result.invalid);
	atomicAdd(&results[index].transmitted, result.transmitted);
	atomicMax(reinterpret_cast<unsigned *>(&results[index].maximum),
			  __float_as_uint(result.maximum));
}

int main() {
	MaterialValues *cases = nullptr;
	Result *results		  = nullptr;
	try {
		Context::ensureInitialized();
		checkPrograms();
		constexpr int count = 16;
		CUDA_CHECK(cudaMallocManaged(&cases, count * sizeof(MaterialValues)));
		CUDA_CHECK(cudaMallocManaged(&results, count * sizeof(Result)));
		CUDA_CHECK(cudaMemset(results, 0, count * sizeof(Result)));
		CUDA_CHECK(cudaDeviceSynchronize());
		using P = MaterialParameter;
		for (int index = 0; index < count; ++index) {
			cases[index]					   = defaultMaterialValues(MaterialModel::OpenPBR);
			cases[index][P::BaseColor]		   = MaterialValue(1);
			cases[index][P::SpecularRoughness] = MaterialValue(.65f);
		}
		cases[0][P::SpecularWeight]		 = MaterialValue(0);
		cases[1][P::DiffuseRoughness]	 = MaterialValue(.8f);
		cases[2][P::Metalness]			 = MaterialValue(1);
		cases[3][P::TransmissionWeight]	 = MaterialValue(1);
		cases[4][P::TransmissionWeight]	 = MaterialValue(1);
		cases[4][P::ThinWalled]			 = MaterialValue(1);
		cases[5][P::CoatWeight]			 = MaterialValue(1);
		cases[5][P::CoatRoughness]		 = MaterialValue(.5f);
		cases[6][P::FuzzWeight]			 = MaterialValue(1);
		cases[6][P::FuzzRoughness]		 = MaterialValue(.65f);
		cases[7][P::SpecularAnisotropy]	 = MaterialValue(.65f);
		cases[7][P::Metalness]			 = MaterialValue(1);
		cases[8][P::Normal]				 = MaterialValue(.2f, .15f, .9682458f);
		cases[8][P::CoatNormal]			 = MaterialValue(-.3f, .1f, .9486833f);
		cases[8][P::CoatWeight]			 = MaterialValue(.5f);
		cases[8][P::CoatRoughness]		 = MaterialValue(.5f);
		cases[9][P::SpecularRoughness]	 = MaterialValue(0);
		cases[9][P::Metalness]			 = MaterialValue(1);
		cases[10][P::SpecularRoughness]	 = MaterialValue(0);
		cases[10][P::TransmissionWeight] = MaterialValue(1);
		cases[11][P::SpecularRoughness]	 = MaterialValue(1e-7f);
		cases[11][P::TransmissionWeight] = MaterialValue(1);
		cases[11][P::ThinWalled]		 = MaterialValue(1);
		cases[12][P::TransmissionWeight] = MaterialValue(1);
		cases[12][P::SpecularIor]		 = MaterialValue(1);
		for (int index = 13; index < 16; ++index) {
			cases[index][P::TransmissionWeight] = MaterialValue(1);
			cases[index][P::ThinWalled]			= MaterialValue(1);
			if (index > 13) cases[index][P::TransmissionColor] = MaterialValue(.2f, .5f, .8f);
		}
		measure<<<count, 128>>>(cases, results, count, KRR_DEFAULT_COLORSPACE);
		CUDA_CHECK(cudaGetLastError());
		CUDA_CHECK(cudaDeviceSynchronize());
		bool failed = false;
		for (int index = 0; index < count; ++index) {
			const auto &r = results[index];
			std::cerr << index << ": energy " << r.sampled << " integral " << r.integrated
					  << " pdf " << r.pdfIntegral << " accepted " << double(r.valid) / 131072
					  << " max " << r.maximum << " invalid " << r.invalid << '\n';
			auto check = [&](bool condition, const char *message) {
				if (!condition) {
					failed = true;
					std::cerr << index << ": " << message << '\n';
				}
			};
			check(!r.invalid && r.valid > 0 && std::isfinite(r.sampled),
				  "Invalid native BSDF sample");
			if (index < 9)
				check(fabs(r.sampled - r.integrated) <= .035,
					  "BSDF sampling disagrees with quadrature");
			if (index < 9)
				check(fabs(double(r.valid) / 131072 - r.pdfIntegral) <= .025,
					  "BSDF PDF disagrees with sample acceptance");
			if (index < 8)
				check(r.sampled >= .87 && r.sampled <= 1.05, "OpenPBR white furnace energy");
			if (index == 10)
				check(r.transmitted == 0,
					  "Smooth internal dielectric must exhibit total internal reflection");
			if (index >= 13) {
				check(fabs(r.sampled - r.integrated) < .035,
					  "Oblique thin-sheet sampling disagrees with integration");
				check(fabs(r.sampled - r.expected) < .025,
					  "Thin-sheet energy differs from window Fresnel and absorption");
			}
		}
		if (failed) throw std::runtime_error("Native material validation failed");
		cudaFree(results);
		cudaFree(cases);
		std::cerr << "Native material GPU validation passed\n";
		return 0;
	} catch (const std::exception &error) {
		std::cerr << error.what() << '\n';
		if (results) cudaFree(results);
		if (cases) cudaFree(cases);
		return 1;
	}
}
