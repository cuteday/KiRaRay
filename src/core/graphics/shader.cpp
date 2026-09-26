#include "shader.h"
#include <file.h>
#include <logger.h>

#ifdef _WIN32
#include <wrl/client.h>
#endif
#include <dxc/dxcapi.h>

NAMESPACE_BEGIN(krr)

#ifdef _WIN32
using Microsoft::WRL::ComPtr;
#else
template <typename T> using ComPtr = CComPtr<T>;
#endif

namespace {

void checkDxc(HRESULT result, const char *message) {
	if (FAILED(result)) throw std::runtime_error(message);
}

const wchar_t *getShaderProfile(nvrhi::ShaderType stage) {
	switch (stage) {
	case nvrhi::ShaderType::Vertex: return L"vs_6_7";
	case nvrhi::ShaderType::Pixel: return L"ps_6_7";
	case nvrhi::ShaderType::Compute: return L"cs_6_7";
	case nvrhi::ShaderType::Geometry: return L"gs_6_7";
	case nvrhi::ShaderType::Mesh: return L"ms_6_7";
	case nvrhi::ShaderType::Hull: return L"hs_6_7";
	case nvrhi::ShaderType::Domain: return L"ds_6_7";
	case nvrhi::ShaderType::Amplification: return L"as_6_7";
	case nvrhi::ShaderType::AllRayTracing: return L"lib_6_7";
	default: throw std::runtime_error("Unsupported shader stage");
	}
}

}

nvrhi::ShaderHandle ShaderLoader::createShader(const char *fileName, const char *entryName,
	const std::vector<ShaderMacro> *defines, nvrhi::ShaderType shaderType) {
	nvrhi::ShaderDesc desc;
	desc.shaderType = shaderType;
	desc.debugName = fileName;
	return createShader(fileName, entryName, defines, desc);
}

nvrhi::ShaderHandle ShaderLoader::createShader(const char *fileName, const char *entryName,
	const std::vector<ShaderMacro> *defines, const nvrhi::ShaderDesc &desc) {
	auto bytecode = getBytecode(fileName, desc.shaderType, entryName, defines);
	nvrhi::ShaderDesc shaderDesc = desc;
	shaderDesc.entryName = entryName ? entryName : "main";
	return mDevice->createShader(shaderDesc, bytecode.data(), bytecode.size());
}

nvrhi::ShaderLibraryHandle ShaderLoader::createShaderLibrary(const char *fileName,
	const std::vector<ShaderMacro> *defines) {
	auto bytecode = getBytecode(fileName, nvrhi::ShaderType::AllRayTracing, nullptr, defines);
	return mDevice->createShaderLibrary(bytecode.data(), bytecode.size());
}

std::vector<char> ShaderLoader::getBytecode(const char *fileName, nvrhi::ShaderType shaderType,
	const char *entryName, const std::vector<ShaderMacro> *defines) {
	const fs::path path = fs::path(KRR_PROJECT_DIR) / fileName;
	auto content = File::readFile(path);
	if (!content) throw std::runtime_error("Failed to read shader: " + path.string());

	std::vector<std::wstring> arguments{path.wstring(), L"-T", getShaderProfile(shaderType),
		L"-I", path.parent_path().wstring()};
	if (shaderType != nvrhi::ShaderType::AllRayTracing) {
		arguments.push_back(L"-E");
		arguments.push_back(stringToWideString(entryName ? entryName : "main"));
	}
	if (mDevice->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN) {
		arguments.insert(arguments.end(), {L"-spirv", L"-fspv-target-env=vulkan1.3",
			L"-D", L"KRR_SHADER_VULKAN=1",
			L"-fvk-t-shift", L"0", L"0", L"-fvk-s-shift", L"128", L"0",
			L"-fvk-b-shift", L"256", L"0", L"-fvk-u-shift", L"384", L"0"});
	} else if (mDevice->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12) {
		arguments.insert(arguments.end(), {L"-D", L"KRR_SHADER_VULKAN=0"});
	} else {
		throw std::runtime_error("Unsupported graphics API for shader compilation");
	}
	if (defines) {
		for (const auto &define : *defines) {
			arguments.push_back(L"-D");
			arguments.push_back(stringToWideString(define.name + "=" + define.definition));
		}
	}
	std::vector<LPCWSTR> argumentPointers;
	argumentPointers.reserve(arguments.size());
	for (const auto &argument : arguments) argumentPointers.push_back(argument.c_str());

	ComPtr<IDxcCompiler3> compiler;
	checkDxc(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler)),
		"Failed to create DXC compiler");
	ComPtr<IDxcUtils> utils;
	checkDxc(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils)),
		"Failed to create DXC utilities");
	ComPtr<IDxcIncludeHandler> includeHandler;
	checkDxc(utils->CreateDefaultIncludeHandler(&includeHandler),
		"Failed to create shader include handler");

	DxcBuffer source{content->data(), content->size(), DXC_CP_UTF8};
	ComPtr<IDxcResult> result;
	checkDxc(compiler->Compile(&source, argumentPointers.data(), UINT32(argumentPointers.size()),
		includeHandler.operator->(), IID_PPV_ARGS(&result)), "Failed to invoke DXC");
	HRESULT status;
	checkDxc(result->GetStatus(&status), "Failed to get shader compilation status");
	ComPtr<IDxcBlobUtf8> errors;
	checkDxc(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr),
		"Failed to get shader diagnostics");
	if (FAILED(status)) {
		const std::string diagnostics = errors ? errors->GetStringPointer() : "Unknown DXC error";
		throw std::runtime_error("Failed to compile shader " + path.string() + ": " + diagnostics);
	}
	if (errors && errors->GetStringLength()) Log(Warning, "%s", errors->GetStringPointer());
	ComPtr<IDxcBlob> output;
	checkDxc(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&output), nullptr),
		"Failed to get shader bytecode");
	if (!output) throw std::runtime_error("DXC returned no shader bytecode");
	const char *begin = static_cast<const char *>(output->GetBufferPointer());
	return {begin, begin + output->GetBufferSize()};
}

NAMESPACE_END(krr)
