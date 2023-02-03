#pragma once

#include <vector>
#include <file.h>
#include <common.h>

#include <nvrhi/nvrhi.h>
#include <nvrhi/common/shader-blob.h>

#ifdef _WIN32
// dxcapi.h expects the COM API to be present on Windows.
// On other platforms, the Vulkan SDK will have WinAdapter.h alongside dxcapi.h that is
// automatically included to stand in for the COM API.
#include <atlbase.h>
#endif
#include <dxc/dxcapi.h>
#define DXC_HR(hr, msg)                                                                            \
	if (FAILED(hr)) {                                                                              \
		throw std::runtime_error{msg};                                                     \
	}

KRR_NAMESPACE_BEGIN

struct ShaderMacro {
	std::string name;
	std::string definition;

	ShaderMacro(const std::string &_name, const std::string &_definition) :
		name(_name), definition(_definition) {}
};

enum class ShaderLanguage {
	GLSL, HLSL, SPIRV
};

class ShaderLoader {
private:
	nvrhi::DeviceHandle m_device;
 	std::unordered_map<std::string, std::vector<char>> m_bytecodeCache;

public:
	ShaderLoader(nvrhi::DeviceHandle rendererInterface) : m_device(rendererInterface){}

	void clearCache() { m_bytecodeCache.clear(); }

	std::vector<char> compileSpirv(std::vector<char> src, ShaderLanguage srcLanguage, 
		nvrhi::ShaderType shaderStage, const char* entryPoint) {
		string entry_str(entryPoint);	// convert string to wide-char string...
		std::wstring entry{entry_str.begin(), entry_str.end()};
		switch (srcLanguage) {
			case ShaderLanguage::HLSL: {
				std::vector<LPCWSTR> arguments;
				arguments.push_back(L"-E");
				arguments.push_back(static_cast<LPCWSTR>(entry.c_str()));
				arguments.push_back(L"-spirv");
				arguments.push_back(L"-fspv-target-env=vulkan1.1");
				arguments.push_back(L"-fvk-use-gl-layout");
				arguments.push_back(L"-no-warnings");

				static const std::unordered_map<nvrhi::ShaderType, LPCWSTR> stage_mappings{
					{nvrhi::ShaderType::Vertex, L"vs_6_7"},
					{nvrhi::ShaderType::Pixel, L"ps_6_7"},
					{nvrhi::ShaderType::Compute, L"cs_6_7"},
					{nvrhi::ShaderType::Geometry, L"gs_6_7"},
					{nvrhi::ShaderType::Mesh, L"ms_6_7"},
					{nvrhi::ShaderType::Hull, L"hs_6_7"},
					{nvrhi::ShaderType::Domain, L"ds_6_7"},
					{nvrhi::ShaderType::Amplification, L"as_6_7"},
					{nvrhi::ShaderType::AllRayTracing, L"lib_6_7"}};

				arguments.push_back(L"-T");
				if (stage_mappings.count(shaderStage))
					arguments.push_back(stage_mappings.at(shaderStage));
				else arguments.push_back(L"lib_6_7");

				DxcBuffer source_buf;
				source_buf.Ptr		= src.data();
				source_buf.Size		= src.size();
				source_buf.Encoding = 0;

				CComPtr<IDxcCompiler3> compiler = nullptr;
				DXC_HR(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler3),
										 (void **) &compiler),
					   "Failed to create DXC compiler");

				CComPtr<IDxcUtils> utils = nullptr;
				DXC_HR(DxcCreateInstance(CLSID_DxcUtils, __uuidof(IDxcUtils), (void **) &utils),
					   "Failed to create DXC utils");

				CComPtr<IDxcIncludeHandler> include_handler = nullptr;
				DXC_HR(utils->CreateDefaultIncludeHandler(&include_handler),
					   "Failed to create include handler");

				CComPtr<IDxcResult> result = nullptr;
				DXC_HR(compiler->Compile(&source_buf, arguments.data(), arguments.size(),
										 &*include_handler, __uuidof(IDxcResult),
										 (void **) &result),
					   "Failed to compile with DXC");

				CComPtr<IDxcBlobUtf8> errors = nullptr;
				DXC_HR(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr),
					   "Failed to get DXC compile errors");
				if (errors && errors->GetStringLength() > 0) {
					std::string message = errors->GetStringPointer();
					throw std::runtime_error{message};
				}

				CComPtr<IDxcBlob> output = nullptr;
				DXC_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&output), nullptr),
					   "Failed to get DXC output");
				assert(output != nullptr);

				const char *begin = (const char *) output->GetBufferPointer();
				const char *end	  = begin + output->GetBufferSize();

				return std::vector<char>{begin, end};
			}
			default:
				throw std::runtime_error("Unsupported shader language");
		}
	}

	nvrhi::ShaderHandle createShader(const char *fileName, const char *entryName,
									 const std::vector<ShaderMacro> *pDefines,
									 nvrhi::ShaderType shaderType){
		nvrhi::ShaderDesc desc = nvrhi::ShaderDesc(shaderType);
		desc.debugName		   = fileName;
		return createShader(fileName, entryName, pDefines, desc);
	}
	
	nvrhi::ShaderHandle createShader(const char *fileName, const char *entryName,
									 const std::vector<ShaderMacro> *pDefines,
									 const nvrhi::ShaderDesc &desc){
		
		std::vector<char> byteCode = getBytecode(fileName, desc.shaderType, entryName);

		std::vector<nvrhi::ShaderConstant> constants;
		if (pDefines) {
			for (const ShaderMacro &define : *pDefines)
				constants.push_back(
					nvrhi::ShaderConstant{define.name.c_str(), define.definition.c_str()});
		}

		nvrhi::ShaderDesc descCopy = desc;
		descCopy.entryName		   = entryName;

		return nvrhi::createShaderPermutation(m_device, descCopy, byteCode.data(),
											  byteCode.size(), constants.data(),
											  uint32_t(constants.size()));
	}
	
	// this is mainly for ray-tracing pipelined shaders (?)
	nvrhi::ShaderLibraryHandle createShaderLibrary(const char* fileName,
		const std::vector<ShaderMacro>* pDefines) {
		std::vector<char> byteCode = getBytecode(fileName, nvrhi::ShaderType::AllRayTracing, nullptr);

		std::vector<nvrhi::ShaderConstant> constants;
		if (pDefines) {
			for (const ShaderMacro &define : *pDefines)
				constants.push_back(
					nvrhi::ShaderConstant{define.name.c_str(), define.definition.c_str()});
		}

		return nvrhi::createShaderLibraryPermutation(m_device, byteCode.data(), byteCode.size(),
													 constants.data(), uint32_t(constants.size()));
	}
	
	static void readFile(fs::path filepath, std::vector<char> &data, bool binary = true) {
		auto flags = binary ? (std::ios::binary | std::ios::in) : std::ios::in;
		std::ifstream s(filepath, flags);
		if (!s.good()) throw std::runtime_error("Failed to load shader contents");
		s.seekg(0, std::ios_base::end);
		data.resize(s.tellg());
		s.clear();
		s.seekg(0, std::ios_base::beg);
		s.read(data.data(), data.size());
	}

	std::vector<char> getBytecode(const char *fileName, nvrhi::ShaderType shaderType, const char *entryName = "main") {
		if (!entryName) entryName = "main";
		
		std::filesystem::path shaderFilePath = File::resolve(fileName);
		std::vector<char> text;		// shader text
		readFile(shaderFilePath, text, false);

		std::vector<char> byteCode =
			compileSpirv(text, ShaderLanguage::HLSL, shaderType, entryName); // compiled shader
		return byteCode;
	}
};

KRR_NAMESPACE_END