#pragma once

#include <common.h>
#include <nvrhi/nvrhi.h>
#include <vector>

NAMESPACE_BEGIN(krr)

struct ShaderMacro {
	std::string name;
	std::string definition;

	ShaderMacro(const std::string &_name, const std::string &_definition) :
		name(_name), definition(_definition) {}
};

class ShaderLoader {
public:
	using SharedPtr = std::shared_ptr<ShaderLoader>;
	ShaderLoader(nvrhi::DeviceHandle device) : mDevice(device) {}

	nvrhi::ShaderHandle createShader(const char *fileName, const char *entryName,
		const std::vector<ShaderMacro> *defines, nvrhi::ShaderType shaderType);
	nvrhi::ShaderHandle createShader(const char *fileName, const char *entryName,
		const std::vector<ShaderMacro> *defines, const nvrhi::ShaderDesc &desc);
	nvrhi::ShaderLibraryHandle createShaderLibrary(const char *fileName,
		const std::vector<ShaderMacro> *defines);
	std::vector<char> getBytecode(const char *fileName, nvrhi::ShaderType shaderType,
		const char *entryName = "main", const std::vector<ShaderMacro> *defines = nullptr);

private:
	nvrhi::DeviceHandle mDevice;
};

NAMESPACE_END(krr)
