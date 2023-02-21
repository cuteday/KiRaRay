#pragma once
#include <nvrhi/nvrhi.h>
#include <unordered_map>
#include <shared_mutex>

#include <common.h>

KRR_NAMESPACE_BEGIN

class BindingCache {
private:
	nvrhi::DeviceHandle m_Device;
	std::unordered_map<size_t, nvrhi::BindingSetHandle> m_BindingSets;
	std::shared_mutex m_Mutex;

public:
	BindingCache(nvrhi::IDevice *device) : m_Device(device) {}
	
	nvrhi::BindingSetHandle GetCachedBindingSet(const nvrhi::BindingSetDesc &desc,
												nvrhi::IBindingLayout *layout);
	nvrhi::BindingSetHandle GetOrCreateBindingSet(const nvrhi::BindingSetDesc &desc,
												  nvrhi::IBindingLayout *layout);
	void Clear();
};

KRR_NAMESPACE_END