#pragma once
#include <nvrhi/nvrhi.h>
#include <unordered_map>
#include <shared_mutex>

#include <common.h>

NAMESPACE_BEGIN(krr)

class BindingCache {
private:
	struct Key {
		nvrhi::BindingSetDesc desc;
		nvrhi::IBindingLayout *layout;

		bool operator==(const Key &other) const {
			if (layout != other.layout || desc.trackLiveness != other.desc.trackLiveness ||
				desc != other.desc) return false;
			for (size_t i = 0; i < desc.bindings.size(); ++i)
				if (desc.bindings[i].arrayElement != other.desc.bindings[i].arrayElement)
					return false;
			return true;
		}
	};

	struct KeyHasher {
		size_t operator()(const Key &key) const {
			size_t hash = 0;
			nvrhi::hash_combine(hash, key.desc);
			nvrhi::hash_combine(hash, key.desc.trackLiveness);
			for (const auto &item : key.desc.bindings)
				nvrhi::hash_combine(hash, item.arrayElement);
			nvrhi::hash_combine(hash, key.layout);
			return hash;
		}
	};

	nvrhi::DeviceHandle m_Device;
	std::unordered_map<Key, nvrhi::BindingSetHandle, KeyHasher> m_BindingSets;
	std::shared_mutex m_Mutex;

public:
	BindingCache(nvrhi::IDevice *device) : m_Device(device) {}
	
	nvrhi::BindingSetHandle GetCachedBindingSet(const nvrhi::BindingSetDesc &desc,
												nvrhi::IBindingLayout *layout);
	nvrhi::BindingSetHandle GetOrCreateBindingSet(const nvrhi::BindingSetDesc &desc,
												  nvrhi::IBindingLayout *layout);
	void Clear();
};

NAMESPACE_END(krr)
