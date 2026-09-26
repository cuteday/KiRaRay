#include "binding.h"
#include <mutex>

NAMESPACE_BEGIN(krr)

nvrhi::BindingSetHandle BindingCache::GetCachedBindingSet(const nvrhi::BindingSetDesc &desc,
														  nvrhi::IBindingLayout *layout) {
	std::shared_lock lock(m_Mutex);
	auto it = m_BindingSets.find(Key{desc, layout});
	return it != m_BindingSets.end() ? it->second : nullptr;
}

nvrhi::BindingSetHandle BindingCache::GetOrCreateBindingSet(const nvrhi::BindingSetDesc &desc,
															nvrhi::IBindingLayout *layout) {
	if (auto result = GetCachedBindingSet(desc, layout)) return result;
	std::unique_lock lock(m_Mutex);
	Key key{desc, layout};
	auto it = m_BindingSets.find(key);
	if (it != m_BindingSets.end()) return it->second;
	auto result = m_Device->createBindingSet(desc, layout);
	if (result) m_BindingSets.emplace(std::move(key), result);
	return result;
}

void BindingCache::Clear() {
	std::unique_lock lock(m_Mutex);
	m_BindingSets.clear();
}

NAMESPACE_END(krr)
