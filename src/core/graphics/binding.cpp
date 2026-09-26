#include "binding.h"

NAMESPACE_BEGIN(krr)

nvrhi::BindingSetHandle BindingCache::GetCachedBindingSet(const nvrhi::BindingSetDesc &desc,
														  nvrhi::IBindingLayout *layout) {
	size_t hash = 0;
	nvrhi::hash_combine(hash, desc);
	nvrhi::hash_combine(hash, layout);

	m_Mutex.lock_shared();

	nvrhi::BindingSetHandle result = nullptr;
	auto it						   = m_BindingSets.find(hash);
	if (it != m_BindingSets.end()) result = it->second;

	m_Mutex.unlock_shared();

	if (result) {
		assert(result->getDesc());
		assert(*result->getDesc() == desc);
	}

	return result;
}

nvrhi::BindingSetHandle BindingCache::GetOrCreateBindingSet(const nvrhi::BindingSetDesc &desc,
															nvrhi::IBindingLayout *layout) {
	size_t hash = 0;
	nvrhi::hash_combine(hash, desc);
	nvrhi::hash_combine(hash, layout);

	m_Mutex.lock_shared();

	nvrhi::BindingSetHandle result;
	auto it = m_BindingSets.find(hash);
	if (it != m_BindingSets.end()) result = it->second;

	m_Mutex.unlock_shared();

	if (!result) {
		m_Mutex.lock();

		nvrhi::BindingSetHandle &entry = m_BindingSets[hash];
		if (!entry) {
			result = m_Device->createBindingSet(desc, layout);
			entry  = result;
		} else
			result = entry;

		m_Mutex.unlock();
	}

	if (result) {
		assert(result->getDesc());
		assert(*result->getDesc() == desc);
	}

	return result;
}

void BindingCache::Clear() {
	m_Mutex.lock();
	m_BindingSets.clear();
	m_Mutex.unlock();
}

NAMESPACE_END(krr)