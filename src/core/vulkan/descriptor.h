#pragma once

#include <nvrhi/nvrhi.h>
#include <unordered_map>
#include <memory>

#include "common.h"

KRR_NAMESPACE_BEGIN

class DescriptorTableManager;
typedef int DescriptorIndex;

// Stores a descriptor index in a descriptor table. Releases the descriptor when destroyed.
class DescriptorHandle {
private:
	std::weak_ptr<DescriptorTableManager> m_Manager;
	DescriptorIndex m_DescriptorIndex;

public:
	DescriptorHandle();
	DescriptorHandle(const std::shared_ptr<DescriptorTableManager> &managerPtr,
					 DescriptorIndex index);
	~DescriptorHandle();

	[[nodiscard]] bool IsValid() const { return m_DescriptorIndex >= 0 && !m_Manager.expired(); }
	[[nodiscard]] DescriptorIndex Get() const {
		if (m_DescriptorIndex >= 0) assert(!m_Manager.expired());
		return m_DescriptorIndex;
	}
	void Reset() {
		m_DescriptorIndex = -1;
		m_Manager.reset();
	}

	// Movable but non-copyable
	DescriptorHandle(const DescriptorHandle &)			  = delete;
	DescriptorHandle(DescriptorHandle &&)				  = default;
	DescriptorHandle &operator=(const DescriptorHandle &) = delete;
	DescriptorHandle &operator=(DescriptorHandle &&)	  = default;
};

class DescriptorTableManager : public std::enable_shared_from_this<DescriptorTableManager> {
protected:
	// Custom hasher that doesn't look at the binding slot
	struct BindingSetItemHasher {
		std::size_t operator()(const nvrhi::BindingSetItem &item) const {
			size_t hash = 0;
			nvrhi::hash_combine(hash, item.resourceHandle);
			nvrhi::hash_combine(hash, item.type);
			nvrhi::hash_combine(hash, item.format);
			nvrhi::hash_combine(hash, item.dimension);
			nvrhi::hash_combine(hash, item.rawData[0]);
			nvrhi::hash_combine(hash, item.rawData[1]);
			return hash;
		}
	};

	// Custom equality tester that doesn't look at the binding slot
	struct BindingSetItemsEqual {
		bool operator()(const nvrhi::BindingSetItem &a, const nvrhi::BindingSetItem &b) const {
			return a.resourceHandle == b.resourceHandle && a.type == b.type &&
				   a.format == b.format && a.dimension == b.dimension &&
				   a.subresources == b.subresources;
		}
	};

	nvrhi::DeviceHandle m_Device;
	nvrhi::DescriptorTableHandle m_DescriptorTable;

	std::vector<nvrhi::BindingSetItem> m_Descriptors;
	std::unordered_map<nvrhi::BindingSetItem, DescriptorIndex, BindingSetItemHasher,
					   BindingSetItemsEqual>
		m_DescriptorIndexMap;
	std::vector<bool> m_AllocatedDescriptors;
	int m_SearchStart = 0;

public:
	DescriptorTableManager(nvrhi::IDevice *device, nvrhi::IBindingLayout *layout);
	~DescriptorTableManager();

	nvrhi::IDescriptorTable *GetDescriptorTable() const { return m_DescriptorTable; }

	DescriptorIndex CreateDescriptor(nvrhi::BindingSetItem item);
	DescriptorHandle CreateDescriptorHandle(nvrhi::BindingSetItem item);
	nvrhi::BindingSetItem GetDescriptor(DescriptorIndex index);
	void ReleaseDescriptor(DescriptorIndex index);
};

KRR_NAMESPACE_END