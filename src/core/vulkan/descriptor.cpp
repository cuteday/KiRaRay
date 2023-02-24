#include "descriptor.h"

KRR_NAMESPACE_BEGIN

DescriptorHandle::DescriptorHandle() : m_DescriptorIndex(-1) {}

DescriptorHandle::DescriptorHandle(
	const std::shared_ptr<DescriptorTableManager> &managerPtr, DescriptorIndex index) :
	m_Manager(managerPtr), m_DescriptorIndex(index) {}

DescriptorHandle::~DescriptorHandle() {
	if (m_DescriptorIndex >= 0) {
		auto managerPtr = m_Manager.lock();
		if (managerPtr) managerPtr->ReleaseDescriptor(m_DescriptorIndex);
		m_DescriptorIndex = -1;
	}
}

DescriptorTableManager::DescriptorTableManager(nvrhi::IDevice *device,
															  nvrhi::IBindingLayout *layout) :
	m_Device(device) {
	m_DescriptorTable = m_Device->createDescriptorTable(layout);

	size_t capacity = m_DescriptorTable->getCapacity();
	m_AllocatedDescriptors.resize(capacity);
	m_Descriptors.resize(capacity);
	memset(m_Descriptors.data(), 0, sizeof(nvrhi::BindingSetItem) * capacity);
}

DescriptorIndex
DescriptorTableManager::CreateDescriptor(nvrhi::BindingSetItem item) {
	const auto &found = m_DescriptorIndexMap.find(item);
	if (found != m_DescriptorIndexMap.end()) return found->second;

	uint32_t capacity  = m_DescriptorTable->getCapacity();
	bool foundFreeSlot = false;
	uint32_t index	   = 0;
	for (index = m_SearchStart; index < capacity; index++) {
		if (!m_AllocatedDescriptors[index]) {
			foundFreeSlot = true;
			break;
		}
	}

	if (!foundFreeSlot) {
		uint32_t newCapacity =
			std::max(64u, capacity * 2); // handle the initial case when capacity == 0
		m_Device->resizeDescriptorTable(m_DescriptorTable, newCapacity);
		m_AllocatedDescriptors.resize(newCapacity);
		m_Descriptors.resize(newCapacity);

		// zero-fill the new descriptors
		memset(&m_Descriptors[capacity], 0,
			   sizeof(nvrhi::BindingSetItem) * (newCapacity - capacity));

		index	 = capacity;
		capacity = newCapacity;
	}

	item.slot					  = index;
	m_SearchStart				  = index + 1;
	m_AllocatedDescriptors[index] = true;
	m_Descriptors[index]		  = item;
	m_DescriptorIndexMap[item]	  = index;
	m_Device->writeDescriptorTable(m_DescriptorTable, item);

	if (item.resourceHandle) item.resourceHandle->AddRef();

	return index;
}

DescriptorHandle
DescriptorTableManager::CreateDescriptorHandle(nvrhi::BindingSetItem item) {
	DescriptorIndex index = CreateDescriptor(item);
	return DescriptorHandle(shared_from_this(), index);
}

nvrhi::BindingSetItem DescriptorTableManager::GetDescriptor(DescriptorIndex index) {
	if (size_t(index) >= m_Descriptors.size()) return nvrhi::BindingSetItem::None(0);

	return m_Descriptors[index];
}

void DescriptorTableManager::ReleaseDescriptor(DescriptorIndex index) {
	nvrhi::BindingSetItem &descriptor = m_Descriptors[index];

	if (descriptor.resourceHandle) descriptor.resourceHandle->Release();

	// Erase the existing descriptor from the index map to prevent its "reuse" later
	const auto indexMapEntry = m_DescriptorIndexMap.find(m_Descriptors[index]);
	if (indexMapEntry != m_DescriptorIndexMap.end()) m_DescriptorIndexMap.erase(indexMapEntry);

	descriptor = nvrhi::BindingSetItem::None(index);

	m_Device->writeDescriptorTable(m_DescriptorTable, descriptor);

	m_AllocatedDescriptors[index] = false;
	m_SearchStart				  = std::min(m_SearchStart, index);
}

DescriptorTableManager::~DescriptorTableManager() {
	for (auto &descriptor : m_Descriptors) {
		if (descriptor.resourceHandle) {
			descriptor.resourceHandle->Release();
			descriptor.resourceHandle = nullptr;
		}
	}
}

KRR_NAMESPACE_END