#pragma once
#include "common.h"

NAMESPACE_BEGIN(krr)

template <typename T> 
class SOA;

template <typename T>
struct SOAGetSetIndirector {
	KRR_CALLABLE SOAGetSetIndirector() = default;
	KRR_CALLABLE SOAGetSetIndirector(SOA<T> *soa, int index) : m_soa(soa), m_index(index) {}
	KRR_CALLABLE operator T() const { return m_soa->operator[](m_index); };
	KRR_CALLABLE void operator=(const T &val);

	SOA<T> *m_soa;
	int m_index;
};

template <typename T> 
class SOAIterator {
public:
	using difference_type = int;

	KRR_CALLABLE SOAIterator() : m_soa(nullptr), m_index(0) {}
	KRR_CALLABLE SOAIterator(SOA<T> *soa, int index) : m_soa(soa), m_index(index) {}
	KRR_CALLABLE SOAIterator(const SOAIterator &it) : m_soa(it.m_soa), m_index(it.m_index) {}

	KRR_CALLABLE SOAIterator& operator +=(int n) { m_index += n; return *this; }
	KRR_CALLABLE SOAIterator& operator -=(int n) { m_index -= n; return *this; }
	KRR_CALLABLE SOAIterator& operator ++() { ++m_index; return *this; }
	KRR_CALLABLE SOAIterator& operator --() { --m_index; return *this; }
	KRR_CALLABLE SOAIterator operator ++(int) { SOAIterator it = *this; ++m_index; return it; }
	KRR_CALLABLE SOAIterator operator --(int) { SOAIterator it = *this; --m_index; return it; }
	KRR_CALLABLE SOAIterator operator+(difference_type n) const { return SOAIterator(m_soa, m_index + n); }
	KRR_CALLABLE SOAIterator operator-(difference_type n) const { return SOAIterator(m_soa, m_index - n); }
	KRR_CALLABLE difference_type operator-(const SOAIterator& it) const { return m_index - it.m_index; }
	KRR_CALLABLE friend SOAIterator operator+(difference_type n, const SOAIterator& it) { return it + n; }
	KRR_CALLABLE friend SOAIterator operator-(difference_type n, const SOAIterator &it) { return it - n; }

	KRR_CALLABLE bool operator ==(const SOAIterator& it) const { return m_index == it.m_index; }
	KRR_CALLABLE bool operator!=(const SOAIterator &it) const { return m_index != it.m_index; }
	KRR_CALLABLE bool operator<(const SOAIterator &it) const { return m_index < it.m_index; }
	KRR_CALLABLE bool operator<=(const SOAIterator &it) const { return m_index <= it.m_index; }
	KRR_CALLABLE bool operator>(const SOAIterator &it) const { return m_index > it.m_index; }
	KRR_CALLABLE bool operator>=(const SOAIterator &it) const { return m_index >= it.m_index; }

	KRR_CALLABLE SOA<T> *operator->() { return m_soa; }
	KRR_CALLABLE SOAGetSetIndirector<T> operator*() { return {m_soa, m_index}; }
	KRR_CALLABLE SOAGetSetIndirector<T> operator[](difference_type n) { return {m_soa, m_index + n}; }

private:
	SOA<T>* m_soa;
	int m_index;
};

NAMESPACE_END(krr)