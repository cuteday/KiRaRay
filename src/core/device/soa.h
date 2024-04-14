#pragma once
#include "common.h"
#include <type_traits>

NAMESPACE_BEGIN(krr)

template <typename T> 
class SOA {
public:
	struct GetSetIndirector { SOA<T> *soa; int i; };
};

template <typename T> 
class SOAIterator {
public:
	using difference_type	= int;
	using value_type		= typename SOA<T>::GetSetIndirector;
	using reference			= typename SOA<T>::GetSetIndirector &;
	using pointer			= void;
	using iterator_category = std::random_access_iterator_tag;

	KRR_CALLABLE SOAIterator() : m_soa(nullptr), m_index(0) {}
	KRR_CALLABLE SOAIterator(SOA<T> *soa, int index) : m_soa(soa), m_index(index) {}
	KRR_CALLABLE SOAIterator(const SOA<T> *soa, int index) : m_soa(const_cast<SOA<T>*>(soa)), m_index(index) {}
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

	KRR_CALLABLE typename SOA<T>::GetSetIndirector operator*() { return {m_soa, m_index}; }
	KRR_CALLABLE typename SOA<T>::GetSetIndirector operator[](difference_type n) { return {m_soa, m_index + n}; }

private:
	std::conditional_t<std::is_const_v<T>, const SOA<T>*, SOA<T>*> m_soa;
	difference_type m_index;
};

NAMESPACE_END(krr)