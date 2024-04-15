#pragma once
#include "common.h"
#include <type_traits>

NAMESPACE_BEGIN(krr)

// specifications for SOA::GetSetIndirector
template <typename T> 
class SOA {
public:
	struct GetSetIndirector { 
		GetSetIndirector() = default; 
		KRR_CALLABLE operator T() const;
		KRR_CALLABLE void operator=(const T &val);
		KRR_CALLABLE void operator=(const GetSetIndirector &other);
		SOA<T> *soa; int i; 
	};
};

// https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator
template <typename T> 
class SOAIterator {
public:
	using difference_type	= int;
	using value_type		= T;
	using reference			= typename SOA<T>::GetSetIndirector;
	using pointer			= void;
	using iterator_category = std::random_access_iterator_tag;

	KRR_CALLABLE SOAIterator() : m_soa(nullptr), m_index(0) {}
	KRR_CALLABLE SOAIterator(SOA<T> *soa, int index) : m_soa(soa), m_index(index) {}
	KRR_CALLABLE SOAIterator(const SOA<T> *soa, int index) : m_soa(const_cast<SOA<T>*>(soa)), m_index(index) {}

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

	KRR_CALLABLE bool operator==(const SOAIterator& it) const { return m_index == it.m_index; }
	KRR_CALLABLE bool operator!=(const SOAIterator &it) const { return m_index != it.m_index; }
	KRR_CALLABLE bool operator<(const SOAIterator &it) const { return m_index < it.m_index; }
	KRR_CALLABLE bool operator<=(const SOAIterator &it) const { return m_index <= it.m_index; }
	KRR_CALLABLE bool operator>(const SOAIterator &it) const { return m_index > it.m_index; }
	KRR_CALLABLE bool operator>=(const SOAIterator &it) const { return m_index >= it.m_index; }

	KRR_CALLABLE reference operator*() { return {m_soa, m_index}; }
	KRR_CALLABLE reference operator[](difference_type n) { return {m_soa, m_index + n}; }

private:
	std::conditional_t<std::is_const_v<T>, const SOA<T>*, SOA<T>*> m_soa;
	difference_type m_index;
};

NAMESPACE_END(krr)