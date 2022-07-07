#pragma once

#include "pybind11/pybind11.h"
#include <pybind11/operators.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>


#define KRR_PY_CLASS(Name, Base, ...)                                          \
	py::class_<Name, Base, ref<Name>>(m, #Name, D(Name), ##__VA_ARGS__)

#define KRR_PY_TRAMPOLINE_CLASS(Trampoline, Name, Base, ...)                   \
	py::class_<Name, Base, ref<Name>, Trampoline>(m, #Name, D(Name),           \
												  ##__VA_ARGS__)

#define KRR_PY_STRUCT(Name, ...)                                               \
	py::class_<Name>(m, #Name, D(Name), ##__VA_ARGS__)

/// Shorthand notation for defining read_write members
#define def_field(Class, Member, ...)                                          \
	def_readwrite(#Member, &Class::Member, ##__VA_ARGS__)

/// Shorthand notation for defining enum members
#define def_value(Class, Value, ...)                                           \
	value(#Value, Class::Value, D(Class, Value), ##__VA_ARGS__)

/// Shorthand notation for defining most kinds of methods
#define def_method(Class, Function, ...)                                       \
	def(#Function, &Class::Function, D(Class, Function), ##__VA_ARGS__)

/// Shorthand notation for defining most kinds of static methods
#define def_static_method(Class, Function, ...)                                \
	def_static(#Function, &Class::Function, D(Class, Function), ##__VA_ARGS__)

/// Shorthand notation for defining __repr__ using operator<<
#define def_repr(Class)                                                        \
	def("__repr__", [](const Class &c) {                                       \
		std::ostringstream oss;                                                \
		oss << c;                                                              \
		return oss.str();                                                      \
	})

namespace py = pybind11;
using namespace py::literals;

template <typename Class, typename ScalarClass, typename PyClass>
void bind_slicing_operators(PyClass &cl) {
	using Float = typename Class::Float;

	if constexpr (is_dynamic_v<Float>) {
		cl.def("__getitem__",
			   [](Class &c, size_t i) -> ScalarClass {
				   if (i >= slices(c))
					   throw py::index_error();
				   return slice(c, i);
			   })
			.def("__setitem__",
				 [](Class &c, size_t i, const ScalarClass &c2) {
					 if (i >= slices(c))
						 throw py::index_error();
					 slice(c, i) = c2;
				 })
			// TODO enabled this when ENOKI_STRUCT_SUPPORT is fixed for structs
			// containing Matrix .def("__setitem__", [](Class &c, const
			// mask_t<Float> &mask, const Class &c2) {
			//     masked(c, mask) = c2;
			// })
			.def("__len__", [](const Class &c) { return slices(c); });
	}

	cl.def_static(
		"zero",
		[](size_t size) {
			if constexpr (!is_dynamic_v<Float>) {
				if (size != 1)
					throw std::runtime_error(
						"zero(): Size must equal 1 in scalar mode!");
			}
			return zero<Class>(size);
		},
		"size"_a = 1);
}

template <typename Source, typename Target> void pybind11_type_alias() {
	auto &types = pybind11::detail::get_internals().registered_types_cpp;
	auto it		= types.find(std::type_index(typeid(Source)));
	if (it == types.end())
		throw std::runtime_error(
			"pybind11_type_alias(): source type not found!");
	types[std::type_index(typeid(Target))] = it->second;
}

template <typename Type> pybind11::handle get_type_handle() {
	return pybind11::detail::get_type_handle(typeid(Type), false);
}

#define KRR_PY_DECLARE(Name) extern void python_export_##Name(py::module &m)
#define KRR_PY_EXPORT(Name) void python_export_##Name(py::module &m)
#define KRR_PY_IMPORT(Name) python_export_##Name(m)
#define KRR_PY_IMPORT_SUBMODULE(Name) python_export_##Name(Name)

#define KRR_MODULE_NAME_1(lib, variant) lib##_##variant##_ext
#define KRR_MODULE_NAME(lib, variant) KRR_MODULE_NAME_1(lib, variant)

template <typename Func> decltype(auto) vectorize(const Func &func) {
#if KRR_VARIANT_VECTORIZE == 1
	return enoki::vectorize_wrapper(func);
#else
	return func;
#endif
}

inline py::module create_submodule(py::module &m, const char *name) {
	std::string full_name = std::string(PyModule_GetName(m.ptr())) + "." + name;
	py::module module =
		py::reinterpret_steal<py::module>(PyModule_New(full_name.c_str()));
	m.attr(name) = module;
	return module;
}

#define KRR_PY_CHECK_ALIAS(Type, Name)                                         \
	if (auto h = get_type_handle<Type>(); h) {                                 \
		m.attr(Name) = h;                                                      \
	} else
