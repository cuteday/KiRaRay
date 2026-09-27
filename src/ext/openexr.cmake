include(FetchContent)

function(krr_add_openexr)
	set(BUILD_SHARED_LIBS OFF)
	set(BUILD_TESTING OFF)
	set(BUILD_WEBSITE OFF)
	set(PYTHON OFF)
	set(PYBIND11 OFF)
	set(IMATH_INSTALL OFF)
	set(IMATH_INSTALL_PKG_CONFIG OFF)
	set(IMATH_HALF_USE_LOOKUP_TABLE OFF)
	set(IMATH_NAMESPACE_CUSTOM 1 CACHE STRING "Private EXR decoder namespace" FORCE)
	set(IMATH_INTERNAL_NAMESPACE krr_imath_3_2 CACHE STRING "Private EXR decoder namespace" FORCE)

	FetchContent_Declare(krr_imath
		URL https://codeload.github.com/AcademySoftwareFoundation/Imath/tar.gz/refs/tags/v3.2.2
		URL_HASH SHA256=b4275d83fb95521510e389b8d13af10298ed5bed1c8e13efd961d91b1105e462
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
	FetchContent_MakeAvailable(krr_imath)
	set_property(DIRECTORY "${krr_imath_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)

	set(OPENEXR_INSTALL OFF)
	set(OPENEXR_INSTALL_TOOLS OFF)
	set(OPENEXR_INSTALL_PKG_CONFIG OFF)
	set(OPENEXR_BUILD_TOOLS OFF)
	set(OPENEXR_BUILD_EXAMPLES OFF)
	set(OPENEXR_BUILD_PYTHON OFF)
	set(OPENEXR_FORCE_EMBEDDED_CORE ON)
	set(OPENEXR_FORCE_INTERNAL_IMATH ON)
	set(OPENEXR_FORCE_INTERNAL_DEFLATE ON)
	set(OPENEXR_FORCE_INTERNAL_OPENJPH ON)
	FetchContent_Declare(krr_openexr
		URL https://codeload.github.com/AcademySoftwareFoundation/openexr/tar.gz/refs/tags/v3.4.13
		URL_HASH SHA256=1ed0cee48ac8c77da235c8ca8ab85d031d43cd790eda36af87fed4cf316cf2df
		DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
	FetchContent_MakeAvailable(krr_openexr)
	set_property(DIRECTORY "${krr_openexr_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
endfunction()

# The static C decoder supports DWA without depending on Blender's OpenEXR ABI.
krr_add_openexr()
