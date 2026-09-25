################################################################################
# Optix
################################################################################
if(OptiX_INSTALL_DIR)
	# An explicit CMake selection takes precedence over the environment.
	set(OPTIX_ROOT "${OptiX_INSTALL_DIR}")
elseif(DEFINED ENV{OptiX_INSTALL_DIR} AND NOT "$ENV{OptiX_INSTALL_DIR}" STREQUAL "")
	set(OPTIX_ROOT "$ENV{OptiX_INSTALL_DIR}")
elseif(DEFINED ENV{OPTIX_ROOT} AND NOT "$ENV{OPTIX_ROOT}" STREQUAL "")
	set(OPTIX_ROOT "$ENV{OPTIX_ROOT}")
else()
	# Prefer the newest installed SDK when no version was selected.
	file(TO_CMAKE_PATH "$ENV{PROGRAMDATA}" PROGRAMDATA)
	file(GLOB OPTIX_ROOT "${PROGRAMDATA}/NVIDIA Corporation/OptiX SDK*")
	if(NOT OPTIX_ROOT)
		message(FATAL_ERROR "OptiX SDK not found. Set OptiX_INSTALL_DIR to the SDK root.")
	endif()
	list(SORT OPTIX_ROOT COMPARE NATURAL ORDER DESCENDING)
	list(GET OPTIX_ROOT 0 OPTIX_ROOT)
endif()

file(TO_CMAKE_PATH "${OPTIX_ROOT}" OPTIX_ROOT)
if(NOT EXISTS "${OPTIX_ROOT}/include/optix.h")
	message(FATAL_ERROR "OptiX headers not found at ${OPTIX_ROOT}/include. Set OptiX_INSTALL_DIR to the SDK root.")
endif()
message(STATUS "Found OptiX SDK at ${OPTIX_ROOT}")
set(OptiX_INSTALL_DIR "${OPTIX_ROOT}" CACHE PATH "Path to OptiX installation location")
# Refresh this derived path when selecting another SDK in an existing build.
set(OptiX_INCLUDE_DIR "${OPTIX_ROOT}/include" CACHE PATH "Path to OptiX include directory" FORCE)
