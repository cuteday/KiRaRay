  CMAKE_MINIMUM_REQUIRED ( VERSION 3.19 )

PROJECT (KiRaRay
	DESCRIPTION "A simple implememtation of an optix renderer."
	LANGUAGES C CXX CUDA
)
SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(KRR_PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

###############################################################################
# Build type and C++ compiler setup
###############################################################################
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# the CONFIG inherently comes from CMAKE_BUILD_TYPE
add_compile_definitions ("$<$<CONFIG:Debug>:KRR_DEBUG_BUILD>")

###############################################################################
# CUDA compiler setup
###############################################################################

find_package(CUDA REQUIRED)
enable_language (CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

set(CUDA_VERBOSE_BUILD ON)

if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

if (NOT CUDA_INCLUDE_DIRS)
	message(STATUS "Setting CUDA include directory to: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

list (APPEND KRR_DEFINITIONS "KRR_ON_GPU")
# auto detect compute arch, from https://stackoverflow.com/a/69353718
if (NOT DEFINED ENV{CUDA_ARCH})
	include(FindCUDA/select_compute_arch)
	CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
	string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
	string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
	string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
	list(REMOVE_ITEM CUDA_ARCH_LIST "86+PTX")
	list(SORT CUDA_ARCH_LIST)
	message(STATUS "Automatically detected GPU architecture: ${CUDA_ARCH_LIST}")
	list(POP_BACK CUDA_ARCH_LIST CMAKE_CUDA_ARCHITECTURES)
	string(APPEND CMAKE_CUDA_ARCHITECTURES "-virtual")
	message(STATUS "Adopted GPU architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  	message(STATUS "Specified GPU architecture: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if (MSVC)
else()
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
endif()
list(APPEND CUDA_NVCC_FLAGS "--std=c++17")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
list(APPEND CUDA_NVCC_FLAGS "--forward-unknown-to-host-compiler")

add_library (krr_cuda_cfg INTERFACE)
target_compile_options (
	krr_cuda_cfg
	INTERFACE
		"$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>"
		# Do not add debugging symbols when generating PTX files for OptiX (not supported).
		"$<$<COMPILE_LANGUAGE:CUDA>:$<IF:$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>>,-G;-g,-lineinfo>>"
)

add_library (krr_cuda_warning INTERFACE)
target_compile_options (
    krr_cuda_warning
    INTERFACE
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4244>" # int -> float conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4267>" # size_t -> int conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4305>" # double constant assigned to float
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4552>" # result of expression not used
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4838>" # double -> int conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4843>" # double -> float conversion
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd26451>" # arithmetic on 4-byte value, then cast to 8-byte
        "$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd26495>" # uninitialized member variable

		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=partial_override>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=virtual_function_decl_hidden>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=integer_sign_change>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=declared_but_not_referenced>"		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=implicit_return_from_non_void_function>"
)

###############################################################################
# platform
###############################################################################
if(WIN32)
	set(KRR_PLATFORM_WINDOWS 1)
elseif(UNIX AND NOT APPLE)
	set(KRR_PLATFORM_LINUX 1)
else()
	set(KRR_PLATFORM_KNOWN 1)
endif()

###############################################################################
# other dependencies
###############################################################################
INCLUDE (${KRR_PROJECT_ROOT}/common/cmake/FindOptiX.cmake)
find_package(OpenGL REQUIRED)

ADD_SUBDIRECTORY (src)