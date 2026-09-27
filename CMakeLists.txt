CMAKE_MINIMUM_REQUIRED ( VERSION 3.24 )

# Choose the installed GPU(s) unless the caller selected target architectures.
# Set this before project() so CUDA compiler detection uses the same setting.
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
	set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to build for")
endif()

PROJECT (KiRaRay
	DESCRIPTION "A simple implememtation of an optix renderer."
	LANGUAGES C CXX CUDA
)
SET(KRR_RENDER_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
SET(KRR_RENDER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(KRR_RENDER_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

###############################################################################
# Options
###############################################################################

OPTION (KRR_BUILD_STARLIGHT 
	"Build some additional code implementation of fancy algorithms."
	ON
)

OPTION (KRR_BUILD_EXAMPLES 
	"Build the additional examples (see misc/samples)."
	OFF
)

OPTION (KRR_RENDER_SPECTRAL 
	"Build the spectral variant of the renderer (WIP)."
	ON
)

include(CTest)
option(KRR_ENABLE_GPU_TESTS "Enable local GPU rendering tests" OFF)
option(KRR_PROFILE_OPTIX "Retain OptiX source information in optimized builds" OFF)
option(KRR_ENABLE_D3D12 "Build the Windows D3D12 graphics backend" ${WIN32})
option(KRR_ENABLE_HYDRA "Build the Blender Hydra renderer" OFF)
option(KRR_ENABLE_USD "Enable native USD scene import" ${KRR_ENABLE_HYDRA})
if(KRR_ENABLE_HYDRA AND NOT KRR_ENABLE_USD)
	message(STATUS "Enabling KRR_ENABLE_USD for the Hydra renderer")
	set(KRR_ENABLE_USD ON CACHE BOOL "Enable native USD scene import" FORCE)
endif()
set(_krr_openvdb_default ON)
set(_krr_python_default ON)
if(KRR_ENABLE_USD)
	set(_krr_openvdb_default OFF)
endif()
if(KRR_ENABLE_HYDRA)
	set(_krr_python_default OFF)
endif()
option(KRR_ENABLE_OPENVDB_IO "Enable scene import through the bundled OpenVDB libraries" ${_krr_openvdb_default})
option(KRR_ENABLE_SDK_TESTS "Enable tests requiring the selected USD SDK" OFF)
option(KRR_ENABLE_PYTHON "Build the existing Python bindings" ${_krr_python_default})
set(KRR_USD_ROOT "" CACHE PATH "Standalone Python-free OpenUSD SDK")
set(KRR_BLENDER_SDK "" CACHE PATH "Development libraries matching the Blender host")
if(KRR_ENABLE_HYDRA)
	if(NOT WIN32 OR KRR_ENABLE_PYTHON)
		message(FATAL_ERROR "The Blender profile requires Windows and KRR_ENABLE_PYTHON=OFF")
	endif()
endif()
if(KRR_ENABLE_USD AND KRR_ENABLE_OPENVDB_IO)
	message(FATAL_ERROR "USD profiles require KRR_ENABLE_OPENVDB_IO=OFF to isolate legacy volume dependencies")
endif()
if(KRR_ENABLE_D3D12 AND NOT WIN32)
	message(FATAL_ERROR "The D3D12 backend requires Windows")
endif()

###############################################################################
# Build type and C++ compiler setup
###############################################################################
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# check if submodules are recursively fetched.
if (NOT IS_DIRECTORY "${KRR_RENDER_ROOT}/src/ext/glfw")
	message(FATAL_ERROR
		"Some external dependencies are missing. "
		"It is most possible that the \"--recursive\" flag is forgotten when cloning this project, "
		"this can be fixed by calling \"git submodule update --init --recursive\"."
	)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_ENABLE_EXTENDED_ALIGNED_STORAGE ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819 /wd4068 /wd4828")	# chn encoding warning

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_DEBUG_POSTFIX "")

if(MSVC)
	# Support mutex initialization with older runtimes bundled by Python environments.
	add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

# the CONFIG inherently comes from CMAKE_BUILD_TYPE
add_compile_definitions ("$<$<CONFIG:Debug>:KRR_DEBUG_BUILD>")
add_compile_definitions ("KRR_BUILD_TYPE=\"$<CONFIG>\"")

###############################################################################
# CUDA compiler setup
###############################################################################

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_library (krr_cuda_cfg INTERFACE)
target_compile_options (
	krr_cuda_cfg
	INTERFACE
		# CUDA's CCCL headers require the conforming MSVC preprocessor.
		"$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:preprocessor>"
		"$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/Zc:preprocessor>"
		"$<$<COMPILE_LANGUAGE:CUDA>:--std=c++17;--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--forward-unknown-to-host-compiler>"
		# Do not add debugging symbols when generating PTX files for OptiX (not supported).
		"$<$<COMPILE_LANGUAGE:CUDA>:$<IF:$<AND:$<CONFIG:Debug>,$<NOT:$<BOOL:$<TARGET_PROPERTY:CUDA_PTX_COMPILATION>>>>,-g,-lineinfo>>"
)

add_library (krr_cuda_warning INTERFACE)
target_compile_options (
	krr_cuda_warning
	INTERFACE
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4244>" # int -> float conversion
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4267>" # size_t -> int conversion
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4305>" # double constant assigned to float
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4552>" # result of expression not used
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4819>" # chn encoding
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4838>" # double -> int conversion
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd4843>" # double -> float conversion
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd26451>" # arithmetic on 4-byte value, then cast to 8-byte
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/wd26495>" # uninitialized member variable
		"$<$<CXX_COMPILER_ID:MSVC>:$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >/D_ENABLE_EXTENDED_ALIGNED_STORAGE>" # conform alignment (c++17)
 
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=partial_override>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=virtual_function_decl_hidden>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=integer_sign_change>"
		"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=declared_but_not_referenced>"		
		#"$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=implicit_return_from_non_void_function>"
)

###############################################################################
# Python bindings, packaging, and tests
###############################################################################
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/pybind11/tools")
set(KRR_PYTHON_PATH "" CACHE STRING "Path to selected python executable")
set(Python_ROOT_DIR ${KRR_PYTHON_PATH})
if(KRR_ENABLE_PYTHON)
	find_package(Python REQUIRED COMPONENTS Interpreter Development)
	set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
	message("Find python at ${Python_INCLUDE_DIRS} : ${Python_LIBRARIES}, selected version: ${Python_VERSION}")
else()
	if(KRR_ENABLE_HYDRA OR (BUILD_TESTING AND KRR_ENABLE_GPU_TESTS))
		find_package(Python REQUIRED COMPONENTS Interpreter)
	elseif(BUILD_TESTING)
		find_package(Python QUIET COMPONENTS Interpreter)
	endif()
	message(STATUS "Python bindings disabled")
endif()

ADD_SUBDIRECTORY (src)
if(BUILD_TESTING)
	add_subdirectory(tests)
endif()
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_DIRS} PARENT_SCOPE)	# exporting KRR_INCLUDE_DIRS to parent scope
endif()
