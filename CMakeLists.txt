CMAKE_MINIMUM_REQUIRED ( VERSION 3.17 )

PROJECT (KIRARAY
	DESCRIPTION "A codebase for rendering odds&ends.")
SET (CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

###############################################################################
# CUDA compiler setup
###############################################################################
enable_language(CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if (MSVC)
else()
	set(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/targets/x86_64-linux)
endif()
find_library(
	CUDA_CUBLASLT_LIBRARY cublasLt
	${CUDA_TOOLKIT_ROOT_DIR}/lib64
	${CUDA_TOOLKIT_ROOT_DIR}/lib
)

# auto detect compute arch, from https://stackoverflow.com/a/69353718
include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
message(STATUS "Automatically detected GPU architectures: ${CUDA_ARCH_LIST}")

###############################################################################
# include files
###############################################################################
ADD_SUBDIRECTORY(src/ext)
ADD_SUBDIRECTORY(src/cuda)

SET ( KRR_CORE_SOURCE
	src/core/light.cpp
)

SET ( KRR_CORE_HEADERS
	src/core/kiraray.h
)

FILE ( GLOB KRR_SOURCE
	src/ext/*
)

INCLUDE_DIRECTORIES ( 
	src 
	src/core
	src/cuda
	${ASSIMP_INCLUDE_DIRS}
	${ENOKI_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
)

###############################################################################
# linking and executables
###############################################################################
ADD_LIBRARY ( krr STATIC
	${KRR_CORE_SOURCE}
	${KRR_CORE_HEADERS}
	${KRR_SOURCE}
)

SET(ALL_KRR_LIBS
	krr
	krr-cuda
	assimp
)

ADD_EXECUTABLE ( kiraray src/main/kiraray.cpp )
TARGET_LINK_LIBRARIES (kiraray ${ALL_KRR_LIBS} )