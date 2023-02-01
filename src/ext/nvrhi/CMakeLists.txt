cmake_minimum_required(VERSION 3.10)

project(nvrhi VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CMakeDependentOption)

option(NVRHI_WITH_SHADER_COMPILER "Build the NVRHI shader compiler executable" ON)
option(NVRHI_WITH_VALIDATION "Build NVRHI the validation layer" ON)
option(NVRHI_WITH_VULKAN "Build the NVRHI Vulkan backend" ON)

if (MSVC)
    string(REGEX REPLACE "/W[0-4]\\s*" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " /W4 /MP")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if (NVRHI_WITH_VULKAN)
    add_subdirectory(thirdparty/Vulkan-Headers)
endif()

if (NVRHI_WITH_SHADER_COMPILER)
    option(CXXOPTS_BUILD_EXAMPLES OFF)
    option(CXXOPTS_BUILD_TESTS OFF)
    option(CXXOPTS_ENABLE_INSTALL OFF)
    add_subdirectory(thirdparty/cxxopts)

    add_subdirectory(tools/shaderCompiler)
endif()

set(include_common
    include/nvrhi/nvrhi.h
    include/nvrhi/utils.h
    include/nvrhi/common/containers.h
    include/nvrhi/common/misc.h
    include/nvrhi/common/resource.h)
set(src_common
    src/common/format-info.cpp
    src/common/misc.cpp
    src/common/shader-blob.cpp
    src/common/state-tracking.cpp
    src/common/state-tracking.h
    src/common/utils.cpp)

if(MSVC)
    set(misc_common 
        tools/nvrhi.natvis)
else()
    set(misc_common "")
endif()

set(include_validation
    include/nvrhi/validation.h)
set(src_validation
    src/validation/validation-commandlist.cpp
    src/validation/validation-device.cpp
    src/validation/validation-backend.h)

set(include_vk
    include/nvrhi/vulkan.h)
set(src_vk
    src/common/versioning.h
    src/vulkan/vulkan-allocator.cpp
    src/vulkan/vulkan-buffer.cpp
    src/vulkan/vulkan-commandlist.cpp
    src/vulkan/vulkan-compute.cpp
    src/vulkan/vulkan-constants.cpp
    src/vulkan/vulkan-device.cpp
    src/vulkan/vulkan-graphics.cpp
    src/vulkan/vulkan-meshlets.cpp
    src/vulkan/vulkan-queries.cpp
    src/vulkan/vulkan-queue.cpp
    src/vulkan/vulkan-raytracing.cpp
    src/vulkan/vulkan-resource-bindings.cpp
    src/vulkan/vulkan-shader.cpp
    src/vulkan/vulkan-staging-texture.cpp
    src/vulkan/vulkan-state-tracking.cpp
    src/vulkan/vulkan-texture.cpp
    src/vulkan/vulkan-upload.cpp
    src/vulkan/vulkan-backend.h)

add_library(nvrhi STATIC
    ${include_common}
    ${src_common}
    ${misc_common})

if (NVRHI_WITH_VALIDATION)
    target_sources(nvrhi PRIVATE
        ${include_validation}
        ${src_validation})
endif()

target_include_directories(nvrhi PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>)

set_target_properties(nvrhi PROPERTIES FOLDER "NVRHI")

set(nvrhi_vulkan_target nvrhi_vk)

add_library(${nvrhi_vulkan_target} STATIC
    ${include_vk}
    ${src_vk})

set_target_properties(${nvrhi_vulkan_target} PROPERTIES FOLDER "NVRHI")
target_include_directories(${nvrhi_vulkan_target} PRIVATE include)
target_link_libraries(${nvrhi_vulkan_target} PUBLIC Vulkan-Headers nvrhi)