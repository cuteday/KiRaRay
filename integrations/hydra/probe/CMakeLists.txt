cmake_minimum_required(VERSION 3.24)
project(KiRaRayHydraProbe LANGUAGES CXX)

set(KRR_BLENDER_SDK "" CACHE PATH "Blender's matching lib-windows_x64 checkout")
if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "The initial Blender compatibility probe targets Windows/MSVC")
endif()
foreach(path usd/include/pxr/pxr.h usd/lib/usd_ms.lib tbb/include/tbb/tbb.h)
    if(NOT EXISTS "${KRR_BLENDER_SDK}/${path}")
        message(FATAL_ERROR "Missing Blender SDK file: ${KRR_BLENDER_SDK}/${path}")
    endif()
endforeach()

add_library(hdKrrProbe SHARED probe.cpp)
target_compile_features(hdKrrProbe PRIVATE cxx_std_17)
target_compile_options(hdKrrProbe PRIVATE /EHsc /bigobj)
target_compile_definitions(hdKrrProbe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(hdKrrProbe PRIVATE
    "${KRR_BLENDER_SDK}/usd/include"
    "${KRR_BLENDER_SDK}/tbb/include"
    "${KRR_BLENDER_SDK}/python/313/include")
target_link_libraries(hdKrrProbe PRIVATE "${KRR_BLENDER_SDK}/usd/lib/usd_ms.lib")
target_link_directories(hdKrrProbe PRIVATE
    "${KRR_BLENDER_SDK}/python/313/libs"
    "${KRR_BLENDER_SDK}/tbb/lib")
set_target_properties(hdKrrProbe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugin"
    MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
configure_file(plugInfo.json "${CMAKE_BINARY_DIR}/plugin/plugInfo.json" COPYONLY)
