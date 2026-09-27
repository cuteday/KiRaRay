if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "The initial Hydra integration targets Windows/MSVC")
endif()

add_custom_target(krr_blender_package
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${KRR_RENDER_ROOT}/integrations/blender/kiraray" "${CMAKE_BINARY_DIR}/blender/kiraray"
    COMMAND "${Python_EXECUTABLE}" "${KRR_RENDER_ROOT}/integrations/blender/package.py"
        --source "${CMAKE_BINARY_DIR}/blender/kiraray"
        --libraries "${KRR_BLENDER_SDK}/MaterialX/libraries"
        --output "${CMAKE_BINARY_DIR}/blender/kiraray-blender-5.2-windows-x64.zip"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/blender/kiraray"
    DEPENDS hdKiRaRay)
if(KRR_ENABLE_OPENVDB_IO)
    message(FATAL_ERROR "The Blender host build requires KRR_ENABLE_OPENVDB_IO=OFF")
endif()

add_library(hdKiRaRay SHARED delegate.cpp bridge.cpp)
target_compile_features(hdKiRaRay PRIVATE cxx_std_17)
target_compile_options(hdKiRaRay PRIVATE /EHsc /bigobj)
target_compile_definitions(hdKiRaRay PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_include_directories(hdKiRaRay PRIVATE "${KRR_RENDER_ROOT}")
target_link_libraries(hdKiRaRay PRIVATE krr_lib krr_usd_sdk "-WHOLEARCHIVE:$<TARGET_FILE:krr_lib>")
set_target_properties(hdKiRaRay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/blender/kiraray/native"
    MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
add_custom_command(TARGET hdKiRaRay POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/plugInfo.json" "$<TARGET_FILE_DIR:hdKiRaRay>/plugInfo.json"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${KRR_RENDER_ROOT}/integrations/blender/kiraray" "${CMAKE_BINARY_DIR}/blender/kiraray"
    COMMAND "${Python_EXECUTABLE}" "${KRR_RENDER_ROOT}/integrations/blender/package.py"
        --source "${CMAKE_BINARY_DIR}/blender/kiraray" --libraries "${KRR_BLENDER_SDK}/MaterialX/libraries" --stage-only)
if(EXISTS "${KRR_VULKAN_ROOT}/Bin/dxcompiler.dll")
    add_custom_command(TARGET hdKiRaRay POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${KRR_VULKAN_ROOT}/Bin/dxcompiler.dll" "$<TARGET_FILE_DIR:hdKiRaRay>/dxcompiler.dll")
endif()
