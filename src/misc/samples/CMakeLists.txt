add_library(samples_common INTERFACE)
target_include_directories(samples_common INTERFACE 
	${CMAKE_CURRENT_SOURCE_DIR}/common)

add_subdirectory(simple-vulkan)
add_dependencies(samples_common compile_shaders)