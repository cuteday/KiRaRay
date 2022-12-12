add_library(samples_common STATIC
	common/VulkanBaseApp.cpp
)
target_include_directories(samples_common PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/common)

target_link_libraries(samples_common PUBLIC
	krr_lib		# this includes glfw and cuda
	krr_cuda_cfg
	krr_cuda_warning
	glfw
	Vulkan::Vulkan
)

add_subdirectory(simple-vulkan)

add_dependencies(samples_common compile_shaders)