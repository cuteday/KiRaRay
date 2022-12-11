add_executable(simple-vulkan
	main.cpp
	SineWaveSimulation.h
	SineWaveSimulation.cu
)

target_include_directories(simple-vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simple-vulkan
	samples_common
	krr_lib		# this includes glfw and cuda
	krr_cuda_cfg
	krr_cuda_warning
	glfw
	Vulkan::Vulkan
)
