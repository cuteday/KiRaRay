find_package(Vulkan REQUIRED)

add_library(samples_common STATIC
	common/VulkanBaseApp.cpp
)
target_include_directories(samples_common PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/common)

target_link_libraries(samples_common PUBLIC
	krr_lib		# this includes 3rdparty libraries
	Vulkan::Vulkan
)

add_dependencies(samples_common compile_shaders)

add_executable(simple-vulkan
	main.cpp
	SineWaveSimulation.h
	SineWaveSimulation.cu
)

target_include_directories(simple-vulkan PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simple-vulkan
	samples_common
	krr_lib		
	Vulkan::Vulkan
)
