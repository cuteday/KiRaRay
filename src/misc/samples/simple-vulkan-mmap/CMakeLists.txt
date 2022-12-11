add_executable(simple-vulkan-mmap
	main.cpp
	MonteCarloPi.h
	MonteCarloPi.cu
)

target_include_directories(simple-vulkan-mmap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simple-vulkan-mmap
	samples_common
	Vulkan::Vulkan
)
