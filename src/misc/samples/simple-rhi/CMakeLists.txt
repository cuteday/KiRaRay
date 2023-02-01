add_executable(simple-rhi
	main.cpp
)

target_include_directories(simple-rhi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(simple-rhi krr_lib)