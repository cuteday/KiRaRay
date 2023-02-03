add_executable(sinewave
	sinewave.cpp
	devicemanager.cpp
	SineWaveSimulation.cu
)

target_include_directories(sinewave PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sinewave krr_lib)

add_executable(triangle
	triangle.cpp
	devicemanager.cpp
)
target_include_directories(triangle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(triangle krr_lib)