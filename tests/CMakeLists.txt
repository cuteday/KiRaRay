add_executable(krr_test_sampler unit/test_sampler.cpp)
target_include_directories(krr_test_sampler PRIVATE ${KRR_INCLUDE_DIRS})
target_link_libraries(krr_test_sampler PRIVATE krr_math krr_cuda_cfg)

add_executable(krr_test_render_options unit/test_render_options.cpp)
target_include_directories(krr_test_render_options PRIVATE
	${KRR_RENDER_SOURCE_DIR} ${KRR_RENDER_SOURCE_DIR}/ext/json)

add_test(NAME krr_sampler COMMAND krr_test_sampler)
add_test(NAME krr_render_options COMMAND krr_test_render_options)
set_tests_properties(krr_sampler krr_render_options PROPERTIES LABELS "cpu;unit" TIMEOUT 30)

if(Python_FOUND)
	add_test(NAME krr_image_metrics
		COMMAND ${Python_EXECUTABLE} -m unittest discover
		-s ${CMAKE_CURRENT_SOURCE_DIR}/unit -p test_image_metrics.py -v)
	add_test(NAME krr_benchmarks
		COMMAND ${Python_EXECUTABLE} -m unittest discover
		-s ${CMAKE_CURRENT_SOURCE_DIR}/unit -p test_benchmarks.py -v)
	set_tests_properties(krr_image_metrics krr_benchmarks PROPERTIES
		LABELS "cpu;unit" TIMEOUT 30
		ENVIRONMENT "PYTHONPATH=${KRR_RENDER_ROOT}/common/scripts")
	set_tests_properties(krr_benchmarks PROPERTIES LABELS "cpu;unit;benchmark")
else()
	message(WARNING "Python was not found; Python unit tests are unavailable")
endif()

if(KRR_ENABLE_GPU_TESTS)
	if(NOT TARGET pykrr)
		message(FATAL_ERROR "GPU tests require the Python bindings; select Python_EXECUTABLE")
	endif()

	set(KRR_TEST_CASE ${CMAKE_CURRENT_SOURCE_DIR}/cases/cornell_wavefront)
	set(KRR_TEST_ARTIFACTS ${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>)
	set(KRR_TEST_RUNNER ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py)
	function(krr_gpu_test name artifact)
		add_test(NAME ${name} COMMAND ${Python_EXECUTABLE} ${KRR_TEST_RUNNER}
			--artifacts ${KRR_TEST_ARTIFACTS}/${artifact} --timeout 240 -- ${ARGN})
		set_tests_properties(${name} PROPERTIES
			LABELS "gpu;smoke" TIMEOUT 270 RESOURCE_LOCK krr_gpu
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			ENVIRONMENT "PYTHONPATH=${KRR_RENDER_ROOT}/common/scripts;KRR_MODULE_DIR=$<TARGET_FILE_DIR:pykrr>;KRR_BUILD_DIR=${CMAKE_BINARY_DIR}"
			ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${KRR_VULKAN_ROOT}/Bin")
	endfunction()

	add_executable(krr_test_readback render/test_readback.cu)
	target_link_libraries(krr_test_readback PRIVATE krr_lib)
	set(KRR_TEST_GRAPHICS_APIS vulkan)
	if(KRR_ENABLE_D3D12)
		list(APPEND KRR_TEST_GRAPHICS_APIS d3d12)
	endif()
	foreach(api IN LISTS KRR_TEST_GRAPHICS_APIS)
		set(suffix "")
		if(NOT api STREQUAL "vulkan")
			set(suffix "_${api}")
		endif()
		krr_gpu_test(krr_readback${suffix} readback${suffix}
			$<TARGET_FILE:krr_test_readback> ${api})
		krr_gpu_test(krr_render_smoke${suffix} cornell_wavefront_smoke${suffix}
			${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/smoke.py
			--config ${KRR_TEST_CASE}/config.json --graphics-api ${api}
			--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_smoke${suffix})
		krr_gpu_test(krr_benchmark_smoke${suffix} cornell_wavefront_benchmark${suffix}
			${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/benchmark_smoke.py
			--config ${KRR_TEST_CASE}/config.json --graphics-api ${api}
			--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_benchmark${suffix})
		set_tests_properties(krr_readback${suffix} krr_render_smoke${suffix}
			PROPERTIES LABELS "gpu;smoke;${api}")
		set_tests_properties(krr_benchmark_smoke${suffix} PROPERTIES LABELS "gpu;smoke;benchmark;${api}")
		if(KRR_RENDER_SPECTRAL)
			krr_gpu_test(krr_render_regression${suffix} cornell_wavefront_regression${suffix}
				${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/regression.py
				--case ${KRR_TEST_CASE} --graphics-api ${api}
				--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_regression${suffix})
			set_tests_properties(krr_render_regression${suffix} PROPERTIES LABELS "gpu;regression;${api}")
		endif()
	endforeach()
endif()
