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
		-s ${CMAKE_CURRENT_SOURCE_DIR}/unit -p test_*.py -v)
	set_tests_properties(krr_image_metrics PROPERTIES
		LABELS "cpu;unit" TIMEOUT 30
		ENVIRONMENT "PYTHONPATH=${KRR_RENDER_ROOT}/common/scripts")
else()
	message(WARNING "Python was not found; Python image metric tests are unavailable")
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
	krr_gpu_test(krr_readback readback $<TARGET_FILE:krr_test_readback>)
	krr_gpu_test(krr_render_smoke cornell_wavefront_smoke
		${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/smoke.py
		--config ${KRR_TEST_CASE}/config.json
		--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_smoke)
	if(KRR_RENDER_SPECTRAL)
		krr_gpu_test(krr_render_regression cornell_wavefront_regression
			${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/regression.py
			--case ${KRR_TEST_CASE}
			--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_regression)
		set_tests_properties(krr_render_regression PROPERTIES LABELS "gpu;regression")
	endif()
endif()
