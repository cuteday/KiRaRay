function(krr_cpu_renderer_test target)
	if(MSVC AND CUDAToolkit_VERSION VERSION_LESS 13)
		# CUDA 12 imports the driver eagerly even when these tests never use it.
		target_link_options(${target} PRIVATE /DELAYLOAD:nvcuda.dll)
		target_link_libraries(${target} PRIVATE delayimp)
		target_sources(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/support/no_cuda_driver.cpp)
	endif()
endfunction()

add_executable(krr_test_publication unit/test_publication.cpp)
target_include_directories(krr_test_publication PRIVATE ${KRR_RENDER_ROOT})
add_test(NAME krr_publication COMMAND krr_test_publication)
set_tests_properties(krr_publication PROPERTIES LABELS "cpu;unit" TIMEOUT 30)

add_executable(krr_test_exr unit/test_exr.cpp)
target_include_directories(krr_test_exr PRIVATE ${KRR_RENDER_SOURCE_DIR})
target_link_libraries(krr_test_exr PRIVATE krr_exr OpenEXR::OpenEXRCore)
add_test(NAME krr_exr COMMAND krr_test_exr ${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>/exr)
set_tests_properties(krr_exr PROPERTIES LABELS "cpu;unit" TIMEOUT 30)

add_executable(krr_test_sampler unit/test_sampler.cpp)
target_include_directories(krr_test_sampler PRIVATE ${KRR_INCLUDE_DIRS})
target_link_libraries(krr_test_sampler PRIVATE krr_math krr_cuda_cfg)

add_executable(krr_test_camera unit/test_camera.cpp)
target_include_directories(krr_test_camera PRIVATE ${KRR_INCLUDE_DIRS})
target_link_libraries(krr_test_camera PRIVATE krr_math krr_cuda_cfg)

add_executable(krr_test_render_options unit/test_render_options.cpp)
target_include_directories(krr_test_render_options PRIVATE
	${KRR_RENDER_SOURCE_DIR} ${KRR_RENDER_SOURCE_DIR}/ext/json)

add_executable(krr_test_material_program unit/test_material_program.cpp
	${KRR_RENDER_SOURCE_DIR}/core/material/description.cpp)
target_include_directories(krr_test_material_program PRIVATE ${KRR_INCLUDE_DIRS})
target_link_libraries(krr_test_material_program PRIVATE krr_math krr_cuda_cfg)

add_executable(krr_test_openpbr unit/test_openpbr.cpp)
target_include_directories(krr_test_openpbr PRIVATE ${KRR_INCLUDE_DIRS})
target_link_libraries(krr_test_openpbr PRIVATE krr_math krr_cuda_cfg)

add_test(NAME krr_sampler COMMAND krr_test_sampler)
add_test(NAME krr_camera COMMAND krr_test_camera)
add_test(NAME krr_render_options COMMAND krr_test_render_options)
add_test(NAME krr_material_program COMMAND krr_test_material_program)
add_test(NAME krr_openpbr COMMAND krr_test_openpbr)
set_tests_properties(krr_openpbr PROPERTIES LABELS "cpu;unit;material" TIMEOUT 30)
if(KRR_ENABLE_SDK_TESTS AND NOT KRR_ENABLE_HYDRA)
	if(NOT KRR_ENABLE_USD)
		message(FATAL_ERROR "KRR_ENABLE_SDK_TESTS requires a USD-enabled build")
	endif()
	add_executable(krr_test_usd unit/test_usd.cpp)
	target_link_libraries(krr_test_usd PRIVATE krr_lib krr_usd_sdk)
	krr_cpu_renderer_test(krr_test_usd)
	add_test(NAME krr_usd COMMAND krr_test_usd
		${CMAKE_CURRENT_SOURCE_DIR}/fixtures/usd ${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>/usd)
	set_tests_properties(krr_usd PROPERTIES LABELS "cpu;unit;sdk;usd" TIMEOUT 60)
endif()

if(KRR_ENABLE_HYDRA)
	set(KRR_BLENDER_EXECUTABLE "" CACHE FILEPATH "Pinned Blender 5.2.2 executable for integration tests")
	if(NOT KRR_BLENDER_EXECUTABLE)
		if(NOT "$ENV{KRR_BLENDER_EXECUTABLE}" STREQUAL "")
			set(KRR_BLENDER_EXECUTABLE "$ENV{KRR_BLENDER_EXECUTABLE}" CACHE FILEPATH "Pinned Blender 5.2.2 executable for integration tests" FORCE)
		elseif(EXISTS "${KRR_RENDER_ROOT}/build/blender-5.2/blender-5.2.2-windows-x64/blender.exe")
			set(KRR_BLENDER_EXECUTABLE "${KRR_RENDER_ROOT}/build/blender-5.2/blender-5.2.2-windows-x64/blender.exe" CACHE FILEPATH "Pinned Blender 5.2.2 executable for integration tests" FORCE)
		endif()
	endif()
	if(EXISTS "${KRR_BLENDER_EXECUTABLE}")
		get_filename_component(KRR_BLENDER_RUNTIME "${KRR_BLENDER_EXECUTABLE}" DIRECTORY)
	elseif(KRR_ENABLE_SDK_TESTS)
		message(FATAL_ERROR "Host SDK tests require KRR_BLENDER_EXECUTABLE (Blender 5.2.2)")
	else()
		message(WARNING "Renderer-linked CPU tests require the Blender runtime; set KRR_BLENDER_EXECUTABLE before running them. The Hydra package can still be built.")
	endif()
endif()

if(KRR_ENABLE_SDK_TESTS AND KRR_ENABLE_HYDRA)
	set(KRR_BLENDER_TEST_CASES export_scene)
	if(KRR_ENABLE_GPU_TESTS)
		list(APPEND KRR_BLENDER_TEST_CASES render_scene emission environment)
		if(KRR_ENABLE_D3D12)
			list(APPEND KRR_BLENDER_TEST_CASES render_scene_d3d12)
		endif()
	endif()
	foreach(case IN LISTS KRR_BLENDER_TEST_CASES)
		set(artifacts "${CMAKE_CURRENT_BINARY_DIR}/artifacts/$<CONFIG>/blender_${case}")
		set(script "${case}")
		set(case_args "")
		if(case STREQUAL "render_scene_d3d12")
			set(script render_scene)
			set(case_args --graphics-api d3d12)
		endif()
		add_test(NAME krr_blender_${case} COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/run_test.py"
			--artifacts "${artifacts}" --timeout 210 -- "${KRR_BLENDER_EXECUTABLE}"
			--background --factory-startup --python-exit-code 1
			--python "${KRR_RENDER_ROOT}/integrations/blender/probe/${script}.py" --
			--addon-dir "${CMAKE_BINARY_DIR}/blender/kiraray"
			--artifacts "${artifacts}" ${case_args})
		set_tests_properties(krr_blender_${case} PROPERTIES LABELS "sdk;blender" TIMEOUT 240)
	endforeach()
	if(KRR_ENABLE_GPU_TESTS)
		set_tests_properties(krr_blender_render_scene krr_blender_emission krr_blender_environment PROPERTIES LABELS "sdk;blender;gpu;smoke" RESOURCE_LOCK krr_gpu)
		if(KRR_ENABLE_D3D12)
			set_tests_properties(krr_blender_render_scene_d3d12 PROPERTIES LABELS "sdk;blender;gpu;smoke;d3d12" RESOURCE_LOCK krr_gpu)
		endif()
	endif()
	set_tests_properties(krr_blender_export_scene PROPERTIES LABELS "sdk;blender;cpu")
endif()
set_tests_properties(krr_sampler krr_render_options krr_material_program PROPERTIES LABELS "cpu;unit" TIMEOUT 30)
set_tests_properties(krr_camera PROPERTIES LABELS "cpu;unit" TIMEOUT 30)

add_executable(krr_test_scene_interop unit/test_scene_interop.cpp)
target_link_libraries(krr_test_scene_interop PRIVATE krr_lib)
krr_cpu_renderer_test(krr_test_scene_interop)
add_test(NAME krr_scene_interop COMMAND krr_test_scene_interop)
set_tests_properties(krr_scene_interop PROPERTIES LABELS "cpu;unit" TIMEOUT 30
	ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${KRR_VULKAN_ROOT}/Bin")

if(NOT KRR_ENABLE_OPENVDB_IO)
	add_executable(krr_test_volume_import unit/test_volume_import.cpp)
	target_link_libraries(krr_test_volume_import PRIVATE krr_lib)
	krr_cpu_renderer_test(krr_test_volume_import)
	add_test(NAME krr_volume_import COMMAND krr_test_volume_import)
	set_tests_properties(krr_volume_import PROPERTIES LABELS "cpu;unit" TIMEOUT 30)
	set_tests_properties(krr_volume_import PROPERTIES
		ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${KRR_VULKAN_ROOT}/Bin")
endif()

foreach(test krr_scene_interop krr_volume_import krr_usd)
	if(TEST ${test})
		if(WIN32)
			set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
		endif()
		if(KRR_ENABLE_HYDRA AND KRR_BLENDER_RUNTIME)
			set_property(TEST ${test} APPEND PROPERTY ENVIRONMENT_MODIFICATION
				"PATH=path_list_prepend:${KRR_BLENDER_RUNTIME}"
				"PATH=path_list_prepend:${KRR_BLENDER_RUNTIME}/blender.shared")
		endif()
	endif()
endforeach()

if(Python_FOUND)
	add_test(NAME krr_image_metrics
		COMMAND ${Python_EXECUTABLE} -m unittest discover
		-s ${CMAKE_CURRENT_SOURCE_DIR}/unit -p test_image_metrics.py -v)
	add_test(NAME krr_benchmarks
		COMMAND ${Python_EXECUTABLE} -m unittest discover
		-s ${CMAKE_CURRENT_SOURCE_DIR}/unit -p test_benchmarks.py -v)
	add_test(NAME krr_blender_preflight
		COMMAND ${Python_EXECUTABLE} -m unittest discover
		-s ${CMAKE_CURRENT_SOURCE_DIR}/unit -p test_blender_preflight.py -v)
	set_tests_properties(krr_image_metrics krr_benchmarks krr_blender_preflight PROPERTIES
		LABELS "cpu;unit" TIMEOUT 30
		ENVIRONMENT "PYTHONPATH=${KRR_RENDER_ROOT}/common/scripts")
	set_tests_properties(krr_benchmarks PROPERTIES LABELS "cpu;unit;benchmark")
	set_tests_properties(krr_blender_preflight PROPERTIES LABELS "cpu;unit;blender")
else()
	message(WARNING "Python was not found; Python unit tests are unavailable")
endif()

if(KRR_ENABLE_GPU_TESTS AND NOT KRR_ENABLE_HYDRA)
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

	add_executable(krr_test_readback render/test_readback.cu render/test_textureloader.cpp)
	target_link_libraries(krr_test_readback PRIVATE krr_lib)
	add_executable(krr_test_session render/test_session.cpp)
	target_link_libraries(krr_test_session PRIVATE krr_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_lib>)
	add_executable(krr_test_materials render/test_materials.cu)
	target_link_libraries(krr_test_materials PRIVATE krr_lib)
	krr_gpu_test(krr_materials materials $<TARGET_FILE:krr_test_materials>)
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
		krr_gpu_test(krr_session${suffix} session${suffix}
			$<TARGET_FILE:krr_test_session> ${api})
		krr_gpu_test(krr_render_smoke${suffix} cornell_wavefront_smoke${suffix}
			${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/smoke.py
			--config ${KRR_TEST_CASE}/config.json --graphics-api ${api}
			--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_smoke${suffix})
		krr_gpu_test(krr_benchmark_smoke${suffix} cornell_wavefront_benchmark${suffix}
			${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/benchmark_smoke.py
			--config ${KRR_TEST_CASE}/config.json --graphics-api ${api}
			--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_benchmark${suffix})
		set_tests_properties(krr_readback${suffix} krr_session${suffix} krr_render_smoke${suffix}
			PROPERTIES LABELS "gpu;smoke;${api}")
		set_tests_properties(krr_benchmark_smoke${suffix} PROPERTIES LABELS "gpu;smoke;benchmark;${api}")
		if(KRR_RENDER_SPECTRAL)
			krr_gpu_test(krr_render_regression${suffix} cornell_wavefront_regression${suffix}
				${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/regression.py
				--case ${KRR_TEST_CASE} --graphics-api ${api}
				--artifacts ${KRR_TEST_ARTIFACTS}/cornell_wavefront_regression${suffix})
			set_tests_properties(krr_render_regression${suffix} PROPERTIES LABELS "gpu;regression;${api}")
		endif()
		if(KRR_ENABLE_SDK_TESTS AND KRR_ENABLE_USD AND NOT KRR_ENABLE_HYDRA)
			set(usd_case ${CMAKE_CURRENT_SOURCE_DIR}/cases/cornell_usd)
			krr_gpu_test(krr_usd_smoke${suffix} cornell_usd_smoke${suffix}
				${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/smoke.py
				--config ${usd_case}/config.json --graphics-api ${api}
				--artifacts ${KRR_TEST_ARTIFACTS}/cornell_usd_smoke${suffix})
			set_tests_properties(krr_usd_smoke${suffix} PROPERTIES LABELS "gpu;smoke;sdk;usd;${api}")
			if(KRR_RENDER_SPECTRAL)
				krr_gpu_test(krr_usd_regression${suffix} cornell_usd_regression${suffix}
					${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/render/regression.py
					--case ${usd_case} --graphics-api ${api}
					--artifacts ${KRR_TEST_ARTIFACTS}/cornell_usd_regression${suffix})
				set_tests_properties(krr_usd_regression${suffix} PROPERTIES LABELS "gpu;regression;sdk;usd;${api}")
			endif()
		endif()
	endforeach()
endif()
