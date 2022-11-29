# Then there are some misc files

set ( KRR_PPG_PT
	render/ppg/backend.h
	render/ppg/backend.cpp
	render/ppg/guideditem.h
	render/ppg/integrator.h
	render/ppg/integrator.cpp
	render/ppg/tree.h
	render/ppg/treemanip.cpp
	render/ppg/ppg.h
)

SET_SOURCE_FILES_PROPERTIES (
	render/ppg/integrator.cpp
	PROPERTIES LANGUAGE CUDA
)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/render/ppg/guideditem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/render/ppg/guideditem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/render/ppg/guideditem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/render/ppg/guideditem.soa)

set (KRR_MISC_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/render/ppg/guideditem_soa.h
)

add_custom_target (krr_misc_soa_generated DEPENDS ${KRR_MISC_SOA_GENERATED})

set ( KRR_MISC_INCLUDE_ALL
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/include
	${KRR_INCLUDE_ALL}
)
set (KRR_MISC_INCLUDE_ALL ${KRR_MISC_INCLUDE_ALL} PARENT_SCOPE)
CUDA_COMPILE_EMBED(PPG_PTX render/ppg/device.cu krr-ppg ${KRR_MISC_INCLUDE_ALL})
add_library(KRR_MISC_PTX STATIC
	# peseudo target for SOA generation priority
	${PPG_PTX}
)

set ( KRR_MISC_LIBS_ALL
	krr_lib
	krr_cuda_cfg
	krr_cuda_warning
	KRR_MISC_PTX
)
add_library (krr_starlight_lib
	${KRR_PPG_PT}
)
add_dependencies(KRR_MISC_PTX krr_misc_soa_generated)
set_property(TARGET krr_starlight_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries (krr_starlight_lib PRIVATE ${KRR_MISC_LIBS_ALL})
target_include_directories(krr_starlight_lib PUBLIC ${KRR_MISC_INCLUDE_ALL})