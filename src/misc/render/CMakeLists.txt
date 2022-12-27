# Then there are some misc files

set ( KRR_PPG_PT
	ppg/backend.h
	ppg/backend.cpp
	ppg/guideditem.h
	ppg/integrator.h
	ppg/integrator.cpp
	ppg/tree.h
	ppg/treemanip.cpp
	ppg/ppg.h
)

SET_SOURCE_FILES_PROPERTIES (
	${CMAKE_CURRENT_SOURCE_DIR}/ppg/integrator.cpp
	PROPERTIES LANGUAGE CUDA
)

add_custom_command (OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/include/ppg/guideditem_soa.h
    COMMAND soac ${CMAKE_CURRENT_SOURCE_DIR}/ppg/guideditem.soa > ${CMAKE_CURRENT_BINARY_DIR}/include/ppg/guideditem_soa.h
    DEPENDS soac ${CMAKE_CURRENT_SOURCE_DIR}/ppg/guideditem.soa)

set (KRR_MISC_SOA_GENERATED 
	${CMAKE_CURRENT_BINARY_DIR}/include/ppg/guideditem_soa.h
)

add_custom_target (krr_misc_soa_generated DEPENDS ${KRR_MISC_SOA_GENERATED})

set ( KRR_MISC_INCLUDE_ALL
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}/include
	${KRR_INCLUDE_ALL}
)
include_directories(${KRR_MISC_INCLUDE_ALL})
set (KRR_MISC_INCLUDE_ALL ${KRR_MISC_INCLUDE_ALL} PARENT_SCOPE)
CUDA_COMPILE_EMBED(PPG_PTX ppg/device.cu krr-ppg krr_misc_soa_generated ${KRR_MISC_INCLUDE_ALL})
add_library(KRR_MISC_PTX STATIC
	# peseudo target for SOA generation priority
	${PPG_PTX}
)
add_dependencies(KRR_MISC_PTX krr_misc_soa_generated)

set ( KRR_MISC_LIBS_ALL
	#krr_lib
	krr_cuda_cfg
	krr_cuda_warning
	KRR_MISC_PTX
)
add_library (krr_starlight_lib STATIC
	${KRR_PPG_PT}
	# SOA
	${KRR_MISC_SOA_GENERATED}
	# PTX
	${PPG_PTX}
)
add_dependencies(krr_starlight_lib krr_misc_soa_generated)
set_property(TARGET krr_starlight_lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
TARGET_COMPILE_DEFINITIONS (krr_starlight_lib PUBLIC ${KRR_DEFINITIONS})
target_link_libraries (krr_starlight_lib PRIVATE ${KRR_MISC_LIBS_ALL})
target_include_directories(krr_starlight_lib PUBLIC ${KRR_MISC_INCLUDE_ALL})