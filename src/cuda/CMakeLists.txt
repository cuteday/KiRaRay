###############################################################################
# Source files
###############################################################################

set(SOURCES
	common.h
	test.cu
)

###############################################################################
# linking / library
###############################################################################

add_library(krr-cuda STATIC ${SOURCES})
target_link_libraries(krr-cuda PUBLIC ${CUDA_LIBRARIES} cuda cublas)
target_compile_options(krr-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)