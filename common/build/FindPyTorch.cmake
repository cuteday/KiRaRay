if (DEFINED ENV{TORCH_INSTALL_DIR} OR DEFINED TORCH_INSTALL_DIR)
if (DEFINED ENV{TORCH_INSTALL_DIR})
	set( PYTORCH_ROOT $ENV{TORCH_INSTALL_DIR} CACHE PATH "PyTorch installation directory.")
else()
	set( PYTORCH_ROOT ${TORCH_INSTALL_DIR} CACHE PATH "PyTorch installation directory.")
endif()
message(STATUS "Found PyTorch installation at ${PYTORCH_ROOT}.")
SET(KRR_ENABLE_PYTORCH ON CACHE INTERNAL "Enable pytorch-interop support.")
else()
message(STATUS "Did not find pytorch. If you want to enable pytorch-interop support, specify its installation location via TORCH_INSTALL_DIR.")
SET(KRR_ENABLE_PYTORCH OFF CACHE INTERNAL "Enable pytorch-interop support.")
endif()

if (KRR_ENABLE_PYTORCH)
set(TORCH_BIN_DIR 
	${PYTORCH_ROOT}/lib
)
set(TORCH_LIBS
	${TORCH_BIN_DIR}/c10.lib
	${TORCH_BIN_DIR}/c10_cuda.lib
	${TORCH_BIN_DIR}/torch.lib
	${TORCH_BIN_DIR}/torch_cpu.lib
	${TORCH_BIN_DIR}/torch_cuda.lib
	${TORCH_BIN_DIR}/torch_cuda_cu.lib
	${TORCH_BIN_DIR}/torch_cuda_cpp.lib
	${TORCH_BIN_DIR}/torch_python.lib
)
set(TORCH_INCLUDE_DIRS 
	${PYTORCH_ROOT}/include 
	${PYTORCH_ROOT}/include/torch/csrc/api/include
)
#c10_cuda.lib torch_cpu.lib torch_cuda.lib
add_library(torch_lib INTERFACE)
target_include_directories(torch_lib INTERFACE ${TORCH_INCLUDE_DIRS})
target_link_libraries(torch_lib INTERFACE ${TORCH_LIBS})
endif()