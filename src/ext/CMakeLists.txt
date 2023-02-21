###############################################################################
# GAPI dependencies
###############################################################################
INCLUDE (${KRR_PROJECT_ROOT}/common/build/FindOptiX.cmake)
find_package(OpenGL REQUIRED)
# Vulkan support 
find_package(Vulkan REQUIRED)

###############################################################################
# third party libraries
###############################################################################

SET (CMAKE_POLICY_DEFAULT_CMP0077 NEW)
STRING (REGEX REPLACE "/W[3|4]" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# glew
SET(glew-cmake_BUILD_SHARED OFF)
ADD_SUBDIRECTORY(glew EXCLUDE_FROM_ALL)
SET(GLEW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glew/include)
SET(GLEW_INCLUDE_DIRS ${GLEW_INCLUDE_DIRS} PARENT_SCOPE)

# vulkan related
# vulkan memory allocator
ADD_LIBRARY(vma INTERFACE)
TARGET_INCLUDE_DIRECTORIES(vma INTERFACE vma)

ADD_SUBDIRECTORY(nvrhi)
SET(NVRHI_INCLUDE_DIRS 
	${CMAKE_CURRENT_SOURCE_DIR}/nvrhi/include
	${CMAKE_CURRENT_SOURCE_DIR}/nvrhi/thirdparty/Vulkan-Headers/include
)
SET(NVRHI_INCLUDE_DIRS ${NVRHI_INCLUDE_DIRS} PARENT_SCOPE)

# glfw3 (lib: glfw)
ADD_SUBDIRECTORY(glfw EXCLUDE_FROM_ALL)
INCLUDE_DIRECTORIES(glfw/include)
SET(GLFW_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/glfw/include)
SET(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIRS} PARENT_SCOPE)

# imgui	(lib: imgui)
ADD_SUBDIRECTORY(imgui)
SET(IMGUI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/imgui)
SET(IMGUI_INCLUDE_DIRS ${IMGUI_INCLUDE_DIRS} PARENT_SCOPE)

# Assimp
SET(BUILD_SHARED_LIBS OFF)
SET(ASSIMP_BUILD_TESTS OFF)
SET(ASSIMP_NO_EXPORT ON)
SET(ASSIMP_BUILD_ASSIMP_TOOLS OFF)
# specify enabled importers for speed
SET(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF)
SET(ASSIMP_BUILD_OBJ_IMPORTER ON)
SET(ASSIMP_BUILD_FBX_IMPORTER ON)
SET(ASSIMP_BUILD_GLTF_IMPORTER ON)
ADD_SUBDIRECTORY(assimp EXCLUDE_FROM_ALL)
INCLUDE_DIRECTORIES(assimp/include)
SET(ASSIMP_INCLUDE_DIRS 
	${CMAKE_CURRENT_SOURCE_DIR}/assimp/include 
	${CMAKE_CURRENT_BINARY_DIR}/assimp/include)
SET(ASSIMP_INCLUDE_DIRS ${ASSIMP_INCLUDE_DIRS} PARENT_SCOPE)

# for tinyexr, currently used the zlib contained in assimp, according to https://github.com/syoyo/tinyexr
# could switch to miniz if anything happens.
SET(IMAGEIO_INCLUDE_DIRS 
	${CMAKE_CURRENT_SOURCE_DIR}/image 
	${CMAKE_CURRENT_SOURCE_DIR}/assimp/contrib/zlib
	${CMAKE_CURRENT_BINARY_DIR}/assimp/contrib/zlib)
SET(IMAGEIO_INCLUDE_DIRS ${IMAGEIO_INCLUDE_DIRS} PARENT_SCOPE)

# json

SET(JSON_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/json)
SET(JSON_INCLUDE_DIRS ${JSON_INCLUDE_DIRS} PARENT_SCOPE)

# pybind11
IF(KRR_ENABLE_PYTHON)
	ADD_SUBDIRECTORY(pybind11)
ENDIF()

# pbrt parser
SET(pbrtParser_ENABLE_UNITTESTS OFF)
ADD_SUBDIRECTORY (pbrtparser EXCLUDE_FROM_ALL)
SET( PBRTPARSER_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/pbrtparser/pbrtParser/include)
SET( PBRTPARSER_INCLUDE_DIRS ${PBRTPARSER_INCLUDE_DIRS} PARENT_SCOPE)
#INCLUDE_DIRECTORIES(PBRTPARSER_INCLUDE_DIRS)

ADD_LIBRARY(krr_ext INTERFACE)

TARGET_LINK_LIBRARIES(krr_ext INTERFACE
	glfw
	imgui
	assimp
	libglew_static
	pbrtParser
	zlibstatic
	OpenGL::GL
	${OPENGL_gl_LIBRARY}
	Vulkan::Vulkan
	nvrhi_vk
	Vulkan-Headers
)

# shaderc for runtime compilation of glsl(shaderc) and hlsl(dxc) shaders
add_library(shaderc UNKNOWN IMPORTED)
if(WIN32)
	# use the version in the SDK
	set_target_properties(shaderc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/shaderc_shared.lib)
	set_property(TARGET shaderc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
	target_link_libraries(krr_ext INTERFACE shaderc)
else()
	target_link_directories(krr_ext INTERFACE $ENV{VULKAN_SDK}/lib)
	target_link_libraries(krr_ext INTERFACE shaderc_combined glslang MachineIndependent OSDependent OGLCompiler GenericCodeGen SPIRV SPIRV-Tools-opt SPIRV-Tools)
endif()
add_library(dxc UNKNOWN IMPORTED)
if (WIN32)
	set_target_properties(dxc PROPERTIES IMPORTED_LOCATION $ENV{VULKAN_SDK}/Lib/dxcompiler.lib)
	set_property(TARGET dxc PROPERTY INTERFACE_INCLUDE_DIRECTORIES $ENV{VULKAN_SDK}/Include)
	target_link_libraries(krr_ext INTERFACE dxc)
else()
	target_link_libraries(krr_ext INTERFACE dxcompiler)
endif()

SET(KRR_EXT_INCLUDES
	${ASSIMP_INCLUDE_DIRS}
	${GLEW_INCLUDE_DIRS}
	${GLFW_INCLUDE_DIRS}
	${IMGUI_INCLUDE_DIRS}
	${JSON_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
	${IMAGEIO_INCLUDE_DIRS}
	${PBRTPARSER_INCLUDE_DIRS}
	${NVRHI_INCLUDE_DIRS}
)
SET(KRR_EXT_INCLUDES ${KRR_EXT_INCLUDES} PARENT_SCOPE)

TARGET_INCLUDE_DIRECTORIES(krr_ext INTERFACE
	KRR_EXT_INCLUDES
)