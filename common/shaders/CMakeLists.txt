# complie shaders to spirv
find_program(GLSL_VALIDATOR glslangValidator HINTS /usr/bin /usr/local/bin $ENV{VULKAN_SDK}/Bin/ $ENV{VULKAN_SDK}/Bin32/)

file(GLOB_RECURSE GLSL_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.comp"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
	message(STATUS "BUILDING SHADER")
	get_filename_component(FILE_NAME ${GLSL} NAME)
	set(SPIRV "${CMAKE_CURRENT_SOURCE_DIR}/${FILE_NAME}.spv")
	message(STATUS ${GLSL})
	add_custom_command(
		OUTPUT ${SPIRV}
		COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
		DEPENDS ${GLSL})
	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
    compile_shaders 
    DEPENDS ${SPIRV_BINARY_FILES}
)