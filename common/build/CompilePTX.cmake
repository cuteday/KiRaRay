get_filename_component (cuda_compiler_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
find_program (BIN2C
			  NAMES bin2c
			  PATHS ${cuda_compiler_bin}
			  DOC "Path to the CUDA SDK bin2c executable."
			  NO_DEFAULT_PATH)
if (NOT BIN2C)
	message (FATAL_ERROR
			 "bin2c not found:\n"
			 "  CMAKE_CUDA_COMPILER='${CMAKE_CUDA_COMPILER}'\n"
			 "  cuda_compiler_bin='${cuda_compiler_bin}'\n"
	)
endif ()

# this macro defines cmake rules that execute the following four steps:
# 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
# 2) use the 'bin2c' tool (that comes with CUDA) to
#    create a second intermediary (.c-)file which defines a const string variable
#    (named '${c_var_name}') whose (constant) value is the PTX output
#    from the previous step.
# 3) compile the given .c file to an intermediary object file (why thus has
#    that PTX string 'embedded' as a global constant.
# 4) assign the name of the intermediary .o file to the cmake variable
#    'output_var', which can then be added to cmake targets.
macro (CUDA_COMPILE_EMBED output_var cuda_file lib_name dependencies)
	
	add_library ("${lib_name}" OBJECT "${cuda_file}")
	set_property (TARGET "${lib_name}" PROPERTY CUDA_PTX_COMPILATION ON)
	target_compile_options ("${lib_name}" PRIVATE
							-Xcudafe=--display_error_number 
							-Xcudafe=--diag_suppress=3089
							-Xcudafe=--diag_suppress=1290		# disable "passing arguments... " warning
							-Xcudafe=--diag_suppress=20044)		# disable "extern declaration... is treated as a static definition" warning
	target_compile_options("${lib_name}" PRIVATE ${CUDA_NVCC_FLAGS})
	# CUDA integration in Visual Studio seems broken as even if "Use
	# Host Preprocessor Definitions" is checked, the host preprocessor
	# definitions are still not used when compiling device code.
	# To work around that, define the macros using --define-macro to
	# avoid CMake identifying those as macros and using the proper (but
	# broken) way of specifying them.
	if (${CMAKE_GENERATOR} MATCHES "^Visual Studio")
		if (CMAKE_BUILD_TYPE MATCHES Debug)
			set (CUDA_DEFINITIONS "--define-macro=KRR_DEBUG_BUILD")
		endif ()
		foreach (arg ${KRR_DEFINITIONS})
			list (APPEND CUDA_DEFINITIONS "--define-macro=${arg}")
		endforeach ()
		target_compile_options ("${lib_name}" PRIVATE ${CUDA_DEFINITIONS})
	else ()
	target_compile_definitions ("${lib_name}" PRIVATE ${KRR_DEFINITIONS})
	endif ()
	target_include_directories ("${lib_name}" PRIVATE 
		${KRR_INCLUDE_ALL}
		${CUDA_INCLUDE_DIRS}
		${CMAKE_BINARY_DIR}
		${ARGN}
	)
	target_link_libraries("${lib_name}" PRIVATE krr_cuda_cfg krr_cuda_warning krr_opt)
	add_dependencies ("${lib_name}" "${dependencies}")
	
	set (c_var_name ${output_var})
	set (embedded_file ${cuda_file}.ptx_embedded.c)
	add_custom_command (
	OUTPUT "${embedded_file}"
	COMMAND ${CMAKE_COMMAND}
		"-DBIN_TO_C_COMMAND=${BIN2C}"
		"-DOBJECTS=$<TARGET_OBJECTS:${lib_name}>"
		"-DVAR_NAME=${c_var_name}"
		"-DOUTPUT=${embedded_file}"
		-P ${KRR_PROJECT_ROOT}/common/build/bin2c_wrapper.cmake
	VERBATIM
	DEPENDS "${lib_name}" $<TARGET_OBJECTS:${lib_name}>
	COMMENT "Embedding PTX generated from ${cuda_file}"
	)
	set (${output_var} ${embedded_file})
endmacro ()