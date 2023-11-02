################################################################################
# Optix
################################################################################
if(NOT DEFINED ENV{OptiX_INSTALL_DIR} AND NOT DEFINED OptiX_INSTALL_DIR)
	if(NOT DEFINED ENV{OPTIX_ROOT})
		message("Try to find OptiX SDK in PROGRAMDATA directory.")
		if(NOT DEFINED ENV{PROGRAMDATA})
			message(FATAL_ERROR "PROGRAMDATA is not defined. OPTIX_ROOT has to be specified manually.")
		else()
			# Transform the path to generic style
			file(TO_CMAKE_PATH "$ENV{PROGRAMDATA}" PROGRAMDATA)
			# Find the directory with prefix of "OptiX SDK" in PROGRAMDATA directory
			file(GLOB OPTIX_ROOT "${PROGRAMDATA}/NVIDIA Corporation/OptiX SDK*")
			# If there are multiple directories, use the first one
			list(GET OPTIX_ROOT 0 OPTIX_ROOT)
			# If there is no directory, popup an error message
			if(NOT OPTIX_ROOT)
				message(FATAL_ERROR "OPTIX_ROOT has to be specified manually.")
			else ()
				message(STATUS "Found OptiX SDK at ${OPTIX_ROOT}")
			endif()
		endif()
	else()
		set( OPTIX_ROOT $ENV{OPTIX_ROOT} )
	endif()
else()
	if(DEFINED ENV{OptiX_INSTALL_DIR})
		set( OPTIX_ROOT $ENV{OptiX_INSTALL_DIR} )
	else()
		set( OPTIX_ROOT ${OptiX_INSTALL_DIR} )
	endif()
	message(STATUS "Using specified OptiX path at ${OPTIX_ROOT}")
endif()

message(STATUS "Found OptiX SDK at ${OPTIX_ROOT}")
# Guess the OptiX install directory by OPTIX_ROOT
set(OptiX_INSTALL_DIR ${OPTIX_ROOT} CACHE PATH "Path to OptiX installation location")
# Include the OptiX header directory
set(OptiX_INCLUDE_DIR ${OPTIX_ROOT}/include CACHE PATH "Path to OptiX include directory")