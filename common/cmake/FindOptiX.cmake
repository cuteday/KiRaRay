# Locate the OptiX distribution.  Search relative to the SDK first, then look in the system.

# Our initial guess will be within the SDK.

if (DEFINED ENV{OptiX_INSTALL_DIR})
  set (OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
elseif(NOT OptiX_INSTALL_DIR)
  if (WIN32)
	find_path(searched_OptiX_INSTALL_DIR
	  NAME include/optix.h
	  PATHS
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
	)
  else()
	find_path(searched_OptiX_INSTALL_DIR
	NAME include/optix.h
	PATHS
	"/usr/local/NVIDIA-OptiX-SDK-7.4.0-linux64-x86_64"
	"/usr/local/NVIDIA-OptiX-SDK-7.3.0-linux64-x86_64"
	"/usr/local/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64"
	"/usr/local/NVIDIA-OptiX-SDK-7.1.0-linux64-x86_64"
	"/usr/local/NVIDIA-OptiX-SDK-7.0.0-linux64-x86_64"
	"/usr/local/NVIDIA-OptiX-SDK-*"
	)
  endif()
endif()
mark_as_advanced(searched_OptiX_INSTALL_DIR)
set(OptiX_INSTALL_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
set(OptiX_ROOT_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
set(OptiX_ROOT_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")

# The distribution contains both 32 and 64 bit libraries.  Adjust the library
# search path based on the bit-ness of the build.  (i.e. 64: bin64, lib64; 32:
# bin, lib).  Note that on Mac, the OptiX library is a universal binary, so we
# only need to look in lib and not lib64 for 64 bit builds.
if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND NOT APPLE)
  set(bit_dest "64")
else()
  set(bit_dest "")
endif()

# Include
find_path(OptiX_INCLUDE
  NAMES optix.h
  PATHS "${OptiX_INSTALL_DIR}/include"
  NO_DEFAULT_PATH
  )
find_path(OptiX_INCLUDE
  NAMES optix.h
  )

# Check to make sure we found what we were looking for
function(OptiX_report_error error_message required)
  if(OptiX_FIND_REQUIRED AND required)
	message(FATAL_ERROR "${error_message}")
  else()
	if(NOT OptiX_FIND_QUIETLY)
	  message(STATUS "${error_message}")
	endif(NOT OptiX_FIND_QUIETLY)
  endif()
endfunction()

if(NOT OptiX_INCLUDE)
  OptiX_report_error("OptiX headers (optix.h and friends) not found.  Please locate before proceeding." TRUE)
endif()