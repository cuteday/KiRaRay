# Locate the OptiX distribution.  Search relative to the SDK first, then look in the system.

# Our initial guess will be within the SDK.

if (DEFINED ENV{OptiX_INSTALL_DIR})
  set (OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR})
  set (searched_OptiX_INSTALL_DIR ${OptiX_INSTALL_DIR})
elseif(NOT OptiX_INSTALL_DIR)
  if (WIN32)
	find_path(searched_OptiX_INSTALL_DIR
	  NAME include/optix.h
	  PATHS
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.7.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.6.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
	  "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
	)
  endif()
endif()
mark_as_advanced(searched_OptiX_INSTALL_DIR)
set(OptiX_INSTALL_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
set(OptiX_ROOT_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
set(OptiX_INSTALL_DIR $ENV{OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")
set(OptiX_ROOT_DIR ${searched_OptiX_INSTALL_DIR} CACHE PATH "Path to OptiX installed location.")

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