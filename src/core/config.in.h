#pragma once

#define KRR_PROJECT_NAME "${CMAKE_PROJECT_NAME}"
#define KRR_PROJECT_DIR "${CMAKE_SOURCE_DIR}"
#define KRR_BUILD_TYPE "${CMAKE_BUILD_TYPE}"

#cmakedefine01  KRR_PLATFORM_WINDOWS 
#cmakedefine01  KRR_PLATFORM_LINUX 
#cmakedefine01	KRR_PLATFORM_UNKNOWN