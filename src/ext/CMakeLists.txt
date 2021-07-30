# Assimp

set(BUILD_SHARED_LIBS OFF)
set(ASSIMP_BUILD_TESTS OFF)
add_subdirectory(assimp)

INCLUDE_DIRECTORIES(assimp/include)