set(_krr_openvdb_root "${CMAKE_CURRENT_SOURCE_DIR}/openvdb")
set(OPENVDB_INCLUDE_DIRS
	"${_krr_openvdb_root}/nanovdb/include"
	"${_krr_openvdb_root}/openvdb/include"
	"${_krr_openvdb_root}/boost/include"
	"${_krr_openvdb_root}/tbb/include"
	"${_krr_openvdb_root}/openexr/include")
set(OPENVDB_INCLUDE_DIRS ${OPENVDB_INCLUDE_DIRS} PARENT_SCOPE)

function(krr_add_legacy_half)
	set(CMAKE_INCLUDE_CURRENT_DIR ON)
	set(CMAKE_DEBUG_POSTFIX _d)
	set(OPENEXR_BUILD_SHARED ON)
	set(OPENEXR_VERSION 2.4.0)
	set(OPENEXR_SOVERSION 24)
	# Only Half is required by the prebuilt OpenVDB library.
	add_subdirectory("${_krr_openvdb_root}/ilmbase/Half"
		"${CMAKE_CURRENT_BINARY_DIR}/openvdb/ilmbase/Half")
endfunction()
krr_add_legacy_half()

add_library(openvdb INTERFACE)
target_include_directories(openvdb INTERFACE ${OPENVDB_INCLUDE_DIRS})
target_link_directories(openvdb INTERFACE
	"${_krr_openvdb_root}/openvdb/lib"
	"${_krr_openvdb_root}/boost/lib"
	"${_krr_openvdb_root}/tbb/lib/$<$<CONFIG:Debug>:debug>")
target_link_libraries(openvdb INTERFACE
	"$<IF:$<CONFIG:Debug>,openvdb_d,openvdb>"
	"$<IF:$<CONFIG:Debug>,tbb_debug,tbb>"
	libboost_system-vc141-mt-x64-1_70
	libboost_iostreams-vc141-mt-x64-1_70
	IlmBase::Half)
