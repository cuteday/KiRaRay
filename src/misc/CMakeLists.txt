# Then there are some misc files

# If a compile target is dependent on KiRaRay, link the following libraries:
# krr_lib

# the optional KRR_BUILD_STARLIGHT has been moved to the main CMakeLists.txt of the sources folder

if (KRR_BUILD_EXAMPLES)
	ADD_SUBDIRECTORY (samples)	# additional use cases of example (render passes)
endif()
