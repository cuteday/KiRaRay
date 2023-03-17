# Then there are some misc files

# If a compile target is dependent on KiRaRay, link the following libraries:
# krr_lib

if (KRR_BUILD_STARLIGHT)
	set(KRR_BUILD_STARLIGHT 1)
	ADD_SUBDIRECTORY (render)	# additional implementations of algorithms
	TARGET_LINK_LIBRARIES (krr_lib PUBLIC krr_starlight_lib -WHOLEARCHIVE:$<TARGET_FILE:krr_starlight_lib>)
	SET (KRR_INCLUDE_DIRS ${KRR_INCLUDE_ALL} ${KRR_MISC_INCLUDE_ALL} PARENT_SCOPE) # export include dirs 
endif()

if (KRR_BUILD_EXAMPLES)
	ADD_SUBDIRECTORY (samples)	# additional use cases of example (render passes)
endif()
