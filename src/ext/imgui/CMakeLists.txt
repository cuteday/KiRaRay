# dependends on glfw and vulkan

add_library(imgui
    imconfig.h
    imgui_draw.cpp
    #imgui_impl_vulkan.h
    #imgui_impl_vulkan.cpp
	imgui_impl_opengl3.cpp
	imgui_impl_opengl3.h
	imgui_impl_opengl3_loader.h
    imgui_impl_glfw.cpp
    imgui_impl_glfw.h
    imgui_internal.h
    imgui_tables.cpp
    imgui_widgets.cpp
    imgui.cpp
    imgui.h
    imstb_rectpack.h
    imstb_textedit.h
    imstb_truetype.h
)

target_include_directories(imgui PUBLIC ../glfw)
target_link_libraries(imgui PRIVATE glfw)
#target_link_libraries(imgui PRIVATE Vulkan::Vulkan)