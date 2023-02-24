# dependends on glfw

add_library(imgui
    imconfig.h
    imgui_draw.cpp
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