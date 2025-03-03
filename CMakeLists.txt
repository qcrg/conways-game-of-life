cmake_minimum_required(VERSION 3.7)
project(conways_game_of_life)

### default
set(WINDOW_TYPE $ENV{WINDOW_TYPE})
string(TOLOWER "${WINDOW_TYPE}" WINDOW_TYPE)
if ("${WINDOW_TYPE}" STREQUAL "")
    set(WINDOW_TYPE "curses")
endif ()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

set(GEN_PATH "${CMAKE_BINARY_DIR}/gen")
file(MAKE_DIRECTORY ${GEN_PATH})
include_directories(${GEN_PATH})

### build type dependencies
if (CMAKE_BUILD_TYPE EQUAL "Release")
    #add_compile_options(-DNDEBUG -O3)
else () # Debug
    #add_compile_options(-O0 -ggdb)
    set(SDL_DEBUG 1)
endif ()

### general source
file(GLOB_RECURSE SRC_LIST src/*.cxx)
list(FILTER SRC_LIST EXCLUDE REGEX ".*display.*")

function (add_display_src dir)
    set(TMP "")
    file(GLOB_RECURSE TMP "src/display/${dir}/*.cxx")
    set(SRC_LIST ${SRC_LIST} ${TMP} PARENT_SCOPE)
    include_directories("src/display/${dir}")
    string(TOUPPER ${dir} DIR)
    add_compile_options(-DPND_USE_${DIR})
endfunction ()

### display type depend source
if ("${WINDOW_TYPE}" STREQUAL "curses") ### curses
    find_package(Curses REQUIRED)
    include_directories(${CURSES_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${CURSES_LIBRARIES})
elseif ("${WINDOW_TYPE}" STREQUAL "sdl") ### SDL
    ### SDL2
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${SDL2_LIBRARIES})

    ### SDL2_ttf
    find_package(SDL2_ttf REQUIRED)
    list(APPEND LIBS_LIST SDL2_ttf::SDL2_ttf)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND
            "${SDL_DEBUG}")
        add_compile_options(-DPND_SDL_DEBUG)
    endif()
elseif ("${WINDOW_TYPE}" STREQUAL "gtk") ### GTK
    ### PkgConfig
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(GTK REQUIRED gtkmm-4.0)
    include_directories(${GTK_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${GTK_LIBRARIES})

    ### convert gtk3 glade file to gtk4 ui file target
    set(INTERFACE_FILE_NAME "interface.ui")
    set(CONVERT_FILE
        "${CMAKE_SOURCE_DIR}/src/display/gtk/interface/interface.cmb")
    set(INTERFACE_FILE
        "${CMAKE_SOURCE_DIR}/src/display/gtk/interface/${INTERFACE_FILE_NAME}")
    file(CREATE_LINK ${INTERFACE_FILE} ${GEN_PATH}/${INTERFACE_FILE_NAME}
        RESULT TMP
        SYMBOLIC)
    if (${TMP})
        message(FATAL_ERROR "${TMP})")
    endif ()
    add_custom_command(OUTPUT ${INTERFACE_FILE}
        COMMAND cambalache -E ${CONVERT_FILE}
        DEPENDS ${CONVERT_FILE})
    add_custom_target(export-cambalache DEPENDS ${INTERFACE_FILE})
else ()
    message(FATAL_ERROR
        "Window type is undefined [${WINDOW_TYPE}]: curses, sdl or gtk")
endif ()
add_display_src(${WINDOW_TYPE})

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(src)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS_LIST})

### gtk additional dependency
if ("${WINDOW_TYPE}" STREQUAL "gtk") ### GTK
    add_dependencies(${PROJECT_NAME} export-cambalache)
endif()
