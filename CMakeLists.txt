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
    set(GLADE_FILE "${CMAKE_SOURCE_DIR}/src/display/gtk/interface.glade")
    set(INTERFACE_FILE ${GEN_PATH}/interface.ui)
    add_custom_command(OUTPUT ${INTERFACE_FILE}
        COMMAND gtk4-builder-tool simplify --3to4
            ${GLADE_FILE} > ${INTERFACE_FILE}
        DEPENDS ${GLADE_FILE})
    add_custom_target(convert-gtk4-ui DEPENDS ${INTERFACE_FILE})
    include_directories(${GEN_PATH})
elseif ("${WINDOW_TYPE}" STREQUAL "qt") ### QT
    find_package(QT REQUIRED)
    include_directories(${QT_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${QT_LIBRARIES})
else ()
    message(FATAL_ERROR
        "Window type is undefined [${WINDOW_TYPE}]: curses, sdl, gtk, or qt")
endif ()
add_display_src(${WINDOW_TYPE})

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(src)

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS_LIST})

### gtk additional dependency
if ("${WINDOW_TYPE}" STREQUAL "gtk") ### GTK
    add_dependencies(${PROJECT_NAME} convert-gtk4-ui)
endif()
