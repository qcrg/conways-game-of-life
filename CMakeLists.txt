cmake_minimum_required(VERSION 3.7)
project(conways_game_of_life)

### default
if ("${DISPLAY_TYPE}" STREQUAL "")
    set(DISPLAY_TYPE "curses")
endif ()

### general source
file(GLOB_RECURSE SRC_LIST src/*.cxx)
list(FILTER SRC_LIST EXCLUDE REGEX ".*display.*")

### PkgConfig
find_package(PkgConfig REQUIRED)

function (add_display_src dir)
    set(TMP "" PARENT_SCOPE)
    file(GLOB_RECURSE TMP "src/display/${dir}/*.cxx")
    list(APPEND SRC_LIST ${TMP})
    include_directories("src/display/${dir}")
    string(TOUPPER ${dir} DIR)
    add_compile_options(-DPND_USE_${DIR})
endfunction ()

### display type depend source
if ("${DISPLAY_TYPE}" STREQUAL "curses") ### curses
    find_package(Curses REQUIRED)
    include_directories(${CURSES_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${CURSES_LIBRARIES})
elseif ("${DISPLAY_TYPE}" STREQUAL "sdl") ### SDL
    find_package(SDL2 REQUIRED)
    include_directories(${SDL2_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${SDL2_LIBRARIES})
elseif ("${DISPLAY_TYPE}" STREQUAL "gtk") ### GTK
    find_package(GTK REQUIRED)
    include_directories(${GTK_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${GTK_LIBRARIES})
elseif ("${DISPLAY_TYPE}" STREQUAL "qt") ### QT
    find_package(QT REQUIRED)
    include_directories(${QT_INCLUDE_DIRS})
    list(APPEND LIBS_LIST ${QT_LIBRARIES})
else ()
    message(FATAL_ERROR "Display type is undefined")
endif ()
add_display_src(${DISPLAY_TYPE})

### SIGC
pkg_check_modules(SIGC REQUIRED sigc++-3.0)
include_directories(${SIGC_INCLUDE_DIRS})
list(APPEND LIBS_LIST ${SIGC_LIBRARIES})

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic)

include_directories(src)

if (CMAKE_BUILD_TYPE EQUAL "Release")
    add_compile_options(-DNDEBUG -O3)
else () # Debug
    add_compile_options(-O0 -ggdb)
endif ()

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS_LIST})
