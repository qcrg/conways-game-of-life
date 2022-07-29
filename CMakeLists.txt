cmake_minimum_required(VERSION 3.7)
project(conways_game_of_life)

file(GLOB SRC_LIST src/*.cc src/*/*.cc)

set(CXX_STANDARD 20)
add_compile_options(-Wall -Wextra -Wpedantic)

#SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
list(APPEND LIBS_LIST ${SDL2_LIBRARIES})
#/SDL2

include_directories(src)

if (CMAKE_BUILD_TYPE EQUAL "Release")
    add_compile_options(-DNDEBUG -O3)
else() # Debug
    add_compile_options(-O0 -ggdb)
endif()

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} ${LIBS_LIST})
