# Standalone consumer project: proves sam3cpplib works as a git submodule of
# a larger app via add_subdirectory(). Built by tests/consume_test/run.sh out
# of tree; never part of the library's own build.
cmake_minimum_required(VERSION 3.18)
project(sam3cpp_consume_test LANGUAGES C CXX)

# Consumers typically don't want the library's examples/tests.
set(SAM3CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SAM3CPP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

# ../../ = the sam3cpplib root, exactly how a submodule consumer would point
# at its checkout (path stays relative; the folder is relocatable).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. sam3cpplib)

add_executable(consume_main main.cpp)
target_link_libraries(consume_main PRIVATE sam3cpp::sam3cpp)
