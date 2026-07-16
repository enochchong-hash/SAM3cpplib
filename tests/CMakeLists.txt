# sam3cpplib tests. parity_test.sh / consume_test are driven by scripts (see
# docs/PLAN.md P4); this only registers the compiled unit tests.
add_executable(sam3cpp_mask_utils_test mask_utils_test.cpp)
target_link_libraries(sam3cpp_mask_utils_test PRIVATE sam3cpp)

enable_testing()
add_test(NAME mask_utils COMMAND sam3cpp_mask_utils_test)
