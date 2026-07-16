# sam3cpplib examples. e2e_check is the parity/validation harness the gates
# in docs/PLAN.md run; segment_image is the minimal consumer demo; quantize
# converts f32/f16 .ggml containers to q4_0/q4_1/q8_0.
if(SAM3CPP_IMAGE_IO)
    add_executable(sam3cpp_e2e_check e2e_check.cpp)
    target_link_libraries(sam3cpp_e2e_check PRIVATE sam3cpp)

    add_executable(sam3cpp_segment_image segment_image.cpp)
    target_link_libraries(sam3cpp_segment_image PRIVATE sam3cpp)
else()
    message(STATUS "sam3cpplib: SAM3CPP_IMAGE_IO=OFF -- skipping e2e_check/segment_image examples")
endif()

add_executable(sam3cpp_quantize quantize.cpp)
target_link_libraries(sam3cpp_quantize PRIVATE ggml)
