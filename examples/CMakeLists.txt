# sam3cpplib examples. e2e_check is the parity/validation harness the gates
# in docs/PLAN.md run; segment_image is the minimal consumer demo; quantize
# converts f32/f16 .ggml containers to q4_0/q4_1/q8_0.
if(SAM3CPP_IMAGE_IO)
    add_executable(sam3cpp_e2e_check e2e_check.cpp)
    target_link_libraries(sam3cpp_e2e_check PRIVATE sam3cpp)

    add_executable(sam3cpp_segment_image segment_image.cpp)
    target_link_libraries(sam3cpp_segment_image PRIVATE sam3cpp)

    # features/ -- the numbered per-feature tour (see examples/README.md)
    set(SAM3CPP_FEATURE_EXAMPLES
        01_load_and_encode
        02_text_prompt
        03_point_prompts
        04_box_prompt
        05_exemplar_boxes
        06_concept_embeddings
        07_encode_once_prompt_many
        08_mask_geometry
        09_tensorrt_config
        10_pcs_fp8
    )
    foreach(ex IN LISTS SAM3CPP_FEATURE_EXAMPLES)
        add_executable(sam3cpp_ex_${ex} features/${ex}.cpp)
        target_link_libraries(sam3cpp_ex_${ex} PRIVATE sam3cpp)
    endforeach()
else()
    message(STATUS "sam3cpplib: SAM3CPP_IMAGE_IO=OFF -- skipping e2e_check/segment_image examples")
endif()

add_executable(sam3cpp_quantize quantize.cpp)
target_link_libraries(sam3cpp_quantize PRIVATE ggml)

# Offline weight-dump tools backing scripts/development/export_onnx.sh (raw .bin blobs +
# manifest.json consumed by the ONNX-authoring scripts in scripts/convert/).
add_executable(sam3cpp_dump_encoder_weights dump_encoder_weights.cpp)
target_link_libraries(sam3cpp_dump_encoder_weights PRIVATE sam3cpp)

add_executable(sam3cpp_dump_pcs_pvs_weights dump_pcs_pvs_weights.cpp)
target_link_libraries(sam3cpp_dump_pcs_pvs_weights PRIVATE sam3cpp)

# FP8-calibration input capture (uses sam3_internal.h -- offline dev tooling
# for scripts/convert/fp8_pcs_amax_calib.py, not consumer API).
if(SAM3CPP_IMAGE_IO)
    add_executable(sam3cpp_dump_pcs_calib_inputs dump_pcs_calib_inputs.cpp)
    target_link_libraries(sam3cpp_dump_pcs_calib_inputs PRIVATE sam3cpp)
    target_include_directories(sam3cpp_dump_pcs_calib_inputs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
endif()
