#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

alpaka_install_catch2()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
option(alpakaNN_ENABLE_TINYLLAMA_LARGE_TESTS "Enable TinyLlama download/convert/chat smoke tests." OFF)

function(alpakaNN_add_test_target target_prefix folder)
    file(GLOB test_folders ${folder})
    foreach(test_folder ${test_folders})
        if(NOT IS_DIRECTORY ${test_folder})
            continue()
        endif()

        file(GLOB test_sources "${test_folder}/**")
        foreach(source_file ${test_sources})
            get_filename_component(source_file_extension ${source_file} EXT)
            if(NOT "${source_file_extension}" STREQUAL ".cpp")
                continue()
            endif()

            get_filename_component(source_file_name ${source_file} NAME_WE)
            get_filename_component(test_folder_name ${source_file} DIRECTORY)
            get_filename_component(test_folder_name ${test_folder_name} NAME)

            set(test_target_name "${target_prefix}${test_folder_name}_${source_file_name}")
            add_executable(${test_target_name} ${source_file})
            target_link_libraries(${test_target_name} PRIVATE Catch2::Catch2WithMain alpaka::nn alpaka::alpaka)
            alpaka_finalize(${test_target_name})
            catch_discover_tests(${test_target_name})
        endforeach()
    endforeach()
endfunction()

alpakann_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit/**)

# Download tiny LLaMA model for testing during build
set(TINY_LLAMA_DIR ${CMAKE_BINARY_DIR}/testdata/tiny_llama)
file(MAKE_DIRECTORY ${TINY_LLAMA_DIR})

message(STATUS "Downloading tiny LLaMA model to: ${TINY_LLAMA_DIR}")
execute_process(
    COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/download_tiny_llama.py tiny_llama ${TINY_LLAMA_DIR}
    RESULT_VARIABLE DOWNLOAD_RESULT
)

if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download tiny LLaMA model")
endif()

# Define macro for test to find model
add_compile_definitions(TINY_LLAMA_MODEL_PATH="${TINY_LLAMA_DIR}/tiny_llama.bin")

add_test(
    NAME tokenizer_helper_roundtrip
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/integration/tokenizer_helper_roundtrip.py
        ${CMAKE_SOURCE_DIR}/tools/llama_tokenizer.py ${TINY_LLAMA_DIR}
)

add_test(
    NAME chat_cli_selftest
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/integration/chat_cli_smoke.py selftest $<TARGET_FILE:ChatCli>
        ${TINY_LLAMA_DIR}
)

add_test(
    NAME chat_cli_interactive
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/integration/chat_cli_smoke.py interactive $<TARGET_FILE:ChatCli>
        ${TINY_LLAMA_DIR}
)

if(alpakaNN_ENABLE_TINYLLAMA_LARGE_TESTS)
    add_test(
        NAME chat_cli_tinyllama_chat
        COMMAND
            ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/integration/chat_cli_smoke.py tinyllama-chat
            $<TARGET_FILE:ChatCli> ${TINY_LLAMA_DIR} ${CMAKE_SOURCE_DIR}
    )
endif()
