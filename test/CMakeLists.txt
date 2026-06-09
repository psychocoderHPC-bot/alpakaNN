#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

alpaka_install_catch2()

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
            target_link_libraries(${test_target_name} PRIVATE Catch2::Catch2WithMain alpakaNN::alpakaNN alpaka::alpaka)
            alpaka_finalize(${test_target_name})
            catch_discover_tests(${test_target_name})
        endforeach()
    endforeach()
endfunction()

alpakaNN_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit/**)

# Download tiny LLaMA model for testing during build
set(TINY_LLAMA_DIR ${CMAKE_BINARY_DIR}/testdata/tiny_llama)
file(MAKE_DIRECTORY ${TINY_LLAMA_DIR})

message(STATUS "Downloading tiny LLaMA model to: ${TINY_LLAMA_DIR}")
execute_process(
    COMMAND python3 ${CMAKE_SOURCE_DIR}/tools/download_tiny_llama.py ${TINY_LLAMA_DIR}
    RESULT_VARIABLE DOWNLOAD_RESULT
)

if(NOT DOWNLOAD_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to download tiny LLaMA model")
endif()

# Define macro for test to find model
add_compile_definitions(TINY_LLAMA_MODEL_PATH="${TINY_LLAMA_DIR}/tiny_llama.bin")
