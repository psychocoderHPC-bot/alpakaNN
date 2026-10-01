#
# SPDX-FileCopyrightText: René Widera
# SPDX-License-Identifier: MPL-2.0
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
            target_link_libraries(${test_target_name} PRIVATE Catch2::Catch2WithMain alpaka::nn alpaka::alpaka)
            target_include_directories(${test_target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/unit/testHelper")
            alpaka_finalize(${test_target_name})
            if(alpaka_DEP_ONEAPI)
                # Enable the documented IGC fp64 emulation for the discovered tests. This
                # makes fp64-incapable Intel GPUs (e.g. Arc) usable instead of excluding
                # the whole oneAPI device spec. The variable is Intel-specific and ignored
                # by other backends, so applying it here is harmless elsewhere.
                catch_discover_tests(
                    ${test_target_name}
                    PROPERTIES ENVIRONMENT "IGC_EnableDPEmulation=1;OverrideDefaultFP64Settings=1"
                )
            else()
                catch_discover_tests(${test_target_name})
            endif()
        endforeach()
    endforeach()
endfunction()

alpakann_add_test_target(unit_test_ ${CMAKE_CURRENT_SOURCE_DIR}/unit/**)
