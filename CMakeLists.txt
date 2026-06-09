#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 20)

add_library(alpakaNN INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

include(FetchContent)
FetchContent_Declare(alpaka3 GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git GIT_TAG dev)
FetchContent_MakeAvailable(alpaka3)


