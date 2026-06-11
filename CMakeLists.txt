#
# Copyright 2026 René Widera
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

project(
    alpakaNN
    VERSION 0.1.0
    DESCRIPTION "Header-only neural network building blocks on top of alpaka3."
    HOMEPAGE_URL "https://github.com/alpaka-group/alpakaNN"
    LANGUAGES CXX
)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(alpakaNN_BUILD_TESTS "Build alpakaNN unit tests." ${PROJECT_IS_TOP_LEVEL})
set(alpaka_COMPILE_PEDANTIC ON CACHE BOOL "Compile all code with strict compiler settings." FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(alpaka_TESTS OFF CACHE BOOL "Disable alpaka3 upstream tests." FORCE)
set(alpaka_DOCS OFF CACHE BOOL "Disable alpaka3 docs tests." FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "Disable alpaka3 benchmarks." FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "Disable alpaka3 examples." FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "Disable alpaka3 header checks." FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/alpaka3/CMakeLists.txt")
    message(
        FATAL_ERROR
        "alpaka3 is missing. Initialize submodules first: git submodule update --init --recursive"
    )
endif()

add_subdirectory(extern/alpaka3)

add_library(alpakaNN INTERFACE)
add_library(alpakaNN::alpakaNN ALIAS alpakaNN)

target_compile_features(alpakaNN INTERFACE cxx_std_20)
target_include_directories(
    alpakaNN
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(alpakaNN INTERFACE $<BUILD_INTERFACE:alpaka::alpaka>)

if(alpakaNN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()

add_subdirectory(example/cli)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(
    TARGETS alpakaNN
    EXPORT alpakaNNTargets
)

install(
    EXPORT alpakaNNTargets
    NAMESPACE alpakaNN::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaNN
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/alpakaNNConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/alpakaNNConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/alpakaNNConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaNN
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/alpakaNNConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/alpakaNNConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaNN
)
