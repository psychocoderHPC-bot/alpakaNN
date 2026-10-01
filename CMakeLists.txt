#
# SPDX-FileCopyrightText: René Widera
# SPDX-License-Identifier: MPL-2.0
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
include(FetchContent)

option(alpakaNN_BUILD_TESTS "Build alpakaNN unit tests." ${PROJECT_IS_TOP_LEVEL})
option(alpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE "Build the conservative heat-closure reference example." OFF)
set(alpaka_COMPILE_PEDANTIC ON CACHE BOOL "Compile all code with strict compiler settings." FORCE)

if(alpaka_DEP_HIP)
    enable_language(HIP)
endif()
if(alpaka_DEP_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    if(NOT CUDAToolkit_FOUND)
        message(STATUS "Optional CUDAToolkit not found, but required by alpaka.")
    else()
        message(STATUS "Found optional CUDAToolkit")
    endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(alpaka_TESTS OFF CACHE BOOL "Disable alpaka3 upstream tests." FORCE)
set(alpaka_DOCS OFF CACHE BOOL "Disable alpaka3 docs tests." FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "Disable alpaka3 benchmarks." FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "Disable alpaka3 examples." FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "Disable alpaka3 header checks." FORCE)

# The vendor bundles alpaka3 (via its own extern submodule or FetchContent) and exposes the
# alpaka::blas / alpaka::fft abstractions used by alpakaNN's host GEMM/GEMV. The matching vendor BLAS
# backend is mandatory for every enabled alpaka backend (enforced in alpakaNNVendorBackends.cmake).
#   alpakaVendor dev = 03a92cc77e2b0b74e658851337cc6256162edcff
#   bundled alpaka3   = e91f6f4d8d1da321a6423a8ad5b4954562c57a9a
set(alpakaV_TESTS OFF CACHE BOOL "Disable alpakaVendor tests." FORCE)
set(alpakaV_DOCS OFF CACHE BOOL "Disable alpakaVendor docs." FORCE)
set(alpakaV_DEP_FFTW ON CACHE BOOL "Enable the FFTW host backend." FORCE)
# Resolve the alpakaVendor backend options from the enabled alpaka backends.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/alpakaNNVendorBackends.cmake)

# Pinned to alpakaVendor dev; its extern/alpaka3 submodule is pinned at
# e91f6f4d8d1da321a6423a8ad5b4954562c57a9a (the single alpaka3 used by alpakaNN).
FetchContent_Declare(
    alpakaVendor
    GIT_REPOSITORY https://github.com/psychocoderHPC/alpakaVendor.git
    GIT_TAG 03a92cc77e2b0b74e658851337cc6256162edcff
)
FetchContent_MakeAvailable(alpakaVendor)

add_library(alpakaNN INTERFACE)
add_library(alpaka::nn ALIAS alpakaNN)
set_target_properties(alpakaNN PROPERTIES EXPORT_NAME nn)

target_compile_features(alpakaNN INTERFACE cxx_std_20)
target_include_directories(
    alpakaNN
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    alpakaNN
    INTERFACE $<BUILD_INTERFACE:alpaka::alpaka> $<BUILD_INTERFACE:alpakaVendor::alpakaVendor>
)

if(alpakaNN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
endif()

if(alpakaNN_BUILD_HEAT_CLOSURE_EXAMPLE)
    include(CTest)
    add_subdirectory(example/heatEquationNn)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS alpakaNN EXPORT alpakaNNTargets)

install(EXPORT alpakaNNTargets NAMESPACE alpaka:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaNN)

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
    FILES "${CMAKE_CURRENT_BINARY_DIR}/alpakaNNConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/alpakaNNConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/alpakaNN
)
