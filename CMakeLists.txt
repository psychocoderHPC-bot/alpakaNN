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
include(FetchContent)

option(alpakaNN_BUILD_TESTS "Build alpakaNN unit tests." ${PROJECT_IS_TOP_LEVEL})
option(alpakaNN_OPENBLAS "Enable OpenBLAS host dispatch for GEMM/GEMV." ON)
set(alpaka_COMPILE_PEDANTIC ON CACHE BOOL "Compile all code with strict compiler settings." FORCE)

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

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/alpaka3/CMakeLists.txt")
    message(FATAL_ERROR "alpaka3 is missing. Initialize submodules first: git submodule update --init --recursive")
endif()

add_subdirectory(extern/alpaka3)

if(alpakaNN_OPENBLAS)
    set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "OpenBLAS: build BLAS only" FORCE)
    set(BUILD_WITHOUT_LAPACKE ON CACHE BOOL "OpenBLAS: disable LAPACKE" FORCE)
    set(BUILD_WITHOUT_CBLAS OFF CACHE BOOL "OpenBLAS: keep CBLAS" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "OpenBLAS: disable upstream testing" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "OpenBLAS: disable upstream benchmarks" FORCE)
    set(BUILD_STATIC_LIBS ON CACHE BOOL "OpenBLAS: build static library" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "OpenBLAS: disable shared library" FORCE)
    set(NOFORTRAN ON CACHE BOOL "OpenBLAS: skip Fortran support" FORCE)
    set(DYNAMIC_ARCH OFF CACHE BOOL "OpenBLAS: build only the selected CPU target" FORCE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(TARGET CORE2 CACHE STRING "OpenBLAS: conservative x86_64 target" FORCE)
    endif()

    FetchContent_Declare(openblas URL https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.30.tar.gz)
    FetchContent_MakeAvailable(openblas)
endif()

add_library(alpakaNN INTERFACE)
add_library(alpaka::nn ALIAS alpakaNN)
set_target_properties(alpakaNN PROPERTIES EXPORT_NAME nn)

target_compile_features(alpakaNN INTERFACE cxx_std_20)
target_compile_definitions(alpakaNN INTERFACE ALPAKANN_OPENBLAS_ENABLED=$<BOOL:${alpakaNN_OPENBLAS}>)
target_include_directories(
    alpakaNN
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(alpakaNN INTERFACE $<BUILD_INTERFACE:alpaka::alpaka>)
if(alpakaNN_OPENBLAS)
    target_link_libraries(alpakaNN INTERFACE openblas)
    target_include_directories(
        alpakaNN
        INTERFACE $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/generated> $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    )
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(alpakaNN INTERFACE CUDA::cublas)
endif()

add_subdirectory(example/cli)

if(alpakaNN_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(test)
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
