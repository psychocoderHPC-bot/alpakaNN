#
# SPDX-FileCopyrightText: René Widera
# SPDX-License-Identifier: MPL-2.0
#
# Emulates a PIConGPU-like consumer: it adds one alpaka, then alpakaVendor with
# alpakaV_ALPAKA_PROVIDER=extern, then alpakaNN. Neither the vendor nor alpakaNN may add a
# second alpaka source tree.
#
#   cmake -S test/integration/picongpu_like -B build/picongpu_like \
#         -DALPAKA_SOURCE_DIR=<alpaka> -DALPAKAVENDOR_SOURCE_DIR=<alpakaVendor> \
#         -DALPAKANN_SOURCE_DIR=<alpakaNN>

cmake_minimum_required(VERSION 3.25)

project(alpakaNNPicongpuLike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ALPAKA_SOURCE_DIR)
    message(FATAL_ERROR "Set ALPAKA_SOURCE_DIR to the alpaka checkout.")
endif()
if(NOT ALPAKAVENDOR_SOURCE_DIR)
    message(FATAL_ERROR "Set ALPAKAVENDOR_SOURCE_DIR to the alpakaVendor checkout.")
endif()
if(NOT ALPAKANN_SOURCE_DIR)
    set(ALPAKANN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
endif()

set(alpaka_TESTS OFF CACHE BOOL "" FORCE)
set(alpaka_DOCS OFF CACHE BOOL "" FORCE)
set(alpaka_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(alpaka_EXAMPLES OFF CACHE BOOL "" FORCE)
set(alpaka_HEADERCHECKS OFF CACHE BOOL "" FORCE)
if(DEFINED alpaka_DEP_CUDA AND alpaka_DEP_CUDA)
    enable_language(CUDA)
endif()

# Exactly one alpaka: the consumer's own checkout, added before everything else.
add_subdirectory("${ALPAKA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/alpaka-consumer")

set(alpakaV_TESTS OFF CACHE BOOL "" FORCE)
set(alpakaV_DOCS OFF CACHE BOOL "" FORCE)
set(alpakaV_DEP_FFTW ON CACHE BOOL "" FORCE)
set(alpakaV_DEP_OPENBLAS ON CACHE BOOL "" FORCE)
if(DEFINED alpaka_DEP_CUDA AND alpaka_DEP_CUDA)
    set(alpakaV_DEP_CUBLAS ON CACHE BOOL "" FORCE)
    set(alpakaV_DEP_CUFFT ON CACHE BOOL "" FORCE)
endif()
set(alpakaV_ALPAKA_PROVIDER extern CACHE STRING "" FORCE)
add_subdirectory("${ALPAKAVENDOR_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/alpakaVendor-consumer")

set(alpakaNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${ALPAKANN_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/alpakaNN-consumer")

add_executable(picongpu_like main.cpp)
target_link_libraries(picongpu_like PRIVATE alpaka::nn alpaka::alpaka alpakaVendor::alpakaVendor)
alpaka_finalize(picongpu_like)
