#
# Copyright 2026 Rene Widera
# SPDX-License-Identifier: ISC
#
# Derive the alpakaVendor backend options from the resolved alpaka backends.
#
# The vendor BLAS backend is MANDATORY for every enabled alpaka backend: GEMM/GEMV are always routed through
# alpaka::blas::onHost (there is no native fallback). This module therefore turns on the matching vendor backend
# for each enabled alpaka backend and fails the configure with a clear message when it cannot be provided:
#
#   Host  -> OpenBLAS (always on)
#   CUDA  -> cuBLAS   (alpakaV_DEP_CUBLAS)
#   HIP   -> rocBLAS  (alpakaV_DEP_ROCBLAS)
#   oneAPI-> oneMKL   (alpakaV_DEP_ONEMKL, MKL::MKL_SYCL::DFT)
#
# The alpaka backend switches (alpaka_DEP_CUDA / alpaka_DEP_HIP / alpaka_DEP_ONEAPI) are declared by alpaka3;
# depending on the configure order they may already be defined or still unset. This module reads them and FORCEs
# the derived alpakaV_* options on every configure so a reconfigured build directory cannot keep a stale backend
# selection. It is factored out so the option logic can be exercised in isolation (cmake -P) without a
# oneAPI/CUDA toolchain.

# The OpenBLAS host backend is no longer optional. Reject the retired alpakaNN_OPENBLAS=OFF switch explicitly
# instead of silently ignoring it.
if(DEFINED alpakaNN_OPENBLAS AND NOT alpakaNN_OPENBLAS)
    message(
        FATAL_ERROR
        "alpakaNN_OPENBLAS=OFF is no longer supported: GEMM/GEMV always go through the vendor BLAS and the \
OpenBLAS host backend is mandatory."
    )
endif()

# Verify the OpenBLAS host dependency up front so a missing library produces an actionable message instead of a
# failure from an unrelated target.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(ALPAKANN_OPENBLAS_PROBE QUIET openblas)
endif()
if(NOT ALPAKANN_OPENBLAS_PROBE_FOUND)
    message(FATAL_ERROR "The OpenBLAS host backend is required for alpakaNN (pkg-config module 'openblas' not found).")
endif()

set(alpakaV_DEP_OPENBLAS ON CACHE BOOL "OpenBLAS host backend (required by alpakaNN)." FORCE)

if(alpaka_DEP_CUDA)
    set(_alpakaNN_vendor_cuda ON)
else()
    set(_alpakaNN_vendor_cuda OFF)
endif()
set(alpakaV_DEP_CUBLAS ${_alpakaNN_vendor_cuda} CACHE BOOL "cuBLAS CUDA backend (required with CUDA)." FORCE)
set(alpakaV_DEP_CUFFT ${_alpakaNN_vendor_cuda} CACHE BOOL "Enable the cuFFT CUDA backend." FORCE)

if(alpaka_DEP_HIP)
    set(_alpakaNN_vendor_hip ON)
else()
    set(_alpakaNN_vendor_hip OFF)
endif()
set(alpakaV_DEP_ROCBLAS ${_alpakaNN_vendor_hip} CACHE BOOL "rocBLAS HIP backend (required with HIP)." FORCE)
set(alpakaV_DEP_ROCFFT ${_alpakaNN_vendor_hip} CACHE BOOL "Enable the rocFFT HIP backend." FORCE)

# oneMKL is mandatory for oneAPI: GEMM/GEMV have no native fallback. The MKL package and its SYCL BLAS/DFT target
# must be discoverable, otherwise the configure fails with an actionable message.
if(alpaka_DEP_ONEAPI)
    find_package(
        MKL
        CONFIG
        QUIET
        HINTS
            "$ENV{MKLROOT}/lib/cmake/mkl"
            "$ENV{ONEAPI_ROOT}/mkl/latest/lib/cmake/mkl"
            "$ENV{ONEAPI_ROOT}/mkl/latest/lib/cmake"
    )
    if(NOT TARGET MKL::MKL_SYCL::DFT)
        message(
            FATAL_ERROR
            "oneAPI requires oneMKL: the MKL package and target MKL::MKL_SYCL::DFT must be discoverable. \
GEMM/GEMV use the vendor BLAS backend and have no native fallback."
        )
    endif()
    message(STATUS "oneMKL BLAS/DFT backend enabled (MKL ${MKL_VERSION}).")
    set(_alpakaNN_vendor_onemkl ON)
else()
    set(_alpakaNN_vendor_onemkl OFF)
endif()
set(alpakaV_DEP_ONEMKL ${_alpakaNN_vendor_onemkl} CACHE BOOL "oneMKL oneAPI backends (required with oneAPI)." FORCE)

unset(_alpakaNN_vendor_cuda)
unset(_alpakaNN_vendor_hip)
unset(_alpakaNN_vendor_onemkl)
