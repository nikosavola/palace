# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Build libCEED
#

# Force build order
set(LIBCEED_DEPENDENCIES)
if(PALACE_WITH_LIBXSMM)
  list(APPEND LIBCEED_DEPENDENCIES libxsmm)
endif()
if(PALACE_WITH_MAGMA)
  list(APPEND LIBCEED_DEPENDENCIES magma)
endif()
if(PALACE_WITH_OCCA AND PALACE_BUILD_EXTERNAL_DEPS)
  list(APPEND LIBCEED_DEPENDENCIES occa)
endif()

# Note on recommended flags for libCEED (from Makefile, Spack):
#   OPT: -O3 -g -march=native -ffp-contract=fast [-fopenmp-simd/-qopenmp-simd]
include(CheckCCompilerFlag)
set(LIBCEED_OPT_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${BUILD_TYPE_UPPER}}")
if(CMAKE_C_COMPILER_ID MATCHES "Intel|IntelLLVM")
  set(OMP_SIMD_FLAG -qopenmp-simd)
else()
  set(OMP_SIMD_FLAG -fopenmp-simd)
endif()
check_c_compiler_flag(${OMP_SIMD_FLAG} SUPPORTS_OMP_SIMD)
if(SUPPORTS_OMP_SIMD)
  set(LIBCEED_OPT_FLAGS "${LIBCEED_OPT_FLAGS} ${OMP_SIMD_FLAG}")
endif()

# Silence some CUDA/HIP include file warnings
if(PALACE_WITH_CUDA)
  set(LIBCEED_OPT_FLAGS "${LIBCEED_OPT_FLAGS} -isystem ${CUDAToolkit_INCLUDE_DIRS}")
endif()
if(PALACE_WITH_HIP)
  set(LIBCEED_OPT_FLAGS "${LIBCEED_OPT_FLAGS} -isystem ${ROCM_DIR}/include")
endif()

# Configure -pedantic flag if specified (don't want to enable for GPU code)
if(LIBCEED_OPT_FLAGS MATCHES "-pedantic")
  string(REGEX REPLACE "-pedantic" "" LIBCEED_OPT_FLAGS ${LIBCEED_OPT_FLAGS})
  set(LIBCEED_PEDANTIC "1")
else()
  set(LIBCEED_PEDANTIC "")
endif()

# Build libCEED (always as a shared library)
set(LIBCEED_OPTIONS
  "prefix=${CMAKE_INSTALL_PREFIX}"
  "LDFLAGS=${CMAKE_EXE_LINKER_FLAGS}"
  "CC=${CMAKE_C_COMPILER}"
  "CXX=${CMAKE_CXX_COMPILER}"
  "FC="
  "OPT=${LIBCEED_OPT_FLAGS}"
  "STATIC="
  "PEDANTIC=${LIBCEED_PEDANTIC}"
)

# Configure OpenMP
if(PALACE_WITH_OPENMP)
  list(APPEND LIBCEED_OPTIONS
    "OPENMP=1"
  )
endif()

# Configure libCEED backends (nvcc, hipcc flags are configured by libCEED)
if(PALACE_WITH_LIBXSMM)
  list(APPEND LIBCEED_OPTIONS
    "XSMM_DIR=${CMAKE_INSTALL_PREFIX}"
  )
  # LIBXSMM can require linkage with BLAS for fallback
  if(NOT "${BLAS_LAPACK_LIBRARIES}" STREQUAL "")
    string(REPLACE "$<SEMICOLON>" " " LIBCEED_BLAS_LAPACK_LIBRARIES "${BLAS_LAPACK_LIBRARIES}")
    list(APPEND LIBCEED_OPTIONS
      "BLAS_LIB=${LIBCEED_BLAS_LAPACK_LIBRARIES}"
    )
  endif()
endif()
if(PALACE_WITH_CUDA)
  list(APPEND LIBCEED_OPTIONS
    "CUDA_DIR=${CUDAToolkit_LIBRARY_ROOT}"
  )
  if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    list(GET CMAKE_CUDA_ARCHITECTURES 0 LIBCEED_CUDA_ARCH)
    list(APPEND LIBCEED_OPTIONS
      "CUDA_ARCH=sm_${LIBCEED_CUDA_ARCH}"
    )
  endif()
endif()
if(PALACE_WITH_HIP)
  list(APPEND LIBCEED_OPTIONS
    "ROCM_DIR=${ROCM_DIR}"
  )
  if(NOT "${CMAKE_HIP_ARCHITECTURES}" STREQUAL "")
    list(GET CMAKE_HIP_ARCHITECTURES 0 LIBCEED_HIP_ARCH)
    list(APPEND LIBCEED_OPTIONS
      "HIP_ARCH=${LIBCEED_HIP_ARCH}"
    )
  endif()
endif()
if(PALACE_WITH_MAGMA)
  list(APPEND LIBCEED_OPTIONS
    "MAGMA_DIR=${CMAKE_INSTALL_PREFIX}"
  )
endif()

# Experimental OCCA backend (ceed-occa; see PALACE_WITH_OCCA in the top-level
# CMakeLists.txt and docs/src/developer/apple-metal-occa-progress.md). libCEED's
# own Makefile auto-detects which OCCA modes to register from `occa modes`, run
# against $(OCCA_DIR)/lib/libocca.*, so nothing mode-specific needs to be passed
# here -- this only points libCEED at an OCCA install.
if(PALACE_WITH_OCCA)
  if(PALACE_BUILD_EXTERNAL_DEPS)
    list(APPEND LIBCEED_OPTIONS
      "OCCA_DIR=${CMAKE_INSTALL_PREFIX}"
    )
  else()
    list(APPEND LIBCEED_OPTIONS
      "OCCA_DIR=${OCCA_DIR}"
    )
  endif()
endif()

string(REPLACE ";" "; " LIBCEED_OPTIONS_PRINT "${LIBCEED_OPTIONS}")
message(STATUS "LIBCEED_OPTIONS: ${LIBCEED_OPTIONS_PRINT}")

# Patch for the experimental ceed-occa backend (see PALACE_WITH_OCCA above and
# docs/src/developer/apple-metal-occa-progress.md): as shipped, ceed-occa
# registers hard "backend does not implement X" stubs for
# LinearAssembleQFunction/LinearAssembleAddDiagonal/
# LinearAssembleAddPointBlockDiagonal/CreateFDMElementInverse. Because
# libCEED's generic operator dispatcher only takes its "not supported" path
# when a backend leaves the corresponding function pointer unregistered
# (null), registering these as always-failing stubs actively prevents
# libCEED's own operator-fallback mechanism -- the same mechanism
# backends/cuda-gen and backends/hip-gen use for their own unimplemented
# operations -- from ever being reached. This patch leaves those functions
# unregistered and registers a reference-backend fallback Ceed in
# initCeed() instead, so CeedOperatorLinearAssembleAddDiagonal (used by
# Palace's Jacobi/Chebyshev smoothers) transparently falls back to
# /cpu/self/ref/serial (or /gpu/cuda/ref, /gpu/hip/ref) instead of aborting.
# Also fixes a segfault in CeedOperatorLinearAssembleSymbolic/
# CeedOperatorLinearAssemble (full sparse-matrix assembly, needed by
# Palace's default AMS/AMG-preconditioned solves) caused by
# CeedElemRestrictionApply being handed vectors from a different (fallback)
# Ceed than the restriction itself; ceed-occa's ElemRestriction now detects
# this and routes through a generic, backend-agnostic host implementation.
# Also implements CeedElemRestrictionCreateOriented (sign-flip restrictions,
# used by Palace for tensor-product elements with non-trivial DOF
# orientation), which ceed-occa did not support at all upstream (hard
# CeedError, not just a stub) -- again via the generic host implementation.
# Also implements CeedElemRestrictionCreateCurlOriented (the tridiagonal
# DOF-transformation restriction used by 3D Nedelec/H(curl) elements, e.g.
# Palace's default edge-element spaces), ported directly from the reference
# backend's tridiagonal apply cores and verified against it as an
# independent oracle; see the doc for details and remaining caveats
# (CUDA/HIP untested, no GPU toolchain on the VM this was developed on).
if(PALACE_WITH_OCCA)
  set(LIBCEED_PATCH_FILES
    "${CMAKE_SOURCE_DIR}/extern/patch/libceed/patch_occa_operator_fallback.diff"
  )
  set(LIBCEED_PATCH_COMMAND
    git reset --hard &&
    git clean -fd &&
    git apply "${LIBCEED_PATCH_FILES}"
  )
else()
  set(LIBCEED_PATCH_COMMAND "")
endif()

include(ExternalProject)
ExternalProject_Add(libCEED
  DEPENDS           ${LIBCEED_DEPENDENCIES}
  GIT_REPOSITORY    ${EXTERN_LIBCEED_URL}
  GIT_TAG           ${EXTERN_LIBCEED_GIT_TAG}
  SOURCE_DIR        ${CMAKE_BINARY_DIR}/extern/libCEED
  INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
  PREFIX            ${CMAKE_BINARY_DIR}/extern/libCEED-cmake
  BUILD_IN_SOURCE   TRUE
  UPDATE_COMMAND    ""
  PATCH_COMMAND     ${LIBCEED_PATCH_COMMAND}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ${CMAKE_MAKE_PROGRAM} ${LIBCEED_OPTIONS} install
  TEST_COMMAND      ""
)
