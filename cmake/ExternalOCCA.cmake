# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Build OCCA (experimental; see PALACE_WITH_OCCA in the top-level CMakeLists.txt and
# docs/src/developer/apple-metal-occa-progress.md). This is only useful today as a step
# toward Apple Metal GPU support via MFEM's own experimental OCCA integration -- Palace's
# own operators (curl-curl, mass, etc. in palace/fem/) are built on libCEED, not OCCA/MFEM
# BilinearForm partial assembly, and do not gain anything from this on their own.
#

set(OCCA_OPTIONS ${PALACE_SUPERBUILD_DEFAULT_ARGS})
list(APPEND OCCA_OPTIONS
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
  "-DOCCA_ENABLE_TESTS=OFF"
  "-DOCCA_ENABLE_EXAMPLES=OFF"
)

# OCCA's own CMake build auto-detects and enables each backend mode independently
# (OCCA_ENABLE_OPENMP/CUDA/OPENCL/HIP/METAL/DPCPP all default ON, but each one's actual
# effect is gated on the corresponding toolkit/framework being found -- Metal specifically
# requires `OCCA_ENABLE_METAL AND APPLE`). Nothing GPU-specific needs to be forced here:
# on a non-Apple platform this simply builds OCCA's Serial (and, if available,
# OpenMP/CUDA/HIP) modes with Metal support compiled out; on macOS with Xcode available it
# builds Metal support automatically.

string(REPLACE ";" "; " OCCA_OPTIONS_PRINT "${OCCA_OPTIONS}")
message(STATUS "OCCA_OPTIONS: ${OCCA_OPTIONS_PRINT}")

include(ExternalProject)
ExternalProject_Add(occa
  GIT_REPOSITORY    ${EXTERN_OCCA_URL}
  GIT_TAG           ${EXTERN_OCCA_GIT_TAG}
  SOURCE_DIR        ${CMAKE_BINARY_DIR}/extern/occa
  BINARY_DIR        ${CMAKE_BINARY_DIR}/extern/occa-build
  INSTALL_DIR       ${CMAKE_INSTALL_PREFIX}
  PREFIX            ${CMAKE_BINARY_DIR}/extern/occa-cmake
  UPDATE_COMMAND    ""
  CONFIGURE_COMMAND ${CMAKE_COMMAND} <SOURCE_DIR> "${OCCA_OPTIONS}"
  TEST_COMMAND      ""
)
