// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "device.hpp"
#include "communication.hpp"

#include <mfem.hpp>

namespace palace::utils
{

int GetDeviceCount()
{
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
  return mfem::Device::GetDeviceCount();
#else
  return 0;
#endif
}

int GetDeviceId(MPI_Comm comm, int ngpu)
{
  // Assign devices round-robin over MPI ranks if GPU support is enabled.
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
  return Mpi::Rank(SharedMemoryComm(comm)) % ngpu;
#else
  return 0;
#endif
}

}  // namespace palace::utils
