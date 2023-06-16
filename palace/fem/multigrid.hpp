// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef PALACE_FEM_MULTIGRID_HPP
#define PALACE_FEM_MULTIGRID_HPP

#include <memory>
#include <vector>
#include <mfem.hpp>
#include "linalg/operator.hpp"
#include "linalg/rap.hpp"
#include "utils/iodata.hpp"

namespace palace::utils
{

//
// Methods for constructing hierarchies of finite element spaces for geometric multigrid.
//

// Convert a configuration file assembly level enum into MFEM's type.
inline mfem::AssemblyLevel
GetAssemblyLevel(config::SolverData::AssemblyLevel assembly_level)
{
  switch (assembly_level)
  {
    case config::SolverData::AssemblyLevel::PARTIAL:
      return mfem::AssemblyLevel::PARTIAL;
    case config::SolverData::AssemblyLevel::FULL:
    default:
      return mfem::AssemblyLevel::LEGACY;
  }
}

// Construct sequence of FECollection objects.
template <typename FECollection>
std::vector<std::unique_ptr<FECollection>> inline ConstructFECollections(
    int p, int dim, int mg_max_levels,
    config::LinearSolverData::MultigridCoarsenType mg_coarsen_type, bool mat_lor)
{
  // If the solver will use a LOR preconditioner, we need to construct with a specific basis
  // type.
  MFEM_VERIFY(p >= 1, "FE space order must not be less than 1!");
  int b1 = mfem::BasisType::GaussLobatto, b2 = mfem::BasisType::GaussLegendre;
  if (mat_lor)
  {
    b2 = mfem::BasisType::IntegratedGLL;
  }
  auto AddLevel = [dim, b1, b2](int o, std::vector<std::unique_ptr<FECollection>> &fecs)
  {
    constexpr int om1 = (std::is_same<FECollection, mfem::H1_FECollection>::value ||
                         std::is_same<FECollection, mfem::ND_FECollection>::value)
                            ? 0
                            : 1;
    if constexpr (std::is_same<FECollection, mfem::ND_FECollection>::value ||
                  std::is_same<FECollection, mfem::RT_FECollection>::value)
    {
      fecs.push_back(std::make_unique<FECollection>(o - om1, dim, b1, b2));
    }
    else
    {
      fecs.push_back(std::make_unique<FECollection>(o - om1, dim, b1));
      MFEM_CONTRACT_VAR(b2);
    }
  };

  // Construct the p-multigrid hierarchy.
  std::vector<std::unique_ptr<FECollection>> fecs;
  if (mg_max_levels > 1)
  {
    switch (mg_coarsen_type)
    {
      case config::LinearSolverData::MultigridCoarsenType::LINEAR:
        {
          int num_levels = std::min(p, mg_max_levels);
          fecs.reserve(num_levels);
          for (int o = p - num_levels + 1; o <= p; o++)
          {
            AddLevel(o, fecs);
          }
        }
        break;
      case config::LinearSolverData::MultigridCoarsenType::LOGARITHMIC:
        {
          MFEM_VERIFY((p & (p - 1)) == 0, "Multigrid with logarithmic coarsening multigrid "
                                          "requires order to be a power of 2!");
          int num_levels = 1, p_min = p;
          while (num_levels < mg_max_levels && p_min > 1)
          {
            num_levels++;
            p_min >>= 1;
          }
          fecs.reserve(num_levels);
          for (int o = p_min; o <= p; o *= 2)
          {
            AddLevel(o, fecs);
          }
        }
        break;
      default:
        MFEM_ABORT("Invalid coarsening type for p-multigrid levels!");
        break;
    }
  }
  else
  {
    AddLevel(p, fecs);
  }
  return fecs;
}

// Construct a hierarchy of finite element spaces given a sequence of meshes and
// finite element collections. Dirichlet boundary conditions are additionally
// marked.
template <typename FECollection>
inline mfem::ParFiniteElementSpaceHierarchy ConstructFiniteElementSpaceHierarchy(
    int mg_max_levels, const std::vector<std::unique_ptr<mfem::ParMesh>> &mesh,
    const std::vector<std::unique_ptr<FECollection>> &fecs,
    const mfem::Array<int> *dbc_marker = nullptr,
    std::vector<mfem::Array<int>> *dbc_tdof_lists = nullptr)
{

  // XX TODO: LibCEED transfer operators!

  MFEM_VERIFY(!mesh.empty() && !fecs.empty() &&
                  (!dbc_tdof_lists || dbc_tdof_lists->empty()),
              "Empty mesh or FE collection for FE space construction!");
  auto mesh_levels = std::min(mesh.size() - 1, mg_max_levels - fecs.size());
  auto *fespace = new mfem::ParFiniteElementSpace(mesh[mesh.size() - mesh_levels - 1].get(),
                                                  fecs[0].get());
  if (dbc_marker && dbc_tdof_lists)
  {
    fespace->GetEssentialTrueDofs(*dbc_marker, dbc_tdof_lists->emplace_back());
  }
  mfem::ParFiniteElementSpaceHierarchy fespaces(mesh[mesh.size() - mesh_levels - 1].get(),
                                                fespace, false, true);

  // h-refinement
  for (std::size_t l = mesh.size() - mesh_levels; l < mesh.size(); l++)
  {
    fespace = new mfem::ParFiniteElementSpace(mesh[l].get(), fecs[0].get());
    if (dbc_marker && dbc_tdof_lists)
    {
      fespace->GetEssentialTrueDofs(*dbc_marker, dbc_tdof_lists->emplace_back());
    }
    auto *P = new ParOperator(
        std::make_unique<mfem::TransferOperator>(fespaces.GetFinestFESpace(), *fespace),
        fespaces.GetFinestFESpace(), *fespace, true);
    fespaces.AddLevel(mesh[l].get(), fespace, P, false, true, true);
  }

  // p-refinement
  for (std::size_t l = 1; l < fecs.size(); l++)
  {
    fespace = new mfem::ParFiniteElementSpace(mesh.back().get(), fecs[l].get());
    if (dbc_marker && dbc_tdof_lists)
    {
      fespace->GetEssentialTrueDofs(*dbc_marker, dbc_tdof_lists->emplace_back());
    }
    auto *P = new ParOperator(
        std::make_unique<mfem::TransferOperator>(fespaces.GetFinestFESpace(), *fespace),
        fespaces.GetFinestFESpace(), *fespace, true);
    fespaces.AddLevel(mesh.back().get(), fespace, P, false, true, true);
  }
  return fespaces;
}

}  // namespace palace::utils

#endif  // PALACE_FEM_MULTIGRID_HPP
