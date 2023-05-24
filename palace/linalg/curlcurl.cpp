// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "curlcurl.hpp"

#include <mfem.hpp>
#include "fem/coefficient.hpp"
#include "linalg/ams.hpp"
#include "linalg/gmg.hpp"
#include "linalg/iterative.hpp"
#include "models/materialoperator.hpp"

namespace palace
{

CurlCurlMassSolver::CurlCurlMassSolver(
    const MaterialOperator &mat_op, mfem::ParFiniteElementSpaceHierarchy &nd_fespaces,
    mfem::ParFiniteElementSpaceHierarchy &h1_fespaces,
    const std::vector<mfem::Array<int>> &nd_dbc_tdof_lists,
    const std::vector<mfem::Array<int>> &h1_dbc_tdof_lists, double tol, int max_it,
    int print)
{
  constexpr MaterialPropertyType MatTypeMuInv = MaterialPropertyType::INV_PERMEABILITY;
  constexpr MaterialPropertyType MatTypeEps = MaterialPropertyType::PERMITTIVITY_REAL;
  MaterialPropertyCoefficient<MatTypeMuInv> muinv_func(mat_op);
  MaterialPropertyCoefficient<MatTypeEps> epsilon_func(mat_op);
  {
    auto A_mg = std::make_unique<MultigridOperator>(nd_fespaces.GetNumLevels());
    for (int s = 0; s < 2; s++)
    {
      auto &fespaces = (s == 0) ? nd_fespaces : h1_fespaces;
      auto &dbc_tdof_lists = (s == 0) ? nd_dbc_tdof_lists : h1_dbc_tdof_lists;
      for (int l = 0; l < fespaces.GetNumLevels(); l++)
      {
        auto &fespace_l = fespaces.GetFESpaceAtLevel(l);
        auto a = std::make_unique<mfem::SymmetricBilinearForm>(&fespace_l);
        if (s == 0)
        {
          a->AddDomainIntegrator(new mfem::CurlCurlIntegrator(muinv_func));
          a->AddDomainIntegrator(new mfem::MixedVectorMassIntegrator(epsilon_func));
        }
        else
        {
          a->AddDomainIntegrator(new mfem::MixedGradGradIntegrator(epsilon_func));
        }
        // XX TODO: Partial assembly option?
        a->SetAssemblyLevel(mfem::AssemblyLevel::LEGACY);
        a->Assemble(0);
        a->Finalize(0);
        auto A_l = std::make_unique<ParOperator>(std::move(a), fespace_l);
        A_l->SetEssentialTrueDofs(dbc_tdof_lists[l], Operator::DiagonalPolicy::DIAG_ONE);
        if (s == 0)
        {
          A_mg.AddOperator(std::move(A_l));
        }
        else
        {
          A_mg.AddAuxiliaryOperator(std::move(A_l));
        }
      }
    }
    A = std::move(A_mg);
  }

  // The system matrix K + M is real and SPD. We use Hypre's AMS solver as the coarse-level
  // multigrid solve.
  auto ams = std::make_unique<HypreAmsSolver>(nd_fespaces.GetFESpaceAtLevel(0),
                                              h1_fespaces.GetFESpaceAtLevel(0), 1, 1, 1,
                                              false, false, 0);
  auto gmg = std::make_unique<GeometricMultigridSolver<Operator>>(
      std::move(ams), nd_fespaces, &h1_fespaces, 1, 1, 2);

  auto pcg =
      std::make_unique<CgSolver<Operator>>(nd_fespaces.GetFinestFESpace().GetComm(), print);
  pcg->SetInitialGuess(false);
  pcg->SetRelTol(tol);
  pcg->SetMaxIter(max_it);

  ksp = std::make_unique<KspSolver>(std::move(pcg), std::move(gmg));
  ksp->SetOperators(*A, *A);
}

}  // namespace palace
