/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA)

HYPRE_Int
hypre_BoomerAMGDD_FAC_Jacobi_device( void *amgdd_vdata, HYPRE_Int level )
{
   HYPRE_Int i,j;

   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   hypre_AMGDDCompGridMatrix *A = hypre_AMGDDCompGridA(compGrid);
   hypre_AMGDDCompGridVector *f = hypre_AMGDDCompGridF(compGrid);
   hypre_AMGDDCompGridVector *u = hypre_AMGDDCompGridU(compGrid);
   HYPRE_Real relax_weight = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   // Calculate l1_norms if necessary (right now, I'm just using this vector for the diagonal of A and doing straight ahead Jacobi)
   if (!hypre_AMGDDCompGridL1Norms(compGrid))
   {
      HYPRE_Int total_real_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) + hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
      hypre_AMGDDCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real, total_real_nodes, hypre_AMGDDCompGridMemoryLocation(compGrid));
      hypre_CSRMatrix *diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_AMGDDCompGridL1Norms(compGrid)[i] = hypre_CSRMatrixData(diag)[j];
         }
      }
      diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i+1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += fabs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i) hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] = hypre_CSRMatrixData(diag)[j];
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }

   hypre_AMGDDCompGridVectorCopy(f, hypre_AMGDDCompGridTemp2(compGrid));

   hypre_AMGDDCompGridMatvec(-relax_weight, A, u, relax_weight, hypre_AMGDDCompGridTemp2(compGrid));

   hypreDevice_IVAXPY(hypre_AMGDDCompGridNumOwnedNodes(compGrid),
         hypre_AMGDDCompGridL1Norms(compGrid),
         hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid))),
         hypre_VectorData(hypre_AMGDDCompGridVectorOwned(u)));
   hypreDevice_IVAXPY(hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid),
         &(hypre_AMGDDCompGridL1Norms(compGrid)[ hypre_AMGDDCompGridNumOwnedNodes(compGrid) ]),
         hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid))),
         hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(u)));

   return 0;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_CFL1Jacobi_device( void *amgdd_vdata, HYPRE_Int level, HYPRE_Int relax_set )
{
   hypre_ParAMGDDData *amgdd_data = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid *compGrid = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   HYPRE_Real relax_weight = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);

   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedNodes(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }
   hypre_AMGDDCompGridVectorCopy(hypre_AMGDDCompGridF(compGrid), hypre_AMGDDCompGridTemp2(compGrid));
   double alpha = -relax_weight;
   double beta = relax_weight;

   HYPRE_Complex *owned_u = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(compGrid)));
   HYPRE_Complex *nonowned_u = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridU(compGrid)));
   HYPRE_Complex *owned_tmp = hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid)));
   HYPRE_Complex *nonowned_tmp = hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid)));

   if (relax_set)
   {
      hypre_CSRMatrix *mat = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, owned_u, beta, owned_tmp, owned_tmp,
            hypre_AMGDDCompGridOwnedCMask(compGrid), hypre_AMGDDCompGridNumOwnedCPoints(compGrid), 0);

      mat = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, nonowned_u, beta, owned_tmp, owned_tmp,
            hypre_AMGDDCompGridOwnedCMask(compGrid), hypre_AMGDDCompGridNumOwnedCPoints(compGrid), 0);

      mat = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, nonowned_u, beta, nonowned_tmp, nonowned_tmp,
            hypre_AMGDDCompGridNonOwnedCMask(compGrid), hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid), 0);

      mat = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, owned_u, beta, nonowned_tmp, nonowned_tmp,
            hypre_AMGDDCompGridNonOwnedCMask(compGrid), hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid), 0);

      hypreDevice_MaskedIVAXPY(hypre_AMGDDCompGridNumOwnedCPoints(compGrid),
            hypre_AMGDDCompGridL1Norms(compGrid),
            owned_tmp,
            owned_u,
            hypre_AMGDDCompGridOwnedCMask(compGrid));
      hypreDevice_MaskedIVAXPY(hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid),
            &(hypre_AMGDDCompGridL1Norms(compGrid)[ hypre_AMGDDCompGridNumOwnedNodes(compGrid) ]),
            nonowned_tmp,
            nononwed_u,
            hypre_AMGDDCompGridNonOwnedCMask(compGrid));
   }
   else
   {
      hypre_CSRMatrix *mat = hypre_AMGDDCompGridMatrixOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, owned_u, beta, owned_tmp, owned_tmp,
            hypre_AMGDDCompGridOwnedFMask(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid) - hypre_AMGDDCompGridNumOwnedCPoints(compGrid), 0);

      mat = hypre_AMGDDCompGridMatrixOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, nonowned_u, beta, owned_tmp, owned_tmp,
            hypre_AMGDDCompGridOwnedFMask(compGrid), hypre_AMGDDCompGridNumOwnedNodes(compGrid) - hypre_AMGDDCompGridNumOwnedCPoints(compGrid), 0);

      mat = hypre_AMGDDCompGridMatrixNonOwnedDiag(hypre_AMGDDCompGridA(compGrid));
      beta = relax_weight;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, nonowned_u, beta, nonowned_tmp, nonowned_tmp,
            hypre_AMGDDCompGridNonOwnedFMask(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) - hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid), 0);

      mat = hypre_AMGDDCompGridMatrixNonOwnedOffd(hypre_AMGDDCompGridA(compGrid));
      beta = 1.0;
      hypre_CSRMatrixMatvecMaskedDevice(0, alpha, mat, owned_u, beta, nonowned_tmp, nonowned_tmp,
            hypre_AMGDDCompGridNonOwnedFMask(compGrid), hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) - hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid), 0);

      hypreDevice_MaskedIVAXPY(hypre_AMGDDCompGridNumOwnedNodes(compGrid) - hypre_AMGDDCompGridNumOwnedCPoints(compGrid),
            hypre_AMGDDCompGridL1Norms(compGrid),
            owned_tmp,
            owned_u,
            hypre_AMGDDCompGridOwnedFMask(compGrid));
      hypreDevice_MaskedIVAXPY(hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid) - hypre_AMGDDCompGridNumNonOwnedRealCPoints(compGrid),
            &(hypre_AMGDDCompGridL1Norms(compGrid)[ hypre_AMGDDCompGridNumOwnedNodes(compGrid) ]),
            nonowned_tmp,
            nononwed_u,
            hypre_AMGDDCompGridNonOwnedFMask(compGrid));
   }

   return 0;
}


#endif
