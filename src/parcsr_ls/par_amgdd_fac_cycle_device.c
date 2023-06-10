/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

HYPRE_Int
hypre_BoomerAMGDD_FAC_JacobiDevice( void     *amgdd_vdata,
                                    HYPRE_Int level )
{
   hypre_ParAMGDDData         *amgdd_data      = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid        *compGrid        = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   HYPRE_Real                  relax_weight    = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);
   HYPRE_MemoryLocation        memory_location = hypre_AMGDDCompGridMemoryLocation(compGrid);

   hypre_AMGDDCompGridMatrix  *A = hypre_AMGDDCompGridA(compGrid);
   hypre_AMGDDCompGridVector  *f = hypre_AMGDDCompGridF(compGrid);
   hypre_AMGDDCompGridVector  *u = hypre_AMGDDCompGridU(compGrid);

   hypre_CSRMatrix            *diag;
   HYPRE_Int                   total_real_nodes;
   HYPRE_Int                   i, j;

   // Calculate l1_norms if necessary (right now, I'm just using this vector for the diagonal of A and doing straight ahead Jacobi)
   if (!hypre_AMGDDCompGridL1Norms(compGrid))
   {
      total_real_nodes = hypre_AMGDDCompGridNumOwnedNodes(compGrid) +
                         hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);
      hypre_AMGDDCompGridL1Norms(compGrid) = hypre_CTAlloc(HYPRE_Real,
                                                           total_real_nodes,
                                                           memory_location);
      diag = hypre_AMGDDCompGridMatrixOwnedDiag(A);

      for (i = 0; i < hypre_AMGDDCompGridNumOwnedNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i + 1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i] += hypre_abs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i)
            {
               hypre_AMGDDCompGridL1Norms(compGrid)[i] = hypre_CSRMatrixData(diag)[j];
            }
         }
      }

      diag = hypre_AMGDDCompGridMatrixNonOwnedDiag(A);
      for (i = 0; i < hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid); i++)
      {
         for (j = hypre_CSRMatrixI(diag)[i]; j < hypre_CSRMatrixI(diag)[i + 1]; j++)
         {
            // hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(compGrid)] += hypre_abs(hypre_CSRMatrixData(diag)[j]);
            if (hypre_CSRMatrixJ(diag)[j] == i)
            {
               hypre_AMGDDCompGridL1Norms(compGrid)[i + hypre_AMGDDCompGridNumOwnedNodes(
                                                       compGrid)] = hypre_CSRMatrixData(diag)[j];
            }
         }
      }
   }

   // Allocate temporary vector if necessary
   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid),
                                          hypre_AMGDDCompGridNumOwnedNodes(compGrid),
                                          hypre_AMGDDCompGridNumNonOwnedNodes(compGrid),
                                          hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid));
   }

   hypre_AMGDDCompGridVectorCopy(f, hypre_AMGDDCompGridTemp2(compGrid));

   hypre_AMGDDCompGridMatvec(-relax_weight, A, u, relax_weight, hypre_AMGDDCompGridTemp2(compGrid));

   hypreDevice_IVAXPY(hypre_AMGDDCompGridNumOwnedNodes(compGrid),
                      hypre_AMGDDCompGridL1Norms(compGrid),
                      hypre_VectorData(hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid))),
                      hypre_VectorData(hypre_AMGDDCompGridVectorOwned(u)));

   hypreDevice_IVAXPY(hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid),
                      &(hypre_AMGDDCompGridL1Norms(compGrid)[hypre_AMGDDCompGridNumOwnedNodes(compGrid)]),
                      hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid))),
                      hypre_VectorData(hypre_AMGDDCompGridVectorNonOwned(u)));

   return hypre_error_flag;
}

HYPRE_Int
hypre_BoomerAMGDD_FAC_CFL1JacobiDevice( void      *amgdd_vdata,
                                        HYPRE_Int  level,
                                        HYPRE_Int  relax_set )
{
   hypre_ParAMGDDData    *amgdd_data      = (hypre_ParAMGDDData*) amgdd_vdata;
   hypre_AMGDDCompGrid   *compGrid        = hypre_ParAMGDDDataCompGrid(amgdd_data)[level];
   HYPRE_Real             relax_weight    = hypre_ParAMGDDDataFACRelaxWeight(amgdd_data);
   hypre_Vector          *owned_u         = hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridU(
                                                                              compGrid));
   hypre_Vector          *nonowned_u      = hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridU(
                                                                                 compGrid));
   HYPRE_Int              num_owned       = hypre_AMGDDCompGridNumOwnedNodes(compGrid);
   HYPRE_Int              num_nonowned    = hypre_AMGDDCompGridNumNonOwnedNodes(compGrid);
   HYPRE_Int              num_nonowned_r  = hypre_AMGDDCompGridNumNonOwnedRealNodes(compGrid);

   hypre_Vector          *owned_tmp;
   hypre_Vector          *nonowned_tmp;

   // Allocate temporary vector if necessary
   if (!hypre_AMGDDCompGridTemp2(compGrid))
   {
      hypre_AMGDDCompGridTemp2(compGrid) = hypre_AMGDDCompGridVectorCreate();
      hypre_AMGDDCompGridVectorInitialize(hypre_AMGDDCompGridTemp2(compGrid),
                                          num_owned,
                                          num_nonowned,
                                          num_nonowned_r);
   }

   hypre_AMGDDCompGridVectorCopy(hypre_AMGDDCompGridF(compGrid),
                                 hypre_AMGDDCompGridTemp2(compGrid));

   hypre_AMGDDCompGridMatvec(-relax_weight,
                             hypre_AMGDDCompGridA(compGrid),
                             hypre_AMGDDCompGridU(compGrid),
                             relax_weight,
                             hypre_AMGDDCompGridTemp2(compGrid));

   owned_tmp    = hypre_AMGDDCompGridVectorOwned(hypre_AMGDDCompGridTemp2(compGrid));
   nonowned_tmp = hypre_AMGDDCompGridVectorNonOwned(hypre_AMGDDCompGridTemp2(compGrid));

   hypreDevice_IVAXPYMarked(num_owned,
                            hypre_AMGDDCompGridL1Norms(compGrid),
                            hypre_VectorData(owned_tmp),
                            hypre_VectorData(owned_u),
                            hypre_AMGDDCompGridCFMarkerArray(compGrid),
                            relax_set);

   hypreDevice_IVAXPYMarked(num_nonowned_r,
                            &(hypre_AMGDDCompGridL1Norms(compGrid)[num_owned]),
                            hypre_VectorData(nonowned_tmp),
                            hypre_VectorData(nonowned_u),
                            hypre_AMGDDCompGridCFMarkerArray(compGrid) + num_owned,
                            relax_set);

   return hypre_error_flag;
}

#endif // defined(HYPRE_USING_GPU)
