/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_parcsr_ls.h"

HYPRE_Int
hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix    *S,
                                  hypre_ParCSRMatrix    *A,
                                  HYPRE_Int              CF_init,
                                  HYPRE_Int              debug_flag,
                                  HYPRE_Int            **CF_marker_ptr )
{
   hypre_CSRMatrix          *S_diag          = hypre_ParCSRMatrixDiag(S);
   HYPRE_Int                *S_diag_i        = hypre_CSRMatrixI(S_diag);
   HYPRE_Int                *S_diag_j        = hypre_CSRMatrixJ(S_diag);

   hypre_CSRMatrix          *S_offd          = hypre_ParCSRMatrixOffd(S);
   HYPRE_Int                *S_offd_i        = hypre_CSRMatrixI(S_offd);
   HYPRE_Int                *S_offd_j        = hypre_CSRMatrixJ(S_offd);

   HYPRE_Int                 num_cols_diag   = hypre_CSRMatrixNumCols(S_diag);
   HYPRE_Int                 num_cols_offd   = hypre_CSRMatrixNumCols(S_offd);

   HYPRE_Real               *measure_diag;
   HYPRE_Real               *measure_offd;
   HYPRE_Int                 ierr = 0;

   measure_diag = hypre_TAlloc(HYPRE_Real, num_cols_diag, HYPRE_MEMORY_DEVICE);
   measure_offd = hypre_TAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------
    * Compute the global measures
    * The measures are currently given by the column sums of S
    * Hence, measure_array[i] is the number of influences of variable i
    * The measures are augmented by a random number between 0 and 1
    *-------------------------------------------------------------------*/
   hypre_GetGlobalMeasureDevice(S, CF_init, measure_diag, measure_offd);

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   if (CF_init == 2 || CF_init == 4)
   {
      hypre_BoomerAMGIndepSetInitDevice(S, measure_diag, 1);
   }
   else
   {
      hypre_BoomerAMGIndepSetInitDevice(S, measure_diag, 0);
   }
   return ierr;
}

