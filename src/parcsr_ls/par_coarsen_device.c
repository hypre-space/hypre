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

#if defined(HYPRE_USING_CUDA)

HYPRE_Int hypre_PMISCoarseningInitDevice( hypre_ParCSRMatrix *S, HYPRE_Int CF_init, HYPRE_Real *measure_diag, HYPRE_Int *graph_diag_size, HYPRE_Int *graph_offd_size, HYPRE_Int *graph_diag, HYPRE_Int *graph_offd, HYPRE_Int *CF_marker_diag, HYPRE_Int *CF_marker_offd);

HYPRE_Int hypre_PMISCoarseningUpdateCFDevice( hypre_ParCSRMatrix *S, HYPRE_Real *measure_diag, HYPRE_Int graph_diag_size, HYPRE_Int *graph_diag, HYPRE_Int *CF_marker_diag, HYPRE_Int *CF_marker_offd);

HYPRE_Int hypre_PMISCoarseningUpdateGraphDevice( HYPRE_Int *graph_diag_size, HYPRE_Int *graph_offd_size, HYPRE_Int *graph_diag, HYPRE_Int *graph_offd, HYPRE_Int *CF_marker_diag, HYPRE_Int *CF_marker_offd, HYPRE_Int *graph_diag_2, HYPRE_Int *graph_offd_2);

HYPRE_Int
hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix    *S,
                                  hypre_ParCSRMatrix    *A,
                                  HYPRE_Int              CF_init,
                                  HYPRE_Int              debug_flag,
                                  HYPRE_Int            **CF_marker_ptr )
{
   MPI_Comm                  comm            = hypre_ParCSRMatrixComm(S);
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

   HYPRE_Int                 graph_diag_size;
   HYPRE_Int                 graph_offd_size;
   HYPRE_Int                *graph_diag;
   HYPRE_Int                *graph_offd;
   HYPRE_Int                *graph_diag_2;
   HYPRE_Int                *graph_offd_2;

   HYPRE_Int                *CF_marker_diag;
   HYPRE_Int                *CF_marker_offd;

   HYPRE_Int                 ierr = 0;
   HYPRE_Int                 iter = 0;
   HYPRE_Int                *temp;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PMIS] -= hypre_MPI_Wtime();
#endif

   /* CF marker */
   CF_marker_diag = hypre_TAlloc(HYPRE_Int, num_cols_diag, HYPRE_MEMORY_DEVICE);
   CF_marker_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE);

   /* arrays for global measure diag and offd parts */
   measure_diag = hypre_TAlloc(HYPRE_Real, num_cols_diag, HYPRE_MEMORY_DEVICE);
   measure_offd = hypre_TAlloc(HYPRE_Real, num_cols_offd, HYPRE_MEMORY_DEVICE);

   /* arrays for dofs that are still in the graph (undetermined) */
   graph_diag = hypre_TAlloc(HYPRE_Int, num_cols_diag, HYPRE_MEMORY_DEVICE);
   graph_offd = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE);

   graph_diag_2 = hypre_TAlloc(HYPRE_Int, num_cols_diag, HYPRE_MEMORY_DEVICE);
   graph_offd_2 = hypre_TAlloc(HYPRE_Int, num_cols_offd, HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------
    * Compute the global measures
    * The measures are currently given by the column sums of S
    * Hence, measure_array[i] is the number of influences of variable i
    * The measures are augmented by a random number between 0 and 1
    *-------------------------------------------------------------------*/
   hypre_GetGlobalMeasureDevice(S, CF_init, 1, measure_diag, measure_offd);

   /* initialize CF marker ang graph arrays */
   hypre_PMISCoarseningInitDevice(S, CF_init, measure_diag, &graph_diag_size, &graph_offd_size,
                                  graph_diag, graph_offd, CF_marker_diag, CF_marker_offd);

   while (1)
   {
      HYPRE_BigInt big_graph_size, global_graph_size;

      big_graph_size = graph_diag_size;

      /* stop the coarsening if nothing left to be coarsened */
      hypre_MPI_Allreduce(&big_graph_size, &global_graph_size, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

      if (global_graph_size == 0)
      {
         break;
      }

      if (!CF_init || iter)
      {
         hypre_BoomerAMGIndepSetDevice(S, measure_diag, measure_offd, graph_diag_size, graph_offd_size,
                                       graph_diag, graph_offd, CF_marker_diag, CF_marker_offd);
      }

      iter ++;

      /* Set C-pts and F-pts */
      hypre_PMISCoarseningUpdateCFDevice(S, measure_diag, graph_diag_size, graph_diag,
                                         CF_marker_diag, CF_marker_offd);

      /* Update graph: remove C and F pts */
      hypre_PMISCoarseningUpdateGraphDevice(&graph_diag_size, &graph_offd_size, graph_diag, graph_offd,
                                            CF_marker_diag, CF_marker_offd, graph_diag_2, graph_offd_2);

      temp = graph_diag;  graph_diag = graph_diag_2;  graph_diag_2 = temp;
      temp = graph_offd;  graph_offd = graph_offd_2;  graph_offd_2 = temp;
   }

   /*---------------------------------------------------
    * Clean up and return
    *---------------------------------------------------*/
   hypre_TFree(measure_diag,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(measure_offd,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(graph_diag,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(graph_offd,     HYPRE_MEMORY_DEVICE);
   hypre_TFree(graph_diag_2,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(graph_offd_2,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(CF_marker_offd, HYPRE_MEMORY_DEVICE);

   *CF_marker_ptr = CF_marker_diag;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_PMIS] += hypre_MPI_Wtime();
#endif

   return ierr;
}

HYPRE_Int
hypre_GetGlobalMeasureDevice( hypre_ParCSRMatrix *S,
                              HYPRE_Int           CF_init,
                              HYPRE_Int           aug_rand,
                              HYPRE_Real         *measure_diag,
                              HYPRE_Real         *measure_offd)
{
   /* compute global column nnz */

   /* this augments the measures with a random number between 0 and 1 */
   /* (only for the local part) */
   if (aug_rand)
   {
      hypre_BoomerAMGIndepSetInitDevice(S, measure_diag, aug_rand);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_PMISCoarseningInitDevice( hypre_ParCSRMatrix *S,               /* in */
                                HYPRE_Int           CF_init,         /* in */
                                HYPRE_Real         *measure_diag,    /* in */
                                HYPRE_Int          *graph_diag_size, /* out */
                                HYPRE_Int          *graph_offd_size, /* out */
                                HYPRE_Int          *graph_diag,      /* out */
                                HYPRE_Int          *graph_offd,      /* out */
                                HYPRE_Int          *CF_marker_diag,  /* out */
                                HYPRE_Int          *CF_marker_offd ) /* out */
{
   return hypre_error_flag;
}

HYPRE_Int
hypre_PMISCoarseningUpdateCFDevice( hypre_ParCSRMatrix *S,               /* in */
                                    HYPRE_Real         *measure_diag,    /* in */
                                    HYPRE_Int           graph_diag_size, /* in */
                                    HYPRE_Int          *graph_diag,      /* in */
                                    HYPRE_Int          *CF_marker_diag,  /* in/out */
                                    HYPRE_Int          *CF_marker_offd ) /* in/out */
{
   return hypre_error_flag;
}

/* Prune graph arrays by removing pts with CF_marker != 0 */
HYPRE_Int
hypre_PMISCoarseningUpdateGraphDevice( HYPRE_Int *graph_diag_size, /* in/out */
                                       HYPRE_Int *graph_offd_size, /* in/out */
                                       HYPRE_Int *graph_diag,      /* in */
                                       HYPRE_Int *graph_offd,      /* in */
                                       HYPRE_Int *CF_marker_diag,  /* in */
                                       HYPRE_Int *CF_marker_offd,  /* in */
                                       HYPRE_Int *graph_diag_2,    /* out */
                                       HYPRE_Int *graph_offd_2 )   /* out */
{
   return hypre_error_flag;
}

#else // #if defined(HYPRE_USING_CUDA)

HYPRE_Int
hypre_BoomerAMGCoarsenPMISDevice( hypre_ParCSRMatrix *S,
                                  hypre_ParCSRMatrix *A,
                                  HYPRE_Int           CF_init,
                                  HYPRE_Int           debug_flag,
                                  HYPRE_Int         **CF_marker_ptr )
{
   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUDA)
