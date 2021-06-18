/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *    
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *******************************************************************************/

/******************************************************************************
 *  
 * AMG solve routine
 * 
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_fsai.h"

/*--------------------------------------------------------------------
 * hypre_FSAISolve
 *--------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISolve( void               *fsai_vdata,
                   hypre_ParCSRMatrix *A         )
{
   MPI_Comm             comm = hypre_ParCSRMatrixComm(A);

   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   HYPRE_Int      fsai_print_level;
   HYPRE_Int      fsai_logging;
   HYPRE_Real     kap_tolerance;
   HYPRE_Int      max_steps;
   HYPRE_Int      max_step_size;
   HYPRE_Int      num_rows;
   
   HYPRE_ParCSRMatrix   *G_mat;
   HYPRE_ParCSRMatrix   *S_pattern;
   HYPRE_ParVector      *kaporin_gradient;
   HYPRE_ParVector      *nnz_per_row;        /* For GPU */
   HYPRE_ParVector      *nnz_cum_sum;        /* For GPU */
   

   /* Local variables */

   HYPRE_Int            num_procs, my_id;
   HYPRE_Int            rows_per_proc, i, j, k;
   HYPRE_Int            d; /* Diagonal scaling */
   hypre_ParVector      *old_psi;
   hypre_ParVector      *new_psi;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   fsai_print_level     = hypre_ParFSAIDataPrintLevel(fsai_data);
   fsai_logging         = hypre_ParFSAIDataLogging(fsai_data);
   tolerance            = hypre_ParFSAIDataTolerence(fsai_data);
   max_steps            = hypre_ParFSAIDataMaxSteps(fsai_data);
   max_step_size        = hypre_ParFSAIDataMaxStepSize(fsai_data);
   num_rows             = hypre_ParFSAIDataNumRows(fsai_data);
   G_mat                = hypre_ParFSAIDataGmat(fsai_data);
   S_Pattern            = hypre_ParFSAIDataSPattern(fsai_data);
   kaporin_gradient     = hypre_ParFSAIDataKaporinGradient(fsai_data);

   rows_per_proc        = int(1.0 + num_rows/num_procs);


   /* TODO: This is the best I could think of at the moment */
   for(i = num_proc*rows_per_proc; i < min(num_rows, (num_proc+1)*rows_per_proc); ++i)
   {
      G_mat[i][i] = 1;     /* G needsto originally be the identity matrix */

      for(k = 0; k < max_steps; ++k)
      {
         /* Compute the Kaporin Gradient */
         for(j = 0; j < i-1; ++j)
         {
            kaporin_gradient[j] = 2 * (ParVectorInnerProd(A[j], G_mat[i]) + A[i][j]);
         }

         /* TODO: Find max_step_size largest values of kaporin_gradient and put merge indices with S_Pattern[i]*/
         
         /* old_psi = G_mat[i, P]*A[P, P]*G[i, P]^T */
         /* Gather A[P, P] and G[i, P] */
         /* Solve G[i, P] = A[P, P]\(-A[P, i]) */
         /* new_psi = G_mat[i, P]*A[P, P]*G[i, P]^T */

         if((HYPRE_Real)abs(old_psi - new_psi)/old_psi < kap_tolerance)
            break;
          
      }
      
      /* d = 1/sqrt(A[i][i] - abs(ParVectorInnerProd(G_mat[i][P]), A[i][P])) */
      /* G_mat[i] = d*G_mat[i]; */

   }     

   

   HYPRE_ANNOTATE_FUNC_END;

}
