/******************************************************************************
 *  Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 *  HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *  
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
#include "par_fsai.h"

#define DEBUG 0
#define PRINT_CF 0
#define DEBUG_SAVE_ALL_OPS 0

/*****************************************************************************
 *  
 * Routine for driving the setup phase of FSAI
 *
 ******************************************************************************/

/******************************************************************************
 * Helper functions. Will move later.
 ******************************************************************************/

/* TODO - Extract A[P, P] into dense matrix */
void
hypre_CSRMatrixExtractDenseMatrix(HYPRE_Real *A_sub, HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Int *A_data, HYPRE_Int *marker, HYPRE_Int *needed_rows, HYPRE_Int nrows_needed)
{

   HYPRE_Int rr, cc;    /* Local dense matrix row and column counter */ 
   HYPRE_Int i, j;      /* Loop variables */
   HYPRE_Int count = 0;

   for(i = 0; i < nrows_needed; i++)
     marker[needed_rows[i]] = count++;    /* Since A[P, P] is symmetric, we mark the same columns as we do rows */  

   for(i = 0; i < nrows_needed; i++)
   {
      rr = needed_rows[i];
      for(j = A_i[rr]; j < A_i[rr+1]; j++)
      {
         if((cc = marker[A_j[j]]) >= 0)
            A_sub[rr + cc*nrows_needed] = A_data[j];
      }
   }

   for(i = 0; i < nrows_needed; i++)
     marker[needed_rows[i]] = -1;    /* Reset marker work array for future use */  

   return;

}

/* Extract the dense sub-row from a matrix (A[i, P]) */
void
hypre_ParCSRMatrixExtractDenseRowFromMatrix(HYPRE_Real *sub_row, HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Real *A_data, HYPRE_Int *marker, HYPRE_Int needed_row, HYPRE_Int *needed_cols, HYPRE_Int ncols_needed)
{

   HYPRE_Int i, cc;
   HYPRE_Int count = 0;

   for(i = 0; i < ncols_needed; i++)
      marker[needed_cols[i]] = count++;

   for(i = A_i[needed_row]; i < A_i[needed_row+1]; i++)
      if((cc = marker[A_j[i]]) >= 0)
         sub_row[cc] = A_data[i];

   for(i = 0; i < ncols_needed; i++)
      marker[needed_cols[i]] = -1;

   return;
}

/*****************************************************************************
 * hypre_FSAISetup
 ******************************************************************************/

HYPRE_Int
hypre_FSAISetup( void               *fsai_vdata,
                      hypre_ParCSRMatrix *A  )
{
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   hypre_ParFSAIData       *fsai_data = (hypre_ParFSAIData*) fsai_vdata;
   hypre_MemoryLocation    memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /* Data structure variables */

   HYPRE_Real              kap_tolerance           = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps               = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size           = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               logging                 = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Int               print_level             = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int               debug_flag;             = hypre_ParFSAIDataDebugFlag(fsai_data);

   /* Declare Local variables */

   HYPRE_Int               num_procs, my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   HYPRE_CSRMatrix         *A_diag;
   HYPRE_CSRMatrix         *G;
   HYPRE_Real              *G_temp;
   HYPRE_Real              *A_sub;
   HYPRE_Real              *kaporin_gradient;
   HYPRE_Real              *A_data;
   HYPRE_Int               *A_i;
   HYPRE_Int               *A_j;
   HYPRE_Int               *row_partition;
   HYPRE_Int               *S_Pattern;
   HYPRE_Int               *markers;
   HYPRE_Real              old_psi, new_psi;
   HYPRE_Real              row_scale;
   HYPRE_Int               num_rows;
   HYPRE_Int               min_row_size;
   HYPRE_Int               i, j, k;       /* Loop variables */
   
   /* Setting local variables */

   A_diag                  = hypre_ParCSRMatrixDiag(A);
   A_i                     = hypre_CSRMatrixI(A_diag);
   A_j                     = hypre_CSRMatrixJ(A_diag);
   A_data                  = hypre_CSRMatrixData(A_diag);
   num_rows                = hypre_CSRMatrixNumRows(A_diag);
   num_cols                = hypre_CSRMatrixNumCols(A_diag);
                          
   /* Allocating local variables */
   
   min_row_size            = min(max_steps*max_step_size, num_rows-1);
   kaporin_gradient        = hypre_CTAlloc(HYPRE_Real, min_row_size, HYPRE_MEMORY_HOST);
   G_temp                  = hypre_CTAlloc(HYPRE_Real, min_row_size, HYPRE_MEMORY_HOST);
   S_Pattern               = hypre_CTAlloc(HYPRE_Int, min_row_size, HYPRE_MEMORY_HOST);
   markers                 = hypre_CTAlloc(HYPRE_Int, num_cols, HYPRE_MEMORY_HOST);       /* For gather functions - don't want to reinitialize */
   for( i = 0; i < num_cols; i++ )
      markers[i] = -1;


   /**********************************************************************
   * Start of Adaptive FSAI algorithm  
   ***********************************************************************/

   for( i = 0; i < num_rows; i++ ){    /* Cycle through each of the local rows */

      for( k = 0; k < max_steps; k++ ){      /* Cycle through each iteration for that row */
         
         /* Steps:
         * Compute Kaporin Gradient
         *  1) kaporin_gradient[j] = 2*( InnerProd(A[j], G_temp[i]) + A[j][i])
         *     kaporin_gradient = 2 * MatVec(A[0:j], G_temp[i]') + 2*A[i] simplified
         *  2) Need a kernel to compute A[P, :]*G_temp - TODO
         
         * Grab max_step_size UNIQUE positions from kaporian gradient
         *  - Need to write my own function. A binary array can be used to mark with locations have already been added to the pattern.
         *
         * Gather A[P, P], G[i, P], and -A[P, i]
         *  - Adapt the hypre_ParCSRMatrixExtractBExt function. Don't want to return a CSR matrix because we're looking for a dense matrix.
         *
         * Determine psi_{k} = G_temp[i]*A*G_temp[i]'
         *
         * Solve A[P, P]G[i, P]' = -A[P, i]
         *
         * Determine psi_{k+1} = G_temp[i]*A*G_temp[i]'
         */
         if(abs( psi_new - psi_old )/psi_old < kaporin_tol)
            break;

      }

      /* row_scale = 1/sqrt(A[i, i] -  abs( InnerProd(G_temp, A[i])) )
      *  G[i] = row_scale * G_temp  
      */

   }

   hypre_TFree(kaporin_gradient, HYPRE_MEMORY_HOST);
   hypre_TFree(G_temp, HYPRE_MEMORY_HOST);
   hypre_TFree(S_Pattern, HYPRE_MEMORY_HOST);

   return(hypre_error_flag);

}
