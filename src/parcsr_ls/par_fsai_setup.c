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
hypre_CSRMatrixExtractDenseMatrix(HYPRE_Real *A_sub, HYPRE_CSRMatrix *A_diag, HYPRE_Int *marker, HYPRE_Int *needed_rows, HYPRE_Int nrows_needed)
{
   HYPRE_Int *A_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int rr, cc;    /* Local dense matrix row and column counter */ 
   HYPRE_Int i, j;      /* Loop variables */
   HYPRE_Int count = 0;

   hypre_qsort0(rows_needed, 0, nrows_needed-1);   /* the rows needed to be in order from least to greatest so matrix functions can be used */

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
hypre_ExtractDenseRowFromCSRMatrix(HYPRE_Real *sub_row, HYPRE_CSRMatrix *A_diag, HYPRE_Int *marker, HYPRE_Int needed_row, HYPRE_Int *needed_cols, HYPRE_Int ncols_needed)
{
   HYPRE_Int *A_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int i, cc;
   HYPRE_Int count = 0;

   hypre_qsort0(cols_needed, 0, ncols_needed-1);   /* the columns needed to be in order from least to greatest so matrix functions can be used */

   for(i = 0; i < ncols_needed; i++)
      marker[needed_cols[i]] = count++;

   for(i = A_i[needed_row]; i < A_i[needed_row+1]; i++)
      if((cc = marker[A_j[i]]) >= 0)
         sub_row[cc] = A_data[i];

   for(i = 0; i < ncols_needed; i++)
      marker[needed_cols[i]] = -1;

   return;
}

/* Extract a subset of a row from a row (G[i, P]) */
void
hypre_ExtractDenseRowFromRow(HYPRE_Real *sub_row, HYPRE_Real *row, HYPRE_Int ncols, HYPRE_Int *cols_needed, HYPRE_Int ncols_needed)
{
 
   HYPRE_Int i;
   hypre_assert(ncols_needed <= ncols);   

   hypre_qsort0(cols_needed, 0, ncols_needed-1);   /* the columns needed to be in order from least to greatest so matrix functions can be used */

   for(i = 0; i < ncols_needed; ++)   
      sub_row[i] = row[cols_needed[i]];
      
   return;

}

/* Find the intersection between arrays x and y, put it in z 
 * XXX: I saw the function 'IntersectTwoArrays in protos.h, but it doesn't do what I want
 */
/*void hypre_IntersectTwoArrays2(HYPRE_Int *x, HYPRE_Int x_len, HYPRE_Int *y, HYPRE_Int *y_len, HYPRE_Int *z, HYPRE_Int *z_len)
{
   
   HYPRE_Int i, j;

   for(i = 0; i < x_len; i++)
      for(j = 0; j < y_len; j++)
         if(x[i] == y[j])
         {
            z[(*z_len)++] = x[i];
            break;
         }

}*/

/* Finds the inner product between x and y, put it in IP */
void hypre_InnerProductTwoArrays(HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int num_elems, HYPRE_Real *IP)
{
   HYPRE_Int i;
   (*IP) = 0.0;

   for(i = 0; i < num_elems; i++)
      (*IP) += x[i]*y[i];
   
   return;
}

/* Finding the Kaporin Gradient
 * Input Arguments:
 *  - kaporin_gradient:       Array holding the kaporin gradient. This will we modified.
 *  - KapGrad_Nonzero_Cols:   Array of the nonzero columns of kaporin_gradient (The intersection of S_Pattern and the nonzero columns of A(i,:) ). To be modified.
 *  - KapGrad_nnz:            Number of elements in KapGrad_Nonzero_Cols. To be modified.
 *  - A_diag:                 CSR matrix diagonal of A.
 *  - row_num:                Which row of G are we computing ('i' in main function)
 *  - G_temp:                 Work array of G for row i
 *  - S_pattern:              Array of column indices of the nonzero elements of G_temp
 *  - S_Pattern_nnz:          Number of non-zero elements in G_temp
 */
void hypre_FindKapGrad(HYPRE_Real *kaporin_gradient, HYPRE_Int *KapGrad_Nonzero_Cols, Hypre_Int *KapGrad_nnz, HYPRE_CSRMatrix *A_diag, HYPRE_Int row_num, HYPRE_Real *G_temp, HYPRE_Int *S_Pattern, HYPRE_Int S_Pattern_nnz)
{

   HYPRE_Int  *A_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int  *A_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Real *A_i_sub  = hypre_CTAlloc(HYPRE_Int, A_i[row_num+1] - A_i[row_num], HYPRE_MEMORY_HOST);
   HYPRE_Int  i;
   (*KapGrad_nnz) = 0;

   for(i = A_i[row_num]; i < A_i[row_num+1]; i++)
      A_i_sub[count++] = A_j[i];

   /* TODO
   for( i = 0; i < (*KapGrad_nnz); i++ )
   {
      * kaporin_gradient[i] = 2*( InnerProd(A[i], G_temp[row_num]) + A[i][row_num]) 
      *  How I did this in Matlab is I took the intersection of the nonzero columns of A[i] and G_temp[row_num]
      *  before doing the inner product between the two subsets. However, finding this intersection each iteration
      *  is expensive and won't provide any benefit over multiplying by a bunch of zeros I think.
      *  Really all this is is a matvec between G_temp and A(1:i, :). Is there a MatVec function that is efficient?
   }
   */

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
   HYPRE_Int               *row_partition;
   HYPRE_Int               *markers;
   HYPRE_Real              old_psi, new_psi;
   HYPRE_Real              row_scale;
   HYPRE_Int               num_rows;
   HYPRE_Int               max_row_size;
   HYPRE_Int               i, j, k;       /* Loop variables */
   HYPRE_Int               *S_Pattern;
   
   /* Setting local variables */

   A_diag                  = hypre_ParCSRMatrixDiag(A);
   num_rows                = hypre_CSRMatrixNumRows(A_diag);
   num_cols                = hypre_CSRMatrixNumCols(A_diag);
                          
   /* Allocating local variables */
   
   max_row_size            = min(max_steps*max_step_size, num_rows-1);
   kaporin_gradient        = hypre_CTAlloc(HYPRE_Real, max_row_size, HYPRE_MEMORY_HOST);
   G_temp                  = hypre_CTAlloc(HYPRE_Real, max_row_size, HYPRE_MEMORY_HOST);
   S_Pattern               = hypre_CTAlloc(HYPRE_Int, max_row_size, HYPRE_MEMORY_HOST);
   markers                 = hypre_CTAlloc(HYPRE_Int, num_cols, HYPRE_MEMORY_HOST);       /* For gather functions - don't want to reinitialize */
   for( i = 0; i < num_cols; i++ )
      markers[i] = -1;


   /**********************************************************************
   * Start of Adaptive FSAI algorithm  
   ***********************************************************************/

   for( i = 0; i < num_rows; i++ )    /* Cycle through each of the local rows */
   {
      
      memset(S_Pattern, 0, max_row_size*sizeof(HYPRE_Int));

      for( k = 0; k < max_steps; k++ ) /* Cycle through each iteration for that row */
      {
         /*
         * Compute Kaporin Gradient
         *  1) kaporin_gradient[j] = 2*( InnerProd(A[j], G_temp[i]) + A[j][i])
         *     kaporin_gradient = 2 * MatVec(A[0:j], G_temp[i]') + 2*A[i] simplified
         *  2) Need a kernel to compute A[P, :]*G_temp - TODO
         */      
         /* Steps:
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
   hypre_TFree(markers, HYPRE_MEMORY_HOST);

   return(hypre_error_flag);

}
