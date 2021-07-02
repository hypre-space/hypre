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

/* Extract A[P, P] into dense matrix 
 * Parameters:
 * - A_sub:          A (nrows_needed)^2 sized array to hold the submatrix A[P, P]. 
 * - A_diag:         CSR Matrix diagonal of A
 * - marker:         A work array of length equal to the number of rows in A_diag specifying 
 *   which column indices should be added to sub_row and in what order (all values should be 
 *   set to -1 if they are not needed and and a number between 0:(ncols_needed-1) otherwise)
 * - nrows_needed:   Number of rows/columns A[P, P] needs.
 */
void
hypre_CSRMatrixExtractDenseMatrix(HYPRE_Real *A_sub, HYPRE_CSRMatrix *A_diag, HYPRE_Int *marker, HYPRE_Int nrows_needed)
{
   HYPRE_Int *A_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int cc;        /* Local dense matrix column counter */ 
   HYPRE_Int i, j;      /* Loop variables */

   for(i = 0; i < hypre_CSRMatrixNumRows(A_diag); i++)
      if(marker[i] >= 0)
         for(j = A_i[i]; j < A_i[i+1]; j++)
            if((cc = marker[A_j[j]]) >= 0)
               A_sub[i + cc*nrows_needed] = A_data[j];

   return;

}

/* Extract the dense sub-row from a matrix (A[i, P]) 
 * Parameters:
 * - A_diag:         CSR Matrix diagonal of A
 * - marker:         A work array of length equal to the number of rows in A_diag specifying 
 *   which column indices should be added to sub_row and in what order (all values should be 
 *   set to -1 if they are not needed and and a number between 0:(ncols_needed-1) otherwise)
 * - needed_row:     Which row of A we are extracting from
 * - ncols_needed:   Number of columns A[i, P] needs.
 */
HYPRE_Vector
hypre_ExtractDenseRowFromCSRMatrix(HYPRE_CSRMatrix *A_diag, HYPRE_Int *marker, HYPRE_Int needed_row, HYPRE_Int ncols_needed)
{
   HYPRE_Int *A_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int i, cc;

   HYPRE_Vector *sub_row = hypre_VectorCreate(ncols_needed);
   hypre_VectorInitialize(sub_row);
   HYPRE_Complex *sub_row_data = hypre_VectorData(sub_row);

   for(i = A_i[needed_row]; i < A_i[needed_row+1]; i++)
      if((cc = marker[A_j[i]]) >= 0)
         sub_row_data[cc] = A_data[i];

   return sub_row;
}

/* Extract a subset of a row from a row (G[i, P]) 
 * - row:            Array holding the elements of G_temp in the main function
 * - ncols:          Length of row
 * - marker:         A work array of length equal to the number of rows in A_diag specifying 
 *   which column indices should be added to sub_row and in what order (all values should be 
 *   set to -1 if they are not needed and and a number between 0:(ncols_needed-1) otherwise)
 * - ncols_needed:   Number of columns G[i, P] needs.
 */
HYPRE_Vector
hypre_ExtractDenseRowFromRow(HYPRE_Real *row, HYPRE_Int nrows, HYPRE_Int *marker, HYPRE_Int ncols_needed)
{
 
   HYPRE_Int i;
   hypre_assert(ncols_needed <= ncols);   

   HYPRE_Vector *sub_row = hypre_VectorCreate(ncols_needed);
   hypre_VectorInitialize(sub_row);
   HYPRE_Complex *sub_row_data = hypre_VectorData(sub_row);
 
   for(i = 0; i < ncols_needed; ++)   
      sub_row_data[i] = row[cols_needed[i]];
      
   return sub_row;

}

/* Find the intersection between vectors x and y, put it in z 
 * XXX: I saw the function 'IntersectTwoArrays' in protos.h, but it doesn't do what I want
 */
HYPRE_Vector hypre_IntersectTwoVectors(HYPRE_Vector *x, HYPRE_Vector *y)
{
 
   HYPRE_Int i, j;

   HYPRE_Int      x_size = HYPRE_VectorSize(x);
   HYPRE_Int      y_size = HYPRE_VectorSize(y);
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);

   HYPRE_Complex *z_data = CTAlloc(HYPRE_Complex, max(x_size, y_size), HYPRE_MEMORY_HOST);
   HYPRE_Int      z_size = 0;

   for(i = 0; i < x_size; i++)
      for(j = 0; j < y_size; j++)
         if(x_data[i] == y_data[j])
         {
            z_data[(*z_size)++] = x_data[i];
            break;
         }

   HYPRE_Vector *z = hypre_VectorCreate(z_size);
   hypre_VectorInitialize(z);
   HYPRE_VectorData(z) = z_data;                /* Unsure if this will actually work */

   return z;                                    /* Can't create this HYPRE_Vector outside of function because I don't know the size */

}

/* Finding the Kaporin Gradient
 * Input Arguments:
 *  - kaporin_gradient:       Array holding the kaporin gradient. This will we modified.
 *  - KapGrad_Nonzero_Cols:   Array of the nonzero columns of kaporin_gradient (The intersection of S_Pattern and the nonzero columns of A(i,:) ). To be modified.
 *  - KapGrad_nnz:            Number of elements in KapGrad_Nonzero_Cols. To be modified.
 *  - A_diag:                 CSR matrix diagonal of A.
 *  - row_num:                Which row of G are we computing ('i' in main function)
 *  - G_temp:                 Work array of G for row i
 *  - row_length:             Length of G_temp
 *  - S_pattern:              Array of column indices of the nonzero elements of G_temp
 *  - S_Pattern_nnz:          Number of non-zero elements in G_temp
 */
void hypre_FindKapGrad(HYPRE_Real *kaporin_gradient, HYPRE_Int *KapGrad_Nonzero_Cols, Hypre_Int *KapGrad_nnz, HYPRE_CSRMatrix *A_diag, HYPRE_Int row_num, HYPRE_Real *G_temp, HYPRE_Int row_length, HYPRE_Int *S_Pattern, HYPRE_Int S_Pattern_nnz)
{

   HYPRE_Int  *A_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int  *A_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Real *A_i_sub  = hypre_CTAlloc(HYPRE_Int, A_i[row_num+1] - A_i[row_num], HYPRE_MEMORY_HOST);
   HYPRE_Int  i, j;
   HYPRE_Int count;
   (*KapGrad_nnz) = 0;

   HYPRE_Vector *G_sub  = hypre_VectorCreate(S_Pattern_nnz);
   hypre_VectorInitialize(G_sub);
   hypre_ExtractDenseRowFromRow(G_sub, G_temp, row_length, S_Pattern, S_Pattern_nnz);

   HYPRE_Vector *A_sub  = hypre_VectorCreate(A_i[row_num+1] - A_i[row_num]);
   hypre_VectorInitialize(A_sub);
   HYPRE_Int *marker = CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(A_diag), HYPRE_MEMORY_HOST);
   for(i = 0; i < hypre_CSRMatrixNumRows(A_diag); i++)
      marker[i] = -1;

   for(i = 0; i < num_row-1; i++)
   {
      count = 0;
      for(j = A_i[i]; j < A_i[i+1]; j++)
         marker[A_j[i]] = count++;
 
      hypre_ExtractDenseRowFromCSRMatrix(A_sub, A_diag, marker, i, HYPRE_Int *needed_cols, count);
   }   

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
