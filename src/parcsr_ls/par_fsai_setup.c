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
 * - A_sub:          A (nrows_needed + (n_row_needed+1))/2 - sized array to hold the lower triangular of the symmetric submatrix A[P, P]. 
 * - A_diag:         CSR Matrix diagonal of A
 * - marker:         A work array of length equal to the number of columns in A_diag specifying 
 *   which column indices should be added to sub_row and in what order (all values should be 
 *   set to -1 if they are not needed and and a number between 0:(ncols_needed-1) otherwise)
 * - nrows_needed:   Number of rows/columns A[P, P] needs.
 */
void
hypre_CSRMatrixExtractDenseMatrix(HYPRE_Vector *A_sub, HYPRE_CSRMatrix *A_diag, HYPRE_Int *marker, HYPRE_Int nrows_needed)
{
   HYPRE_Int      *A_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data     = hypre_CSRMatrixData(A_diag);
   HYPRE_Int      cc;         /* Local dense matrix column counter */ 
   HYPRE_Int      i, j;       /* Loop variables */

   HYPRE_Complex  *A_sub_data = hypre_SeqVectorData(A_sub);

   for(i = 0; i < hypre_CSRMatrixNumRows(A_diag); i++)
      if(marker[i] >= 0)
         for(j = A_i[i]; j < A_i[i+1]; j++)
            if((cc = marker[A_j[j]]) >= 0 && A_j[j] <= i)   /* Only gather lower triagonal since it's a symmetric matrix? */
               A_sub_data[i + cc*nrows_needed] = A_data[j];

   return;

}

/* Extract the dense sub-row from a matrix (A[i, P]) 
 * Parameters:
 * - A_diag:         CSR Matrix diagonal of A
 * - A_subrow:       The extracted sub-row of A[i, P]
 * - marker:         A work array of length equal to the number of rows in A_diag specifying 
 *   which column indices should be added to sub_row and in what order (all values should be 
 *   set to -1 if they are not needed and and a number between 0:(ncols_needed-1) otherwise)
 * - needed_row:     Which row of A we are extracting from
 * - ncols_needed:   Number of columns A[i, P] needs.
 */
void
hypre_ExtractDenseRowFromCSRMatrix(HYPRE_CSRMatrix *A_diag, HYPRE_Vector *A_subrow, HYPRE_Int *marker, HYPRE_Int needed_row, HYPRE_Int ncols_needed)
{
   HYPRE_Int *A_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int i, cc;

   HYPRE_Complex *sub_row_data = hypre_VectorData(A_subrow);

   for(i = A_i[needed_row]; i < A_i[needed_row+1]; i++)
      if((cc = marker[A_j[i]]) >= 0)
         sub_row_data[cc] = A_data[i];

   return;
}

/* Extract a subset of a row from a row (G[i, P]) 
 * - row:            Vector holding the elements of G_temp in the main function
 * - ncols:          Length of row
 * - marker:         A work array of length equal to row_length specifying which column indices 
 *   should be added to sub_row and in what order (all values should be set to -1 if they are 
 *   not needed and and a number between 0:(ncols_needed-1) otherwise)
 * - ncols_needed:   Number of columns G[i, P] needs.
 */
void
hypre_ExtractDenseRowFromRow(HYPRE_Vector *row, HYPRE_Vector *sub_row, HYPRE_Int *marker, HYPRE_Int ncols_needed)
{
 
   HYPRE_Int i, j;
   hypre_assert(ncols_needed <= hypre_VectorSize(row));   

   HYPRE_Complex *sub_row_data = hypre_VectorData(sub_row);
   HYPRE_Complex *row_data     = hypre_VectorData(row);
 
   for(i = 0; i < hypre_VectorSize(row); ++i)
      if((j = marker[i]) >= 0)   
         sub_row_data[j] = row_data[i];
      
   return;

}

/* Find the intersection between arrays x and y, put it in z 
 * XXX: I saw the function 'IntersectTwoArrays' in protos.h, but it doesn't do what I want
 */
void 
hypre_IntersectTwoVectors(HYPRE_Vector *x, HYPRE_Vector *y, HYPRE_Vector *z, HYPRE_Int x_size, HYPRE_Int y_size, HYPRE_Int *z_size)
{

   /* Since x, y, and z are all pre-initialized and allocated (and may have been used before), I want to specify 
   * exactly how many spots in x, y, and z refer to this current call, which is why the sizes are input arguments */
 
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex *z_data = hypre_VectorData(z);

   HYPRE_Int i, j;

   z_size = 0;

   for(i = 0; i < x_size; i++)
      for(j = 0; j < y_size; j++)
         if(x_data[i] == y_data[j])
         {
            z_data[(*z_size)++] = x_data[i];
            break;
         }

   return;
}

/* Finding the Kaporin Gradient
 * Input Arguments:
 *  - A_diag:                 CSR matrix diagonal of A.
 *  - kaporin_gradient:       Array holding the kaporin gradient. This will we modified.
 *  - kap_grad_nonzeros:      Array of the nonzero columns of kaporin_gradient. To be modified.
 *  - kap_grad_nnz:           Number of elements in kaporin_gradient. To be modified.
 *  - A_kg:                   To hold a subrow of A[i] to perform the inner product
 *  - G_kg:                   To hold a subrow of G_temp to perform the inner product
 *  - G_temp:                 Work array of G for row i
 *  - S_pattern:              Array of column indices of the nonzero elements of G_temp
 *  - S_Pattern_nnz:          Number of non-zero elements in G_temp
 *  - max_row_size:           To ensure we don't overfill the kaporin_gradient vector
 *  - row_num:                Which row of G we are working on
 *  - marker:                 Array of length equal to the number of rows in A. Assume to be all -1's when passed in.
 */
void hypre_FindKapGrad(hypre_CSRMatrix *A_diag, HYPRE_Vector *kaporin_gradient, HYPRE_Int *kap_grad_nnz, HYPRE_Vector *kap_grad_nonzeros, HYPRE_Vector *A_kg, HYPRE_Vector *G_kg, HYPRE_Vector *G_temp, HYPRE_Vector *S_Pattern, HYPRE_Int S_Pattern_nnz, HYPRE_Int max_row_size, HYPRE_Int row_num, HYPER_Int *marker)
{

   HYPRE_Int      *A_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int      i, j, k, count;
   HYPRE_Real     A_j_i;
   HYPRE_Complex  temp;

   hypre_SeqVectorSetConstantValues(A_kg, 0.0);
   hypre_SeqVectorSetConstantValues(G_kg, 0.0);
   hypre_SeqVectorSetConstantValues(kaporin_gradient, 0.0);
   hypre_SeqVectorSetConstantValues(kap_grad_nonzeros, 0.0);

   HYPRE_Complex *S_Pattern_data          = hypre_VectorData(S_Pattern);
   HYPRE_Complex *A_kg_data               = hypre_VectorData(A_kg);
   HYPRE_Complex *G_kg_data               = hypre_VectorData(G_kg);
   HYPRE_Complex *G_temp_data             = hypre_VectorData(G_temp);
   HYPRE_Complex *kap_grad_data           = hypre_VectorData(kaporin_gradient);
   HYPRE_Complex *kap_grad_nonzero_data   = hypre_VectorData(kap_grad_nonzeros);
   (*kap_grad_nnz)                        = 0;

   for(i = 0; i < row_num; ++i)
      marker[i] = -1;
   for(i = 0; i < S_pattern_nnz; i++)
      marker[S_Pattern[i]] = 1;

   for(i = 0; i < row_num-1; i++)
   {
      if(marker[i] == 1)
         continue;
      count = 0;
      A_j_i = 0;
      for(j = A_i[i]; j < A_i[i+1]; j++)  /* Intersection + Gather */
      {
         if(A_j[j] == row_num)
            A_j_i = A_data[j];
         for(k = 0; k < S_Pattern_nnz; k++)
            if(A_j[j] == S_Pattern_data[k])
            {
               A_kg_data[count] = A_data[j];
               G_kg_data[count] = G_temp_data[k];
               count++;  
               break;     
            }
      }
       
      temp = abs(2 * (hypre_SeqVectorInnerProd(A_kg, G_kg) + A_j_i));
      if(temp > 0)
      {
         kap_grad_data[kap_grad_nnz] = temp;
         kap_grad_nonzero_data[kap_grad_nnz] = i;
         kap_grad_nnz++;
         if(kap_grad_nnz == max_row_size)
            return;
      }
      
   }  

   return;

}

/*****************************************************************************
 * hypre_FSAISetup
 ******************************************************************************/

HYPRE_Int
hypre_FSAISetup( void               *fsai_vdata,
                      hypre_ParCSRMatrix *A  )
{
   MPI_Comm                comm              = hypre_ParCSRMatrixComm(A);
   hypre_ParFSAIData       *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   hypre_MemoryLocation    memory_location   = hypre_ParCSRMatrixMemoryLocation(A);

   /* Data structure variables */

   HYPRE_Real              kap_tolerance     = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps         = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size     = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               logging           = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Int               print_level       = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int               debug_flag;       = hypre_ParFSAIDataDebugFlag(fsai_data);

   /* Declare Local variables */

   HYPRE_Int               num_procs, my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   HYPRE_CSRMatrix         *A_diag;
   HYPRE_CSRMatrix         *G;                           /* Final FSAI factor for this block */
   HYPRE_Vector            *G_temp;                      /* A vector to hold a single row of G while it's being calculated */
   HYPRE_Vector            *A_sub;                       /* A vector to hold a the A[P, P] submatrix of A */
   HYPRE_Vector            *A_kg;                        /* Vector to hold A[i, P] for the kaporin gradient */
   HYPRE_Vector            *G_kg;                        /* Vector to hold A[i, P] for the kaporin gradient */
   HYPRE_Vector            *kaporin_gradient;            /* A vector to hold the Kaporin Gradient for each iteration of aFSAI */
   HYPRE_Vector            *kaporin_gradient_nonzeros;   /* A vector to hold the kaporin_gradient nonzero indices for each iteration of aFSAI */
   HYPRE_Vector            *S_Pattern;
   HYPRE_Int               *marker;                      /* An array that holds which values need to be gathered when finding a sub-row or submatrix */
   HYPRE_Real              old_psi, new_psi;             /* GAG' before and after the k-th interation of aFSAI */   
   HYPRE_Real              row_scale;                    /* The value to scale G_temp by before adding it to G */
   HYPRE_Int               num_rows;
   HYPRE_Int               num_cols;
   HYPRE_Int               max_row_size;
   HYPRE_Int               S_Pattern_nnz;
   HYPRE_Int               kap_grad_nnz;
   HYPRE_Int               i, j, k;       /* Loop variables */
   
   /* Setting local variables */

   A_diag                  = hypre_ParCSRMatrixDiag(A);
   num_rows                = hypre_CSRMatrixNumRows(A_diag);
   num_cols                = hypre_CSRMatrixNumCols(A_diag);
   max_row_size            = min(max_steps*max_step_size, num_rows-1);
                          
   /* Allocating local vector variables */

   G_temp                        = hypre_SeqVectorCreate(max_row_size);
   A_kg                          = hypre_SeqVectorCreate(max_row_size);
   G_kg                          = hypre_SeqVectorCreate(max_row_size);
   kaporin_gradient              = hypre_SeqVectorCreate(max_row_size);
   kaporin_gradient_nonzeros     = hypre_SeqVectorCreate(max_row_size);
   S_Pattern                     = hypre_SeqVectorCreate(max_row_size);
   A_sub                         = hypre_SeqVectorCreate(max_row_size*(max_row_size+1)/2);/* Since the submatrix will be symmetric, can we only store the lower diagonal? */
   marker                        = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);       /* For gather functions - don't want to reinitialize */

   /* Initializing local variables */

   hypre_SeqVectorInitialize(G_temp);
   hypre_SeqVectorInitialize(A_kg);
   hypre_SeqVectorInitialize(G_kg);
   hypre_SeqVectorInitialize(kaporin_gradient);
   hypre_SeqVectorInitialize(kaporin_gradient_nonzeros);
   hypre_SeqVectorInitialize(S_Pattern);
   hypre_SeqVectorInitialize(A_sub);
   for( i = 0; i < num_rows; i++ )
      marker[i] = -1;

   /* Setting data variables for vectors */
   HYPRE_Complex *G_temp_data             = hypre_VectorData(G_temp);
   HYPRE_Complex *A_kg                    = hypre_VectorData(A_kg);
   HYPRE_Complex *G_kg                    = hypre_VectorData(G_kg);
   HYPRE_Complex *kaporin_gradient_data   = hypre_VectorData(kaporin_gradient);
   HYPRE_Complex *kap_grad_nonzero_data   = hypre_VectorData(kaporin_gradient_nonzeros);
   HYPRE_Complex *S_Pattern_data          = hypre_VectorData(S_Pattern);
   HYPRE_Complex *A_sub_data              = hypre_VectorData(A_sub_data);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm  
   ***********************************************************************/

   for( i = 0; i < num_rows; i++ )    /* Cycle through each of the local rows */
   {
      S_Pattern_nnz           = 1;
      S_Pattern_data[0]       = i;
      G_temp_data[0]          = 1;
      
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


   return(hypre_error_flag);

}
