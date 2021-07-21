/******************************************************************************
 *  Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 *  HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

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
HYPRE_Int
hypre_CSRMatrixExtractDenseMatrix( hypre_CSRMatrix *A_diag,
                                   hypre_Vector    *A_sub,
                                   HYPRE_Int       *marker,
                                   HYPRE_Int       nrows_needed)
{
   HYPRE_Int      *A_i        = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j        = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data     = hypre_CSRMatrixData(A_diag);
   HYPRE_Int      cc;         /* Local dense matrix column counter */
   HYPRE_Int      i, j;       /* Loop variables */

   HYPRE_Complex  *A_sub_data = hypre_VectorData(A_sub);

   for(i = 0; i < hypre_CSRMatrixNumRows(A_diag); i++)
      if(marker[i] >= 0)
         for(j = A_i[i]; j < A_i[i+1]; j++)
            if((cc = marker[A_j[j]]) >= 0 && A_j[j] <= i)
               A_sub_data[i + cc*nrows_needed] = A_data[j];

   return hypre_error_flag;

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
HYPRE_Int
hypre_ExtractDenseRowFromCSRMatrix( hypre_CSRMatrix *A_diag,
                                    hypre_Vector    *A_subrow,
                                    HYPRE_Int       *marker,
                                    HYPRE_Int       ncols_needed,
                                    HYPRE_Int       needed_row )
{
   HYPRE_Int *A_i       = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_j       = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int i, cc;

   HYPRE_Complex *sub_row_data = hypre_VectorData(A_subrow);

   for(i = A_i[needed_row]; i < A_i[needed_row+1]; i++)
      if((cc = marker[A_j[i]]) >= 0)
         sub_row_data[cc] = A_data[i];

   return hypre_error_flag;
}

/* Finding the Kaporin Gradient
 * Input Arguments:
 *  - A_diag:                 CSR matrix diagonal of A.
 *  - kaporin_gradient:       Array holding the kaporin gradient. This will we modified.
 *  - kap_grad_nonzeros:      Array of the nonzero columns of kaporin_gradient. To be modified.
 *  - A_kg:                   To hold a subrow of A[i] to perform the inner product
 *  - G_kg:                   To hold a subrow of G_temp to perform the inner product
 *  - G_temp:                 Work array of G for row i
 *  - S_pattern:              Array of column indices of the nonzero elements of G_temp
 *  - max_row_size:           To ensure we don't overfill the kaporin_gradient vector
 *  - row_num:                Which row of G we are working on
 *  - marker:                 Array of length equal to the number of rows in A.
 */
HYPRE_Int
hypre_FindKapGrad( hypre_CSRMatrix  *A_diag,
                   hypre_Vector     *kaporin_gradient,
                   hypre_Vector     *kap_grad_nonzeros,
                   hypre_Vector     *A_kg,
                   hypre_Vector     *G_kg,
                   hypre_Vector     *G_temp,
                   hypre_Vector     *S_Pattern,
                   HYPRE_Int        max_row_size,
                   HYPRE_Int        row_num,
                   HYPRE_Int        *marker )
{

   HYPRE_Int      *A_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int      i, j, k, count;
   HYPRE_Int      count2    = 0;
   HYPRE_Real     A_j_i;
   HYPRE_Complex  temp;

   hypre_VectorSize(A_kg)                 = max_row_size;
   hypre_VectorSize(G_kg)                 = max_row_size;
   hypre_VectorSize(kaporin_gradient)     = max_row_size;
   hypre_VectorSize(kap_grad_nonzeros)    = max_row_size;

   HYPRE_Complex *S_Pattern_data          = hypre_VectorData(S_Pattern);
   HYPRE_Complex *A_kg_data               = hypre_VectorData(A_kg);
   HYPRE_Complex *G_kg_data               = hypre_VectorData(G_kg);
   HYPRE_Complex *G_temp_data             = hypre_VectorData(G_temp);
   HYPRE_Complex *kap_grad_data           = hypre_VectorData(kaporin_gradient);
   HYPRE_Complex *kap_grad_nonzero_data   = hypre_VectorData(kap_grad_nonzeros);

   for(i = 0; i < row_num; ++i)
      marker[i] = 0;
   for(i = 0; i < hypre_VectorSize(S_Pattern); i++)
      marker[(HYPRE_Int)S_Pattern_data[i]] = 1;
   
   for(i = 0; i < row_num-1; i++)
   {
    
      hypre_VectorSize(A_kg)                 = max_row_size;
      hypre_VectorSize(G_kg)                 = max_row_size;

      if(marker[i]) /* If this spot is already part of S_Pattern, we don't need to compute it because it doesn't need to be re-added */
         continue;

      A_j_i = 0;
      count = 0;
      temp  = 0;

      for(j = A_i[i]; j < A_i[i+1]; j++)  /* Intersection + Gather */
      {
         if(A_j[j] == row_num)
            A_j_i = A_data[j];
         for(k = 0; k < hypre_VectorSize(S_Pattern); k++)
            if(S_Pattern_data[k] == A_j[j])
            {
               G_kg_data[count] = G_temp_data[(HYPRE_Int)S_Pattern_data[k]];
               A_kg_data[count] = A_data[j];
               count++;
            }
            if(A_j[j] == row_num)
               temp = A_data[j];
      }

      hypre_VectorSize(A_kg) = count;
      hypre_VectorSize(G_kg) = count;

      temp += hypre_abs(2 * (hypre_SeqVectorInnerProd(A_kg, G_kg) + A_j_i));
      if(temp > 0)
      {
         kap_grad_data[count2] = temp;
         kap_grad_nonzero_data[count2] = i;
         count2++;
         if(count2 == max_row_size)
            break;
      }
      

   }

   hypre_VectorSize(kaporin_gradient)  = count2;
   hypre_VectorSize(kap_grad_nonzeros) = count2;

   for(i = 0; i < hypre_VectorSize(S_Pattern); i++)     /* Reset marker array for future use */
      marker[(HYPRE_Int)S_Pattern_data[i]] = 0;

   return hypre_error_flag;

}

void
hypre_swap2C( HYPRE_Complex  *v,
              HYPRE_Complex  *w,
              HYPRE_Int      i,
              HYPRE_Int      j )
{
   HYPRE_Complex  temp;
   HYPRE_Complex  temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/* Quick Sort (largest to smallest) for complex arrays - hypre/utilities/qsort.c did not have what I wanted */
/* sort on w (HYPRE_Complex), move v */

void
hypre_qsort2C( HYPRE_Complex  *v,
               HYPRE_Complex  *w,
               HYPRE_Int      left,
               HYPRE_Int      right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap2C( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (w[i] > w[left])
      {
         hypre_swap2C(v, w, ++last, i);
      }
   }
   hypre_swap2C(v, w, left, last);
   hypre_qsort2C(v, w, left, last-1);
   hypre_qsort2C(v, w, last+1, right);
}

/* Take the largest elements from the kaporin gradient and add their locations to S_Pattern */
HYPRE_Int
hypre_AddToPattern( hypre_Vector *kaporin_gradient,
                    hypre_Vector *kap_grad_nonzeros,
                    hypre_Vector *S_Pattern,
                    HYPRE_Int    max_step_size )
{

   HYPRE_Complex *kap_grad_data           = hypre_VectorData(kaporin_gradient);
   HYPRE_Complex *kap_grad_nonzero_data   = hypre_VectorData(kap_grad_nonzeros);
   HYPRE_Complex *S_Pattern_data          = hypre_VectorData(S_Pattern);

   HYPRE_Int     i, count;

   hypre_qsort2C(kap_grad_data, kap_grad_nonzero_data, 0, hypre_VectorSize(kaporin_gradient)-1);

   count = hypre_VectorSize(S_Pattern);
   hypre_VectorSize(S_Pattern) = count + hypre_min(hypre_VectorSize(kaporin_gradient), max_step_size);

   for(i = 0; i < hypre_min(hypre_VectorSize(kaporin_gradient), max_step_size); i++)
      S_Pattern_data[count++] = kap_grad_nonzero_data[i];

   return hypre_error_flag;

}

/*****************************************************************************
 * hypre_FSAISetup
 ******************************************************************************/

HYPRE_Int
hypre_FSAISetup( void               *fsai_vdata,
                 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *f,
                 hypre_ParVector    *u )
{

   MPI_Comm                comm              = hypre_ParCSRMatrixComm(A);
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   hypre_MemoryLocation    memory_location   = hypre_ParCSRMatrixMemoryLocation(A);

   /* Data structure variables */

   hypre_ParCSRMatrix     *G;
   HYPRE_Real              kap_tolerance     = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps         = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size     = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               logging           = hypre_ParFSAIDataLogging(fsai_data);
   HYPRE_Int               print_level       = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int               debug_flag        = hypre_ParFSAIDataDebugFlag(fsai_data);

   /* Declare Local variables */

   HYPRE_Int               num_procs, my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   hypre_Vector           *G_temp;            /* A vector to hold a single row of G while it's being calculated */
   hypre_Vector           *A_sub;             /* A vector to hold a the A[P, P] submatrix of A */
   hypre_Vector           *A_subrow;          /* Vector to hold A[i, P] for the kaporin gradient */
   hypre_Vector           *sol_vec;           /* Vector to hold solution of hypre_dpetrs */
   hypre_Vector           *G_subrow;          /* Vector to hold A[i, P] for the kaporin gradient */
   hypre_Vector           *kaporin_gradient;  /* A vector to hold the Kaporin Gradient for each iteration of aFSAI */
   hypre_Vector           *kap_grad_nonzeros; /* A vector to hold the kaporin_gradient nonzero indices for each iteration of aFSAI */
   hypre_Vector           *S_Pattern;
   HYPRE_Int              *marker;            /* An array that holds which values need to be gathered when finding a sub-row or submatrix */
   HYPRE_Real              old_psi, new_psi;  /* GAG' before and after the k-th interation of aFSAI */
   HYPRE_Real              row_scale;         /* The value to scale G_temp by before adding it to G */
   HYPRE_Int               info;
   HYPRE_Int               i, j, k;           /* Loop variables */

   char uplo = 'L';

   /* Create and initialize G_mat and work vectors */
   G = hypre_ParCSRMatrixCreate (comm,
       hypre_ParCSRMatrixGlobalNumRows(A),
       hypre_ParCSRMatrixGlobalNumCols(A),
       hypre_ParCSRMatrixRowStarts(A),
       hypre_ParCSRMatrixColStarts(A),
       0,
       hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A))*max_steps*max_step_size,
       0);
   hypre_ParCSRMatrixInitialize(G);
   hypre_ParFSAIDataGmat(fsai_data) = G;

   hypre_ParVector         *residual = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));      
   hypre_ParVector         *x_work   = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));      
   hypre_ParVector         *r_work   = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));    
   hypre_ParVector         *z_work   = hypre_ParVectorCreate(comm, hypre_ParCSRMatrixGlobalNumRows(A), hypre_ParCSRMatrixRowStarts(A));      

   hypre_ParVectorInitialize(residual);
   hypre_ParVectorInitialize(x_work);
   hypre_ParVectorInitialize(r_work);
   hypre_ParVectorInitialize(z_work);
   
   hypre_ParFSAIDataResidual(fsai_data)      = residual;
   hypre_ParFSAIDataXWork(fsai_data)         = x_work;
   hypre_ParFSAIDataRWork(fsai_data)         = r_work;
   hypre_ParFSAIDataZWork(fsai_data)         = z_work;

   /* Setting local variables */

   hypre_CSRMatrix        *A_diag        = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Int              *A_j           = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real             *A_data        = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix        *G_diag        = hypre_ParCSRMatrixDiag(G);
   HYPRE_Int              *G_i           = hypre_CSRMatrixI(G_diag);
   HYPRE_Int              *G_j           = hypre_CSRMatrixJ(G_diag);
   HYPRE_Real             *G_data        = hypre_CSRMatrixData(G_diag);

   HYPRE_Int               num_rows      = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               max_row_size  = hypre_min(max_steps*max_step_size, num_rows-1);

   /* Allocating local vector variables */

   G_temp            = hypre_SeqVectorCreate(max_row_size);
   A_subrow          = hypre_SeqVectorCreate(max_row_size);
   sol_vec           = hypre_SeqVectorCreate(max_row_size);
   G_subrow          = hypre_SeqVectorCreate(max_row_size);
   kaporin_gradient  = hypre_SeqVectorCreate(max_row_size);
   kap_grad_nonzeros = hypre_SeqVectorCreate(max_row_size);
   S_Pattern         = hypre_SeqVectorCreate(max_row_size);
   A_sub             = hypre_SeqVectorCreate(max_row_size*max_row_size);
   marker            = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);       /* For gather functions - don't want to reinitialize */

   /* Initializing local variables */

   hypre_SeqVectorInitialize(G_temp);
   hypre_SeqVectorInitialize(A_subrow);
   hypre_SeqVectorInitialize(sol_vec);
   hypre_SeqVectorInitialize(G_subrow);
   hypre_SeqVectorInitialize(kaporin_gradient);
   hypre_SeqVectorInitialize(kap_grad_nonzeros);
   hypre_SeqVectorInitialize(S_Pattern);
   hypre_SeqVectorInitialize(A_sub);
   for( i = 0; i < num_rows; i++ )
      marker[i] = -1;

   /* Setting data variables for vectors */
   HYPRE_Complex *G_temp_data             = hypre_VectorData(G_temp);
   HYPRE_Complex *A_subrow_data           = hypre_VectorData(A_subrow);
   HYPRE_Complex *G_subrow_data           = hypre_VectorData(G_subrow);
   HYPRE_Complex *kaporin_gradient_data   = hypre_VectorData(kaporin_gradient);
   HYPRE_Complex *kap_grad_nonzero_data   = hypre_VectorData(kap_grad_nonzeros);
   HYPRE_Complex *S_Pattern_data          = hypre_VectorData(S_Pattern);
   HYPRE_Complex *A_sub_data              = hypre_VectorData(A_sub);
   HYPRE_Complex *sol_vec_data            = hypre_VectorData(sol_vec);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   for( i = 0; i < num_rows; i++ )    /* Cycle through each of the local rows */
   {
      hypre_VectorSize(S_Pattern) = 0;

      /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
      old_psi = A_data[A_i[i]];

      for( k = 0; k < max_steps; k++ ) /* Cycle through each iteration for that row */
      {

         /* Compute Kaporin Gradient */
         hypre_FindKapGrad(A_diag, kaporin_gradient, kap_grad_nonzeros, A_subrow, G_subrow, G_temp, S_Pattern, max_row_size, i, marker);
 
         if(hypre_VectorSize(kaporin_gradient) == 0)
            break;     

         /* Find max_step_size largest values of the kaporin gradient, find their column indices, and add it to S_Pattern */
         hypre_AddToPattern(kaporin_gradient, kap_grad_nonzeros, S_Pattern, max_step_size);

         /* Gather A[P, P] and -A[P, i] */
         for(j = 0; j < hypre_VectorSize(S_Pattern); j++)
            marker[(HYPRE_Int)S_Pattern_data[j]] = j;

         hypre_VectorSize(A_sub)    = hypre_VectorSize(S_Pattern) * hypre_VectorSize(S_Pattern);
         hypre_VectorSize(A_subrow) = hypre_VectorSize(S_Pattern);
         hypre_VectorSize(G_temp)   = hypre_VectorSize(S_Pattern);
         hypre_VectorSize(sol_vec)  = hypre_VectorSize(S_Pattern);

         hypre_CSRMatrixExtractDenseMatrix(A_diag, A_sub, marker, hypre_VectorSize(S_Pattern));         /* A[P, P] */
         hypre_ExtractDenseRowFromCSRMatrix(A_diag, A_subrow, marker, hypre_VectorSize(S_Pattern), i);  /* A[P, i] */
         hypre_SeqVectorScale(-1, A_subrow);                                                            /* -A[P, i] */

         for(j = 0; j < hypre_VectorSize(A_subrow); j++)
            sol_vec_data[j] = A_subrow_data[j];

         /* Solve A[P, P]G[i, P]' = -A[P, i] */
         hypre_dpotrf(&uplo, &hypre_VectorSize(S_Pattern), A_sub_data, &hypre_VectorSize(S_Pattern), &info);
         
         hypre_dpotrs(&uplo, &hypre_VectorSize(S_Pattern), &hypre_VectorSize(S_Pattern), A_sub_data, &hypre_VectorSize(S_Pattern), sol_vec_data, &hypre_VectorSize(S_Pattern), &info); /* A_subrow becomes the solution vector... */

         for(j = 0; j < hypre_VectorSize(sol_vec); j++)
            G_temp_data[j] = sol_vec_data[j];

         /* Determine psi_{k+1} = G_temp[i]*A*G_temp[i]' */
         hypre_SeqVectorScale(-1, A_subrow);                         /* A[P, i] */
         new_psi = hypre_SeqVectorInnerProd(G_temp, A_subrow) + A_data[A_i[i]];

         if(hypre_abs( new_psi - old_psi )/old_psi < kap_tolerance){
            break;
         }

         old_psi = new_psi;

      }

      /*XXX: Memory corruption seems to be here, but G_temp and A_subrow are of the same size, and less than max_row_size */
      /* Calculate value to scale G_temp */
      row_scale = 1/(sqrt(A_data[A_i[i]] - hypre_abs(hypre_SeqVectorInnerProd(G_temp, A_subrow))));

      /* Re-add diagonal component of G_temp before scaling */
      hypre_VectorSize(G_temp) += 1;
      hypre_VectorSize(S_Pattern) += 1;
      G_temp_data[hypre_VectorSize(G_temp)-1] = 1;
      S_Pattern_data[hypre_VectorSize(S_Pattern)-1] = i;

      hypre_SeqVectorScale(row_scale, G_temp);

      /* Pass values of G_temp into G */
      for(k = 0; k < hypre_VectorSize(S_Pattern); k++)
      {
         j           = k + G_i[i-1];
         G_j[j]      = S_Pattern_data[k];
         G_data[j]   = G_temp_data[k];
      }
      G_i[i] = G_i[i-1] + hypre_VectorSize(S_Pattern);
   
   }

   /* Compute G^T */
   hypre_ParCSRMatrixTranspose(G, &hypre_ParFSAIDataGTmat(fsai_data), 1);

   hypre_SeqVectorDestroy(G_temp);
   hypre_SeqVectorDestroy(A_subrow);
   hypre_SeqVectorDestroy(sol_vec);
   hypre_SeqVectorDestroy(G_subrow);
   hypre_SeqVectorDestroy(kaporin_gradient);
   hypre_SeqVectorDestroy(kap_grad_nonzeros);
   hypre_SeqVectorDestroy(S_Pattern);
   hypre_SeqVectorDestroy(A_sub);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);

   return(hypre_error_flag);

}
