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

/*****************************************************************************
 *
 * Routine for driving the setup phase of FSAI
 *
 ******************************************************************************/

/* Extract A[P, P] into dense matrix
 * Parameters:
 * - A_diag:         CSR Matrix diagonal of A
 * - A_sub:          A S_nnz^2 - sized array to hold the lower triangular of the symmetric submatrix A[P, P].
 * - S_Pattern:      A S_nnz - sized array to hold the wanted rows/cols.
 * - marker:         A work array of length equal to the number of columns in A_diag - all values should be -1
 */
HYPRE_Int
hypre_CSRMatrixExtractDenseMatrix( hypre_CSRMatrix *A_diag,
                                   hypre_Vector    *A_sub,
                                   HYPRE_Int       *S_Pattern,
                                   HYPRE_Int        S_nnz,
                                   HYPRE_Int       *marker )
{
   HYPRE_Int      *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j              = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data           = hypre_CSRMatrixData(A_diag);
   HYPRE_Complex  *A_sub_data       = hypre_VectorData(A_sub);
   HYPRE_Int      cc;               /* Local dense matrix column counter */
   HYPRE_Int      i;                /* Loop variables */
   HYPRE_Int      j;

   for(i = 0; i < S_nnz; i++)
      for(j = A_i[S_Pattern[i]]; j < A_i[S_Pattern[i]+1]; j++)
      {
         if(A_j[j] > S_Pattern[i])
            break;

         if((cc = marker[A_j[j]]) >= 0)
            A_sub_data[cc*S_nnz + i] = A_data[j];

      }

   return hypre_error_flag;

}

/* Extract the dense sub-row from a matrix (A[i, P])
 * Parameters:
 * - A_diag:         CSR Matrix diagonal of A
 * - A_subrow:       The extracted sub-row of A[i, P]
 * - marker:         A work array of length equal to the number of row in A_diag. Assumed to be set to all -1
 * - row_num:        which row we want
 */
HYPRE_Int
hypre_ExtractDenseRowFromCSRMatrix( hypre_CSRMatrix *A_diag,
                                    hypre_Vector    *A_subrow,
                                    HYPRE_Int       *marker,
                                    HYPRE_Int        row_num )
{
   HYPRE_Int      *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j              = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data           = hypre_CSRMatrixData(A_diag);
   HYPRE_Complex  *sub_row_data     = hypre_VectorData(A_subrow);
   HYPRE_Int i, cc;

   for(i = A_i[row_num]; i < A_i[row_num+1]; i++)
      if((cc = marker[A_j[i]]) >= 0)
         sub_row_data[cc] = A_data[i];

   return hypre_error_flag;
}

/* Finding the Kaporin Gradient
 * Input Arguments:
 *  - A_diag:                 CSR matrix diagonal of A.
 *  - kaporin_gradient:       Array holding the kaporin gradient. This will we modified.
 *  - kap_grad_nonzeros:      Array of the nonzero columns of kaporin_gradient. To be modified.
 *  - G_temp:                 Work array of G for row i
 *  - S_pattern:              Array of column indices of the nonzero elements of G_temp
 *  - S_nnz:                  Number of column indices of the nonzero elements of G_temp
 *  - max_row_size:           To ensure we don't overfill the kaporin_gradient vector
 *  - row_num:                Which row of G we are working on
 *  - marker:                 Array of length equal to the number of rows in A - assumed to all be set to -1.
 */
HYPRE_Int
hypre_FindKapGrad( hypre_CSRMatrix  *A_diag,
                   hypre_Vector     *kaporin_gradient,
                   HYPRE_Int        *kap_grad_nonzeros,
                   hypre_Vector     *G_temp,
                   HYPRE_Int        *S_Pattern,
                   HYPRE_Int        S_nnz,
                   HYPRE_Int        max_row_size,
                   HYPRE_Int        row_num,
                   HYPRE_Int        *S_marker,
                   HYPRE_Int        *kg_marker )
{

   HYPRE_Int      *A_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j      = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int      i, j, count;

   HYPRE_Complex *G_temp_data             = hypre_VectorData(G_temp);
   HYPRE_Complex *kap_grad_data           = hypre_VectorData(kaporin_gradient);

   count = 0;

   // A[0:(row_num-2), P]*G_temp[i, P]
   for(i = 0; i < S_nnz; i++)
      for(j = A_i[S_Pattern[i]]; j < A_i[S_Pattern[i]+1]; j++)
      {
         // Position can't be added to the Kaporin Gradient
         if(S_marker[A_j[j]] > -1)
            continue;
         else if(A_j[j] >= row_num)
            break;

         // Position can be added - Check if it already has been
         if(kg_marker[A_j[j]] > -1)
            kap_grad_data[kg_marker[A_j[j]]] += G_temp_data[i]*A_data[j];
         else if(count < max_row_size)
         {
            kg_marker[A_j[j]] = count;
            kap_grad_data[count] = G_temp_data[i]*A_data[j];
            kap_grad_nonzeros[count] = A_j[j];
            count++;
         }
      }

   // + A[0:(row_num-2), row_num]
   for(j = A_i[row_num]+1; j < A_i[row_num+1]; j++){
      // Position cna't be added to the Kaporin Gradient
      if(S_marker[A_j[j]] > -1)
         continue;
      else if(A_j[j] >= row_num)
         break;

      // Position can be added - Check if it already has been
      if(kg_marker[A_j[j]] > -1)
         kap_grad_data[kg_marker[A_j[j]]] += A_data[j];
      else if(count < max_row_size)
      {
         kap_grad_data[count] = A_data[j];
         kap_grad_nonzeros[count] = A_j[j];
         count++;
      }
   }

   hypre_VectorSize(kaporin_gradient) = count;
   for(i = 0; i < count; i++)
   {
      kg_marker[kap_grad_nonzeros[i]] = -1;
      kap_grad_data[i] = hypre_abs(kap_grad_data[i]);
   }

   return hypre_error_flag;

}

void
hypre_swap2_ci( HYPRE_Complex  *v,
                HYPRE_Int      *w,
                HYPRE_Int       i,
                HYPRE_Int       j )
{
   HYPRE_Complex  temp;
   HYPRE_Int      temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/* Quick Sort (largest to smallest) for complex arrays - hypre/utilities/qsort.c did not have what I wanted */
/* sort on v (HYPRE_Complex), move w */

void
hypre_qsort2_ci( HYPRE_Complex  *v,
                 HYPRE_Int      *w,
                 HYPRE_Int      left,
                 HYPRE_Int      right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;

   hypre_swap2_ci( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] > v[left])
         hypre_swap2_ci(v, w, ++last, i);

   hypre_swap2_ci(v, w, left, last);
   hypre_qsort2_ci(v, w, left, last-1);
   hypre_qsort2_ci(v, w, last+1, right);
}

/* Take the largest elements from the kaporin gradient and add their locations to S_Pattern */
HYPRE_Int
hypre_AddToPattern( hypre_Vector *kaporin_gradient,
                    HYPRE_Int    *kap_grad_nonzeros,
                    HYPRE_Int    *S_Pattern,
                    HYPRE_Int    *S_nnz,
                    HYPRE_Int     max_step_size )
{

   HYPRE_Complex *kap_grad_data = hypre_VectorData(kaporin_gradient);
   hypre_qsort2_ci(kap_grad_data, kap_grad_nonzeros, 0, hypre_VectorSize(kaporin_gradient)-1);

   HYPRE_Int i;

   for(i = 0; i < hypre_min(hypre_VectorSize(kaporin_gradient), max_step_size); i++)
      S_Pattern[(*S_nnz)++] = kap_grad_nonzeros[i];

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
   HYPRE_Real              density;

   /* Declare Local variables */

   HYPRE_Int               num_procs, my_id;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   hypre_Vector           *G_temp;            /* A vector to hold a single row of G while it's being calculated */
   hypre_Vector           *A_sub;             /* A vector to hold a the A[P, P] submatrix of A */
   hypre_Vector           *A_subrow;          /* Vector to hold A[i, P] for the kaporin gradient */
   hypre_Vector           *sol_vec;           /* Vector to hold solution of hypre_dpetrs */
   hypre_Vector           *kaporin_gradient;  /* A vector to hold the Kaporin Gradient for each iteration of aFSAI */
   HYPRE_Int              *kap_grad_nonzeros; /* An array to hold the kaporin_gradient nonzero indices for each iteration of aFSAI */
   HYPRE_Int              *S_Pattern;
   HYPRE_Int              *marker;            /* An array that holds which values need to be gathered when finding a sub-row or submatrix */
   HYPRE_Int              *kg_marker;
   HYPRE_Real              old_psi, new_psi;  /* GAG' before and after the k-th interation of aFSAI */
   HYPRE_Real              row_scale;         /* The value to scale G_temp by before adding it to G */
   HYPRE_Int               S_nnz;
   HYPRE_Int               i, j, k, l;           /* Loop variables */
   char uplo = 'L';

   /* Create and initialize G_mat and work vectors for solve */
   G = hypre_ParCSRMatrixCreate (comm,
       hypre_ParCSRMatrixGlobalNumRows(A),
       hypre_ParCSRMatrixGlobalNumCols(A),
       hypre_ParCSRMatrixRowStarts(A),
       hypre_ParCSRMatrixColStarts(A),
       0,
       hypre_ParCSRMatrixGlobalNumRows(A)*(max_steps*max_step_size+1),
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
   kaporin_gradient  = hypre_SeqVectorCreate(max_row_size);
   A_sub             = hypre_SeqVectorCreate(max_row_size*max_row_size);
   S_Pattern         = hypre_CTAlloc(HYPRE_Int, max_row_size, HYPRE_MEMORY_HOST);
   kap_grad_nonzeros = hypre_CTAlloc(HYPRE_Int, max_row_size, HYPRE_MEMORY_HOST);
   marker            = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);       /* For gather functions - don't want to reinitialize */
   kg_marker         = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_HOST);       /* For kaporin gradient functions */

   /* Initializing local variables */

   hypre_SeqVectorInitialize(G_temp);
   hypre_SeqVectorInitialize(A_subrow);
   hypre_SeqVectorInitialize(sol_vec);
   hypre_SeqVectorInitialize(kaporin_gradient);
   hypre_SeqVectorInitialize(A_sub);
   for( i = 0; i < num_rows; i++ )
   {
      marker[i] = -1;
      kg_marker[i] = -1;
   }

   /* Setting data variables for vectors */
   HYPRE_Complex *G_temp_data             = hypre_VectorData(G_temp);
   HYPRE_Complex *A_subrow_data           = hypre_VectorData(A_subrow);
   HYPRE_Complex *kaporin_gradient_data   = hypre_VectorData(kaporin_gradient);
   HYPRE_Complex *A_sub_data              = hypre_VectorData(A_sub);
   HYPRE_Complex *sol_vec_data            = hypre_VectorData(sol_vec);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   for( i = 0; i < num_rows; i++ )    /* Cycle through each of the local rows */
   {

      S_nnz = 0;

      /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
      old_psi = A_data[A_i[i]];

      if(max_step_size > 0)
         for( k = 0; k < max_steps; k++ ) /* Cycle through each iteration for that row */
         {

            /* Compute Kaporin Gradient */
            hypre_FindKapGrad(A_diag, kaporin_gradient, kap_grad_nonzeros, G_temp, S_Pattern, S_nnz, max_row_size, i, marker, kg_marker);

            if(hypre_VectorSize(kaporin_gradient) == 0)
               break;

            /* Find max_step_size largest values of the kaporin gradient, find their column indices, and add it to S_Pattern */
            hypre_AddToPattern(kaporin_gradient, kap_grad_nonzeros, S_Pattern, &S_nnz, max_step_size);

            /* Gather A[P, P] and -A[P, i] */
            hypre_qsort0(S_Pattern, 0, S_nnz-1);

            for(j = 0; j < S_nnz; ++j)
               marker[S_Pattern[j]] = j;

            hypre_VectorSize(A_sub)    = S_nnz * S_nnz;
            hypre_VectorSize(A_subrow) = S_nnz;
            hypre_VectorSize(G_temp)   = S_nnz;

            hypre_SeqVectorSetConstantValues(A_sub, 0.0);
            hypre_SeqVectorSetConstantValues(A_subrow, 0.0);

            hypre_CSRMatrixExtractDenseMatrix(A_diag, A_sub, S_Pattern, S_nnz, marker);                           /* A[P, P] */
            hypre_ExtractDenseRowFromCSRMatrix(A_diag, A_subrow, marker, i);                                      /* A[i, P] */

            hypre_SeqVectorScale(-1, A_subrow);                                                                   /* -A[P, i] */
            for(j = 0; j < S_nnz; j++)
               sol_vec_data[j] = A_subrow_data[j];

            /* Solve A[P, P]G[i, P]' = -A[P, i] */
            hypre_dpotrf(&uplo, &S_nnz, A_sub_data, &S_nnz, &l);

            j = 1;

            hypre_dpotrs(&uplo, &S_nnz, &j, A_sub_data, &S_nnz, sol_vec_data, &S_nnz, &l);                        /* A_subrow becomes the solution vector... */

            for(j = 0; j < hypre_VectorSize(sol_vec); j++)
               G_temp_data[j] = sol_vec_data[j];


            /* Determine psi_{k+1} = G_temp[i]*A*G_temp[i]' */
            //hypre_SeqVectorScale(-1, A_subrow);                         /* A[P, i] */
            new_psi = hypre_SeqVectorInnerProd(G_temp, A_subrow) - A_data[A_i[i]];
            if(hypre_abs( new_psi + old_psi ) < kap_tolerance*old_psi)
               break;

            old_psi = -new_psi;
         }

      for(j = 0; j < S_nnz; j++)
         marker[S_Pattern[j]] = -1;

      row_scale = 1/(sqrt(A_data[A_i[i]] - hypre_abs(hypre_SeqVectorInnerProd(G_temp, A_subrow))));
      G_j[G_i[i]] = i;
      G_data[G_i[i]] = row_scale;

      /* Pass values of G_temp into G */
      hypre_SeqVectorScale(row_scale, G_temp);
      for(k = 0; k < hypre_VectorSize(G_temp); k++)
      {
         j           = G_i[i] + k + 1;
         G_j[j]      = S_Pattern[k];
         G_data[j]   = G_temp_data[k];
      }
      G_i[i+1] = G_i[i] + k + 1;
   }

   /* Update local number of nonzeros of G */
   hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows];

   /* Compute G^T */
   hypre_ParCSRMatrixTranspose(G, &hypre_ParFSAIDataGTmat(fsai_data), 1);

#if DEBUG
   char filename[] = "FSAI.out.G.ij";
   hypre_ParCSRMatrixPrintIJ(G, 0, 0, filename);
#endif

   /* Print Setup info*/
   if (print_level == 1)
   {
      hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);

      /* Compute density */
      hypre_ParCSRMatrixSetDNumNonzeros(G);
      hypre_ParCSRMatrixSetDNumNonzeros(A);
      density = hypre_ParCSRMatrixDNumNonzeros(G)/
                hypre_ParCSRMatrixDNumNonzeros(A);
      hypre_ParFSAIDataDensity(fsai_data) = density;

      if (!my_id)
      {
         hypre_printf("*************************\n");
         hypre_printf("* HYPRE FSAI Setup Info *\n");
         hypre_printf("*************************\n\n");

         hypre_printf("+--------------------------+\n");
         hypre_printf("| Max no. steps:   %6d |\n", max_steps);
         hypre_printf("| Max step size:   %6d |\n", max_step_size);
         hypre_printf("| Kap grad tol:    %8.1e |\n", kap_tolerance);
         hypre_printf("| G Prec. density: %8.3f |\n", density);
         hypre_printf("| Omega factor:    %8.3f |\n", hypre_ParFSAIDataOmega(fsai_data));
         hypre_printf("+--------------------------+\n");

         hypre_printf("\n\n");
      }
   }

   hypre_SeqVectorDestroy(G_temp);
   hypre_SeqVectorDestroy(A_subrow);
   hypre_SeqVectorDestroy(sol_vec);
   hypre_SeqVectorDestroy(kaporin_gradient);
   hypre_SeqVectorDestroy(A_sub);
   hypre_TFree(kap_grad_nonzeros, HYPRE_MEMORY_HOST);
   hypre_TFree(S_Pattern, HYPRE_MEMORY_HOST);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   hypre_TFree(kg_marker, HYPRE_MEMORY_HOST);

   return(hypre_error_flag);

}
