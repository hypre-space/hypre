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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDenseMat
 *
 * Extract A[P, P] into a dense matrix.
 *
 * Parameters:
 * - A:       The hypre_CSRMatrix whose submatrix will be extracted.
 * - A_sub:   A patt_size^2 - sized array to hold the lower triangular of
 *            the symmetric submatrix A[P, P].
 * - pattern: A patt_size - sized array to hold the wanted rows/cols.
 * - marker:  A work array of length equal to the number of columns in A.
 *            All values should be -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixExtractDenseMat( hypre_CSRMatrix *A,
                                hypre_Vector    *A_sub,
                                HYPRE_Int       *pattern,
                                HYPRE_Int        patt_size,
                                HYPRE_Int       *marker )
{
   HYPRE_Int     *A_i        = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j        = hypre_CSRMatrixJ(A);
   HYPRE_Real    *A_data     = hypre_CSRMatrixData(A);
   HYPRE_Complex *A_sub_data = hypre_VectorData(A_sub);

   /* Local variables */
   HYPRE_Int      cc, i, ii, j;

   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii+1]; j++)
      {
         if ((A_j[j] <= ii) &&
             (cc = marker[A_j[j]]) >= 0)
         {
            A_sub_data[cc*patt_size + i] = A_data[j];
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDenseRow
 *
 * Extract the dense subrow from a matrix (A[i, P])
 *
 * Parameters:
 * - A:         The hypre_CSRMatrix whose subrow will be extracted.
 * - A_subrow:  The extracted subrow of A[i, P].
 * - marker:    A work array of length equal to the number of row in A.
 *              Assumed to be set to all -1.
 * - row_num:   which row index of A we want to extract data from.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixExtractDenseRow( hypre_CSRMatrix *A,
                                hypre_Vector    *A_subrow,
                                HYPRE_Int       *marker,
                                HYPRE_Int        row_num )
{
   HYPRE_Int      *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j          = hypre_CSRMatrixJ(A);
   HYPRE_Real     *A_data       = hypre_CSRMatrixData(A);
   HYPRE_Complex  *sub_row_data = hypre_VectorData(A_subrow);

   /* Local variables */
   HYPRE_Int       j, cc;

   for (j = A_i[row_num]; j < A_i[row_num+1]; j++)
   {
      if ((cc = marker[A_j[j]]) >= 0)
      {
         sub_row_data[cc] = A_data[j];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FindKapGrad
 *
 * Finding the Kaporin Gradient contribution (psi) of a given row.
 *
 * Parameters:
 *  - A:            CSR matrix diagonal of A.
 *  - kap_grad:     Array holding the kaporin gradient.
 *                  This will we modified.
 *  - kg_pos:       Array of the nonzero column indices of kap_grad.
 *                  To be modified.
 *  - G_temp:       Work array of G for row i.
 *  - pattern:      Array of column indices of the nonzeros of G_temp.
 *  - patt_size:    Number of column indices of the nonzeros of G_temp.
 *  - max_row_size: To ensure we don't overfill kap_grad.
 *  - row_num:      Which row of G we are working on.
 *  - marker:       Array of length equal to the number of rows in A.
 *                  Assumed to all be set to -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FindKapGrad( hypre_CSRMatrix  *A_diag,
                   hypre_Vector     *kap_grad,
                   HYPRE_Int        *kg_pos,
                   hypre_Vector     *G_temp,
                   HYPRE_Int        *pattern,
                   HYPRE_Int         patt_size,
                   HYPRE_Int         max_row_size,
                   HYPRE_Int         row_num,
                   HYPRE_Int        *kg_marker )
{

   HYPRE_Int      *A_i           = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j           = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real     *A_data        = hypre_CSRMatrixData(A_diag);
   HYPRE_Complex  *G_temp_data   = hypre_VectorData(G_temp);
   HYPRE_Complex  *kap_grad_data = hypre_VectorData(kap_grad);

   /* Local Variables */
   HYPRE_Int       i, ii, j, k, count, col;

   count = 0;

   /* Compute A[row_num, 0:(row_num-1)]*G_temp[i,i] */
   for (j = A_i[row_num]; j < A_i[row_num+1]; j++)
   {
      col = A_j[j];
      if (col < row_num)
      {
         if (kg_marker[col] > -1)
         {
            /* Add A[row_num, col] to the tentative pattern */
            kg_marker[col] = count + 1;
            kg_pos[count] = col;
            kap_grad_data[count] = A_data[j];
            count++;
         }
      }
   }

   /* Compute A[0:(row_num-1), P]*G_temp[P, i] */
   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii+1]; j++)
      {
         col = A_j[j];
         if (col < row_num)
         {
            k = kg_marker[col];
            if (k == 0)
            {
               /* New entry in the tentative pattern */
               kg_marker[col] = count + 1;
               kg_pos[count] = col;
               kap_grad_data[count] = G_temp_data[i]*A_data[j];
               count++;
            }
            else if (k > 0)
            {
               /* Already existing entry in the tentative pattern */
               kap_grad_data[k-1] += G_temp_data[i]*A_data[j];
            }
         }
      }
   }

   /* Update number of nonzero coefficients held in kap_grad */
   hypre_VectorSize(kap_grad) = count;

   /* Update to absolute values */
   for (i = 0; i < count; i++)
   {
      kap_grad_data[i] = hypre_abs(kap_grad_data[i]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_swap2_ci
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_qsort2_ci
 *
 * Quick Sort (largest to smallest) for complex arrays.
 * Sort on v (HYPRE_Complex), move w.
 *--------------------------------------------------------------------------*/

void
hypre_qsort2_ci( HYPRE_Complex  *v,
                 HYPRE_Int      *w,
                 HYPRE_Int      left,
                 HYPRE_Int      right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }

   hypre_swap2_ci(v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] > v[left])
      {
         hypre_swap2_ci(v, w, ++last, i);
      }
   }

   hypre_swap2_ci(v, w, left, last);
   hypre_qsort2_ci(v, w, left, last-1);
   hypre_qsort2_ci(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 * hypre_PartialSelectSortCI
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_PartialSelectSortCI( HYPRE_Complex  *v,
                           HYPRE_Int      *w,
                           HYPRE_Int       size,
                           HYPRE_Int       nentries )
{
   HYPRE_Int  i, k, pos;

   for (k = 0; k < nentries; k++)
   {
      /* Find largest kth entry */
      pos = k;
      for (i = k + 1; i < size; i++)
      {
         if (v[i] > v[pos])
         {
            pos = i;
         }
      }

      /* Move entry to beggining of the array */
      hypre_swap2_ci(v, w, k, pos);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AddToPattern
 *
 * Take the largest elements from the kaporin gradient and add their
 * locations to pattern.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AddToPattern( hypre_Vector *kap_grad,
                    HYPRE_Int    *kg_pos,
                    HYPRE_Int    *pattern,
                    HYPRE_Int    *patt_size,
                    HYPRE_Int    *kg_marker,
                    HYPRE_Int     max_step_size )
{
   HYPRE_Int       kap_grad_size = hypre_VectorSize(kap_grad);
   HYPRE_Complex  *kap_grad_data = hypre_VectorData(kap_grad);

   HYPRE_Int       i, nentries;
   HYPRE_Int       patt_size_old;

   /* Number of entries that can be added */
   nentries = hypre_min(kap_grad_size, max_step_size);

   /* Reorder candidates according to larger weights */
   //hypre_qsort2_ci(kap_grad_data, &kg_pos, 0, kap_grad_size-1);
   hypre_PartialSelectSortCI(kap_grad_data, kg_pos, kap_grad_size, nentries);

   /* Update pattern with new entries */
   patt_size_old = *patt_size;
   for (i = 0; i < nentries; i++)
   {
      pattern[*patt_size + i] = kg_pos[i];
   }
   *patt_size += nentries;

   /* Put pattern in ascending order */
   hypre_qsort0(pattern, 0, (*patt_size)-1);

   /* Reset marked entries that are added to pattern */
   for (i = 0; i < nentries; i++)
   {
      kg_marker[kg_pos[i]] = -1;
   }
   for (i = nentries; i < kap_grad_size; i++)
   {
      kg_marker[kg_pos[i]] = 0;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_DenseSPDSystemSolve
 *
 * Solve the dense SPD linear system with LAPACK:
 *
 *    mat*lhs = -rhs
 *
 * Note: the contents of A change to its Cholesky factor.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DenseSPDSystemSolve( hypre_Vector *mat,
                           hypre_Vector *rhs,
                           hypre_Vector *lhs )
{
   HYPRE_Int      size = hypre_VectorSize(rhs);
   HYPRE_Complex *mat_data = hypre_VectorData(mat);
   HYPRE_Complex *rhs_data = hypre_VectorData(rhs);
   HYPRE_Complex *lhs_data = hypre_VectorData(lhs);

   /* Local variables */
   HYPRE_Int      num_rhs = 1;
   char           uplo = 'L';
   char           msg[512];
   HYPRE_Int      i, info;

   /* Copy RHS into LHS */
   for (i = 0; i < size; i++)
   {
      lhs_data[i] = -rhs_data[i];
   }

   /* Compute Cholesky factor */
   hypre_dpotrf(&uplo, &size, mat_data, &size, &info);
   if (info)
   {
      hypre_sprintf(msg, "Error: dpotrf failed with code %d\n", info);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   /* Solve dense linear system */
   hypre_dpotrs(&uplo, &size, &num_rhs, mat_data, &size, lhs_data, &size, &info);
   if (info)
   {
      hypre_sprintf(msg, "Error: dpotrs failed with code %d\n", info);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetup( void               *fsai_vdata,
                 hypre_ParCSRMatrix *A,
                 hypre_ParVector    *f,
                 hypre_ParVector    *u )
{
   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               print_level      = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Real              density;

   /* ParCSRMatrix A variables */
   MPI_Comm                comm             = hypre_ParCSRMatrixComm(A);
   hypre_MemoryLocation    mem_loc          = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_BigInt            num_rows_A       = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt            num_cols_A       = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt           *row_starts_A     = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt           *col_starts_A     = hypre_ParCSRMatrixColStarts(A);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Int              *A_j              = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real             *A_data           = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               num_nnzs_diag_A  = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int               avg_nnzrow_diag_A;

   /* Matrix G variables */
   hypre_ParCSRMatrix     *G;
   hypre_CSRMatrix        *G_diag;
   HYPRE_Int              *G_i;
   HYPRE_Int              *G_j;
   HYPRE_Real             *G_data;
   HYPRE_Int               max_nnzrow_diag_G;   /* Maximum number of nonzeros per row in G_diag */
   HYPRE_Int               max_nonzeros_diag_G; /* Maximum number of nonzeros in G_diag */

   /* Work vectors*/
   hypre_ParVector         *r_work;
   hypre_ParVector         *z_work;

   /* Local variables */
   char                     msg[512];      /* Warning message */
   HYPRE_Int                num_procs;
   HYPRE_Int                my_id;
   hypre_Vector            *G_temp;        /* Vector holding the values of G[i,:] */
   hypre_Vector            *A_sub;         /* Vector holding the dense submatrix A[P, P] */
   hypre_Vector            *A_subrow;      /* Vector holding A[i, P] */
   hypre_Vector            *kap_grad;      /* Vector holding the Kaporin gradient values */
   HYPRE_Int               *kg_pos;        /* Indices of nonzero entries of kap_grad */
   HYPRE_Int               *kg_marker;     /* Marker array with nonzeros pointing to kg_pos */
   HYPRE_Int               *marker;        /* Marker array with nonzeros pointing to P */
   HYPRE_Int               *pattern;       /* Array holding column indices of G[i,:] */
   HYPRE_Int                max_cand_size; /* Max size of kg_pos */
   HYPRE_Int                patt_size;     /* Number of entries in current pattern */
   HYPRE_Int                patt_size_old; /* Number of entries in previous pattern */
   HYPRE_Int                i, j, k, l;    /* Loop variables */
   HYPRE_Real               old_psi;       /* GAG' before k-th interation of aFSAI */
   HYPRE_Real               new_psi;       /* GAG' after k-th interation of aFSAI */
   HYPRE_Real               row_scale;     /* Scaling factor for G_temp */
   HYPRE_Complex           *G_temp_data;
   HYPRE_Complex           *A_subrow_data;
   HYPRE_Complex           *kap_grad_data;
   HYPRE_Complex           *A_sub_data;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   /* Create and initialize the matrix G */
   avg_nnzrow_diag_A   = num_nnzs_diag_A/num_rows_diag_A;
   max_nnzrow_diag_G   = max_steps*max_step_size + 1;
   max_nonzeros_diag_G = num_rows_diag_A*max_nnzrow_diag_G;
   max_cand_size       = avg_nnzrow_diag_A*max_nnzrow_diag_G;
   G = hypre_ParCSRMatrixCreate(comm, num_rows_A, num_cols_A,
                                row_starts_A, col_starts_A,
                                0, max_nonzeros_diag_G, 0);
   hypre_ParCSRMatrixInitialize(G);
   G_diag = hypre_ParCSRMatrixDiag(G);
   G_data = hypre_CSRMatrixData(G_diag);
   G_i = hypre_CSRMatrixI(G_diag);
   G_j = hypre_CSRMatrixJ(G_diag);
   hypre_ParFSAIDataGmat(fsai_data) = G;

   /* Create and initialize work vectors used in the solve phase */
   r_work = hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);
   z_work = hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);

   hypre_ParVectorInitialize(r_work);
   hypre_ParVectorInitialize(z_work);

   hypre_ParFSAIDataRWork(fsai_data) = r_work;
   hypre_ParFSAIDataZWork(fsai_data) = z_work;

   /* Allocate and initialize local vector variables */
   G_temp    = hypre_SeqVectorCreate(max_nnzrow_diag_G);
   A_subrow  = hypre_SeqVectorCreate(max_nnzrow_diag_G);
   kap_grad  = hypre_SeqVectorCreate(max_cand_size); // TODO: size
   A_sub     = hypre_SeqVectorCreate(max_nnzrow_diag_G*max_nnzrow_diag_G);
   pattern   = hypre_CTAlloc(HYPRE_Int, max_nnzrow_diag_G, HYPRE_MEMORY_HOST);
   kg_pos    = hypre_CTAlloc(HYPRE_Int, max_cand_size, HYPRE_MEMORY_HOST); // TODO: size
   kg_marker = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
   marker    = hypre_TAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

   hypre_SeqVectorInitialize(G_temp);
   hypre_SeqVectorInitialize(A_subrow);
   hypre_SeqVectorInitialize(kap_grad);
   hypre_SeqVectorInitialize(A_sub);
   hypre_Memset(marker, -1, num_rows_diag_A*sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

   /* Setting data variables for vectors */
   G_temp_data     = hypre_VectorData(G_temp);
   A_subrow_data   = hypre_VectorData(A_subrow);
   kap_grad_data   = hypre_VectorData(kap_grad);
   A_sub_data      = hypre_VectorData(A_sub);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
   for (i = 0; i < num_rows_diag_A; i++)
   {
      patt_size = 0;

      /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
      old_psi = A_data[A_i[i]];

      /* Cycle through each iteration for that row */
      for (k = 0; k < max_steps; k++)
      {
         /* Compute Kaporin Gradient */
         hypre_FindKapGrad(A_diag, kap_grad, kg_pos, G_temp, pattern,
                           patt_size, max_nnzrow_diag_G, i, kg_marker);

         /* Find max_step_size largest values of the kaporin gradient,
            find their column indices, and add it to pattern */
         patt_size_old = patt_size;
         hypre_AddToPattern(kap_grad, kg_pos, pattern, &patt_size,
                            kg_marker, max_step_size);

         if (patt_size == patt_size_old)
         {
            new_psi = old_psi;
            hypre_VectorSize(G_temp) = patt_size;
            break;
         }

         /* Gather A[P, P] and -A[i, P] */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = j;
         }

         hypre_VectorSize(A_sub)    = patt_size * patt_size;
         hypre_VectorSize(A_subrow) = patt_size;
         hypre_VectorSize(G_temp)   = patt_size;

         hypre_SeqVectorSetConstantValues(A_sub, 0.0);
         hypre_SeqVectorSetConstantValues(A_subrow, 0.0);

         hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker); /* A[P, P] */
         hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);               /* A[i, P] */

         /* Solve A[P, P] G[i, P]' = -A[i, P] */
         hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

         /* Determine psi_{k+1} = G_temp[i]*A*G_temp[i]' */
         new_psi = hypre_SeqVectorInnerProd(G_temp, A_subrow) + A_data[A_i[i]];
         if (hypre_abs(new_psi - old_psi) < kap_tolerance*old_psi)
         {
            break;
         }
         old_psi = new_psi;
      }

      /* Reset marker for building dense linear system */
      for (j = 0; j < patt_size; j++)
      {
         marker[pattern[j]] = -1;
      }

      /* Compute scaling factor */
      if (new_psi > 0)
      {
         row_scale = 1.0/sqrt(new_psi);
      }
      else
      {
         hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
         hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);

         row_scale = 1.0/A_data[A_i[i]];
         hypre_VectorSize(G_temp) = 0;
      }

      /* Pass values of G_temp into G */
      G_j[G_i[i]] = i;
      G_data[G_i[i]] = row_scale;
      for (k = 0; k < hypre_VectorSize(G_temp); k++)
      {
         j         = G_i[i] + k + 1;
         G_j[j]    = pattern[k];
         G_data[j] = row_scale*G_temp_data[k];
         kg_marker[pattern[k]] = 0;
      }
      G_i[i+1] = G_i[i] + k + 1;
   }
   HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Update local number of nonzeros of G */
   hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

   /* Compute G^T */
   hypre_ParCSRMatrixTranspose(G, &hypre_ParFSAIDataGTmat(fsai_data), 1);

   /* Print Setup info */
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

         hypre_printf("+---------------------------+\n");
         hypre_printf("| Max no. steps:   %8d |\n", max_steps);
         hypre_printf("| Max step size:   %8d |\n", max_step_size);
         hypre_printf("| Kap grad tol:    %8.1e |\n", kap_tolerance);
         hypre_printf("| Prec. density:   %8.3f |\n", density);
         hypre_printf("| Eig max iters:   %8d |\n", hypre_ParFSAIDataEigMaxIters(fsai_data));
         hypre_printf("| Omega factor:    %8.3f |\n", hypre_ParFSAIDataOmega(fsai_data));
         hypre_printf("+---------------------------+\n");

         hypre_printf("\n\n");
      }
   }

#if DEBUG
   char filename[] = "FSAI.out.G.ij";
   hypre_ParCSRMatrixPrintIJ(G, 0, 0, filename);
#endif

   /* Free memory */
   hypre_SeqVectorDestroy(G_temp);
   hypre_SeqVectorDestroy(A_subrow);
   hypre_SeqVectorDestroy(kap_grad);
   hypre_SeqVectorDestroy(A_sub);
   hypre_TFree(kg_pos, HYPRE_MEMORY_HOST);
   hypre_TFree(pattern, HYPRE_MEMORY_HOST);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   hypre_TFree(kg_marker, HYPRE_MEMORY_HOST);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*****************************************************************************
 * hypre_FSAIComputeOmega
 *
 * Approximates the relaxation factor omega with 1/eigmax(G^T*G*A), where the
 * maximum eigenvalue is computed with a fixed number of iterations via the
 * power method.
 ******************************************************************************/

HYPRE_Int
hypre_FSAIComputeOmega( void *fsai_vdata,
                        hypre_ParCSRMatrix *A )
{
   hypre_ParFSAIData    *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   hypre_ParCSRMatrix   *G  =  hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix   *GT =  hypre_ParFSAIDataGTmat(fsai_data);
   hypre_ParVector      *r_work = hypre_ParFSAIDataRWork(fsai_data);
   hypre_ParVector      *z_work = hypre_ParFSAIDataZWork(fsai_data);
   HYPRE_Int             eig_max_iters = hypre_ParFSAIDataEigMaxIters(fsai_data);

   hypre_ParVector      *eigvec;
   hypre_ParVector      *eigvec_old;

   HYPRE_Int             i;
   HYPRE_Real            norm, invnorm, lambda, omega;

   if (eig_max_iters)
   {
      eigvec_old = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                         hypre_ParCSRMatrixGlobalNumRows(A),
                                         hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(eigvec_old);
      eigvec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                     hypre_ParCSRMatrixGlobalNumRows(A),
                                     hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(eigvec);
      hypre_ParVectorSetRandomValues(eigvec, 256);

      /* Power method iteration */
      for (i = 0; i < eig_max_iters; i++)
      {
         norm = hypre_ParVectorInnerProd(eigvec, eigvec);
         invnorm = 1.0/sqrt(norm);
         hypre_ParVectorScale(invnorm, eigvec);

         if (i == (eig_max_iters - 1))
         {
            hypre_ParVectorCopy(eigvec, eigvec_old);
         }

         /* eigvec = GT * G * A * eigvec */
         hypre_ParCSRMatrixMatvec(1.0, A,  eigvec, 0.0, r_work);
         hypre_ParCSRMatrixMatvec(1.0, G,  r_work, 0.0, z_work);
         hypre_ParCSRMatrixMatvec(1.0, GT, z_work, 0.0, eigvec);
      }
      norm = hypre_ParVectorInnerProd(eigvec, eigvec_old);
      lambda = sqrt(norm);

      /* Free memory */
      hypre_ParVectorDestroy(eigvec_old);
      hypre_ParVectorDestroy(eigvec);

      /* Update omega */
      omega = 1.0/lambda;
      hypre_FSAISetOmega(fsai_vdata, omega);
   }

   return hypre_error_flag;
}
