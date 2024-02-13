/******************************************************************************
 *  Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 *  HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 *  SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

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
   HYPRE_Int     *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_a = hypre_CSRMatrixData(A);
   HYPRE_Complex *A_sub_data = hypre_VectorData(A_sub);

   /* Local variables */
   HYPRE_Int      cc, i, ii, j;

   // TODO: Do we need to reinitialize all entries?
   for (i = 0; i < hypre_VectorSize(A_sub); i++)
   {
      A_sub_data[i] = 0.0;
   }

   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
      {
         if ((A_j[j] <= ii) &&
             (cc = marker[A_j[j]]) >= 0)
         {
            A_sub_data[cc * patt_size + i] = A_a[j];
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
   HYPRE_Int      *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex  *A_a = hypre_CSRMatrixData(A);
   HYPRE_Complex  *sub_row_data = hypre_VectorData(A_subrow);

   /* Local variables */
   HYPRE_Int       j, cc;

   for (j = 0; j < hypre_VectorSize(A_subrow); j++)
   {
      sub_row_data[j] = 0.0;
   }

   for (j = A_i[row_num]; j < A_i[row_num + 1]; j++)
   {
      if ((cc = marker[A_j[j]]) >= 0)
      {
         sub_row_data[cc] = A_a[j];
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
   HYPRE_UNUSED_VAR(max_row_size);

   HYPRE_Int      *A_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int      *A_j = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex  *A_a = hypre_CSRMatrixData(A_diag);
   HYPRE_Complex  *G_temp_data   = hypre_VectorData(G_temp);
   HYPRE_Complex  *kap_grad_data = hypre_VectorData(kap_grad);

   /* Local Variables */
   HYPRE_Int       i, ii, j, k, count, col;

   count = 0;

   /* Compute A[row_num, 0:(row_num-1)]*G_temp[i,i] */
   for (j = A_i[row_num]; j < A_i[row_num + 1]; j++)
   {
      col = A_j[j];
      if (col < row_num)
      {
         if (kg_marker[col] > -1)
         {
            /* Add A[row_num, col] to the tentative pattern */
            kg_marker[col] = count + 1;
            kg_pos[count] = col;
            kap_grad_data[count] = A_a[j];
            count++;
         }
      }
   }

   /* Compute A[0:(row_num-1), P]*G_temp[P, i] */
   for (i = 0; i < patt_size; i++)
   {
      ii = pattern[i];
      for (j = A_i[ii]; j < A_i[ii + 1]; j++)
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
               kap_grad_data[count] = G_temp_data[i] * A_a[j];
               count++;
            }
            else if (k > 0)
            {
               /* Already existing entry in the tentative pattern */
               kap_grad_data[k - 1] += G_temp_data[i] * A_a[j];
            }
         }
      }
   }

   /* Update number of nonzero coefficients held in kap_grad */
   hypre_VectorSize(kap_grad) = count;

   /* Update to absolute values */
   for (i = 0; i < count; i++)
   {
      kap_grad_data[i] = hypre_cabs(kap_grad_data[i]);
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
 * Sort on real portion of v (HYPRE_Complex), move w.
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

   hypre_swap2_ci(v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (hypre_creal(v[i]) > hypre_creal(v[left]))
      {
         hypre_swap2_ci(v, w, ++last, i);
      }
   }

   hypre_swap2_ci(v, w, left, last);
   hypre_qsort2_ci(v, w, left, last - 1);
   hypre_qsort2_ci(v, w, last + 1, right);
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
         if (hypre_creal(v[i]) > hypre_creal(v[pos]))
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

   /* Number of entries that can be added */
   nentries = hypre_min(kap_grad_size, max_step_size);

   /* Reorder candidates according to larger weights */
   //hypre_qsort2_ci(kap_grad_data, &kg_pos, 0, kap_grad_size-1);
   hypre_PartialSelectSortCI(kap_grad_data, kg_pos, kap_grad_size, nentries);

   /* Update pattern with new entries */
   for (i = 0; i < nentries; i++)
   {
      pattern[*patt_size + i] = kg_pos[i];
   }
   *patt_size += nentries;

   /* Put pattern in ascending order */
   hypre_qsort0(pattern, 0, (*patt_size) - 1);

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
 * hypre_FSAISetupNative
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupNative( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex          *A_a              = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               num_nnzs_diag_A  = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int               avg_nnzrow_diag_A;

   /* Matrix G variables */
   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag;
   HYPRE_Int              *G_i;
   HYPRE_Int              *G_j;
   HYPRE_Complex          *G_a;
   HYPRE_Int               max_nnzrow_diag_G;   /* Max. number of nonzeros per row in G_diag */
   HYPRE_Int               max_cand_size;       /* Max size of kg_pos */

   /* Local variables */
   char                     msg[512];    /* Warning message */
   HYPRE_Int           *twspace;     /* shared work space for omp threads */

   /* Initalize some variables */
   avg_nnzrow_diag_A = (num_rows_diag_A > 0) ? num_nnzs_diag_A / num_rows_diag_A : 0;
   max_nnzrow_diag_G = max_steps * max_step_size + 1;
   max_cand_size     = avg_nnzrow_diag_A * max_nnzrow_diag_G;

   G_diag = hypre_ParCSRMatrixDiag(G);
   G_a = hypre_CSRMatrixData(G_diag);
   G_i = hypre_CSRMatrixI(G_diag);
   G_j = hypre_CSRMatrixJ(G_diag);

   /* Allocate shared work space array for OpenMP threads */
   twspace = hypre_CTAlloc(HYPRE_Int, hypre_NumThreads() + 1, HYPRE_MEMORY_HOST);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      hypre_Vector   *G_temp;        /* Vector holding the values of G[i,:] */
      hypre_Vector   *A_sub;         /* Vector holding the dense submatrix A[P, P] */
      hypre_Vector   *A_subrow;      /* Vector holding A[i, P] */
      hypre_Vector   *kap_grad;      /* Vector holding the Kaporin gradient values */
      HYPRE_Int      *kg_pos;        /* Indices of nonzero entries of kap_grad */
      HYPRE_Int      *kg_marker;     /* Marker array with nonzeros pointing to kg_pos */
      HYPRE_Int      *marker;        /* Marker array with nonzeros pointing to P */
      HYPRE_Int      *pattern;       /* Array holding column indices of G[i,:] */
      HYPRE_Int       patt_size;     /* Number of entries in current pattern */
      HYPRE_Int       patt_size_old; /* Number of entries in previous pattern */
      HYPRE_Int       ii;            /* Thread identifier */
      HYPRE_Int       num_threads;   /* Number of active threads */
      HYPRE_Int       ns, ne;        /* Initial and last row indices */
      HYPRE_Int       i, j, k, iloc; /* Loop variables */
      HYPRE_Complex   old_psi;       /* GAG' before k-th interation of aFSAI */
      HYPRE_Complex   new_psi;       /* GAG' after k-th interation of aFSAI */
      HYPRE_Complex   row_scale;     /* Scaling factor for G_temp */
      HYPRE_Complex  *G_temp_data;
      HYPRE_Complex  *A_subrow_data;

      HYPRE_Int       num_rows_Gloc;
      HYPRE_Int       num_nnzs_Gloc;
      HYPRE_Int      *Gloc_i;
      HYPRE_Int      *Gloc_j;
      HYPRE_Complex  *Gloc_a;

      /* Allocate and initialize local vector variables */
      G_temp    = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      A_subrow  = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      kap_grad  = hypre_SeqVectorCreate(max_cand_size);
      A_sub     = hypre_SeqVectorCreate(max_nnzrow_diag_G * max_nnzrow_diag_G);
      pattern   = hypre_CTAlloc(HYPRE_Int, max_nnzrow_diag_G, HYPRE_MEMORY_HOST);
      kg_pos    = hypre_CTAlloc(HYPRE_Int, max_cand_size, HYPRE_MEMORY_HOST);
      kg_marker = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
      marker    = hypre_TAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

      hypre_SeqVectorInitialize_v2(G_temp, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_subrow, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(kap_grad, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_sub, HYPRE_MEMORY_HOST);
      hypre_Memset(marker, -1, num_rows_diag_A * sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

      /* Setting data variables for vectors */
      G_temp_data   = hypre_VectorData(G_temp);
      A_subrow_data = hypre_VectorData(A_subrow);

      ii = hypre_GetThreadNum();
      num_threads = hypre_NumActiveThreads();
      hypre_partition1D(num_rows_diag_A, num_threads, ii, &ns, &ne);

      num_rows_Gloc = ne - ns;
      if (num_threads == 1)
      {
         Gloc_i = G_i;
         Gloc_j = G_j;
         Gloc_a = G_a;
      }
      else
      {
         num_nnzs_Gloc = num_rows_Gloc * max_nnzrow_diag_G;

         Gloc_i = hypre_CTAlloc(HYPRE_Int, num_rows_Gloc + 1, HYPRE_MEMORY_HOST);
         Gloc_j = hypre_CTAlloc(HYPRE_Int, num_nnzs_Gloc, HYPRE_MEMORY_HOST);
         Gloc_a = hypre_CTAlloc(HYPRE_Complex, num_nnzs_Gloc, HYPRE_MEMORY_HOST);
      }

      for (i = ns; i < ne; i++)
      {
         patt_size = 0;

         /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
         new_psi = old_psi = A_a[A_i[i]];

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

            /* Update sizes */
            hypre_VectorSize(A_sub)    = patt_size * patt_size;
            hypre_VectorSize(A_subrow) = patt_size;
            hypre_VectorSize(G_temp)   = patt_size;

            if (patt_size == patt_size_old)
            {
               new_psi = old_psi;
               break;
            }
            else
            {
               /* Gather A[P, P] and -A[i, P] */
               for (j = 0; j < patt_size; j++)
               {
                  marker[pattern[j]] = j;
               }
               hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker);
               hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);

               /* Solve A[P, P] G[i, P]' = -A[i, P] */
               hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

               /* Determine psi_{k+1} = G_temp[i] * A[P, P] * G_temp[i]' */
               new_psi = A_a[A_i[i]];
               for (j = 0; j < patt_size; j++)
               {
                  new_psi += G_temp_data[j] * A_subrow_data[j];
               }

               /* Check psi reduction */
               if (hypre_cabs(new_psi - old_psi) < hypre_creal(kap_tolerance * old_psi))
               {
                  break;
               }
               else
               {
                  old_psi = new_psi;
               }
            }
         }

         /* Reset marker for building dense linear system */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = -1;
         }

         /* Compute scaling factor */
         if (hypre_creal(new_psi) > 0 && hypre_cimag(new_psi) == 0)
         {
            row_scale = 1.0 / hypre_csqrt(new_psi);
         }
         else
         {
            hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);

            row_scale = 1.0 / hypre_cabs(A_a[A_i[i]]);
            hypre_VectorSize(G_temp) = patt_size = 0;
         }

         /* Pass values of G_temp into G */
         iloc = i - ns;
         Gloc_j[Gloc_i[iloc]] = i;
         Gloc_a[Gloc_i[iloc]] = row_scale;
         for (k = 0; k < patt_size; k++)
         {
            j = Gloc_i[iloc] + k + 1;
            Gloc_j[j] = pattern[k];
            Gloc_a[j] = row_scale * G_temp_data[k];
            kg_marker[pattern[k]] = 0;
         }
         Gloc_i[iloc + 1] = Gloc_i[iloc] + k + 1;
      }

      /* Copy data to shared memory */
      twspace[ii + 1] = Gloc_i[num_rows_Gloc] - Gloc_i[0];
#ifdef HYPRE_USING_OPENMP
      #pragma omp barrier
      #pragma omp single
#endif
      {
         for (i = 0; i < num_threads; i++)
         {
            twspace[i + 1] += twspace[i];
         }
      }

      if (num_threads > 1)
      {
         /* Correct row pointer G_i */
         G_i[ns] = twspace[ii];
         for (i = ns; i < ne; i++)
         {
            iloc = i - ns;
            G_i[i + 1] = G_i[i] + Gloc_i[iloc + 1] - Gloc_i[iloc];
         }

         /* Move G_j and G_a */
         for (i = ns; i < ne; i++)
         {
            for (j = G_i[i]; j < G_i[i + 1]; j++)
            {
               G_j[j] = Gloc_j[j - G_i[ns]];
               G_a[j] = Gloc_a[j - G_i[ns]];
            }
         }

         hypre_TFree(Gloc_i, HYPRE_MEMORY_HOST);
         hypre_TFree(Gloc_j, HYPRE_MEMORY_HOST);
         hypre_TFree(Gloc_a, HYPRE_MEMORY_HOST);
      }

      /* Free memory */
      hypre_SeqVectorDestroy(G_temp);
      hypre_SeqVectorDestroy(A_subrow);
      hypre_SeqVectorDestroy(kap_grad);
      hypre_SeqVectorDestroy(A_sub);
      hypre_TFree(kg_pos, HYPRE_MEMORY_HOST);
      hypre_TFree(pattern, HYPRE_MEMORY_HOST);
      hypre_TFree(marker, HYPRE_MEMORY_HOST);
      hypre_TFree(kg_marker, HYPRE_MEMORY_HOST);
   } /* end openmp region */
   HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);

   /* Update local number of nonzeros of G */
   hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAISetupOMPDyn
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAISetupOMPDyn( void               *fsai_vdata,
                       hypre_ParCSRMatrix *A,
                       hypre_ParVector    *f,
                       hypre_ParVector    *u )
{
   HYPRE_UNUSED_VAR(f);
   HYPRE_UNUSED_VAR(u);

   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Complex          *A_a              = hypre_CSRMatrixData(A_diag);
   HYPRE_Int               num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int               num_nnzs_diag_A  = hypre_CSRMatrixNumNonzeros(A_diag);
   HYPRE_Int               avg_nnzrow_diag_A;

   /* Matrix G variables */
   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag;
   HYPRE_Int              *G_i;
   HYPRE_Int              *G_j;
   HYPRE_Complex          *G_a;
   HYPRE_Int              *G_nnzcnt;          /* Array holding number of nonzeros of row G[i,:] */
   HYPRE_Int               max_nnzrow_diag_G; /* Max. number of nonzeros per row in G_diag */
   HYPRE_Int               max_cand_size;     /* Max size of kg_pos */

   /* Local variables */
   HYPRE_Int                i, j, jj;
   char                     msg[512];    /* Warning message */
   HYPRE_Complex           *twspace;     /* shared work space for omp threads */

   /* Initalize some variables */
   avg_nnzrow_diag_A = num_nnzs_diag_A / num_rows_diag_A;
   max_nnzrow_diag_G = max_steps * max_step_size + 1;
   max_cand_size     = avg_nnzrow_diag_A * max_nnzrow_diag_G;

   G_diag = hypre_ParCSRMatrixDiag(G);
   G_a = hypre_CSRMatrixData(G_diag);
   G_i = hypre_CSRMatrixI(G_diag);
   G_j = hypre_CSRMatrixJ(G_diag);
   G_nnzcnt = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

   /* Allocate shared work space array for OpenMP threads */
   twspace = hypre_CTAlloc(HYPRE_Complex, hypre_NumThreads() + 1, HYPRE_MEMORY_HOST);

   /**********************************************************************
   * Start of Adaptive FSAI algorithm
   ***********************************************************************/

   /* Cycle through each of the local rows */
   HYPRE_ANNOTATE_REGION_BEGIN("%s", "MainLoop");
#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel
#endif
   {
      hypre_Vector   *G_temp;        /* Vector holding the values of G[i,:] */
      hypre_Vector   *A_sub;         /* Vector holding the dense submatrix A[P, P] */
      hypre_Vector   *A_subrow;      /* Vector holding A[i, P] */
      hypre_Vector   *kap_grad;      /* Vector holding the Kaporin gradient values */
      HYPRE_Int      *kg_pos;        /* Indices of nonzero entries of kap_grad */
      HYPRE_Int      *kg_marker;     /* Marker array with nonzeros pointing to kg_pos */
      HYPRE_Int      *marker;        /* Marker array with nonzeros pointing to P */
      HYPRE_Int      *pattern;       /* Array holding column indices of G[i,:] */
      HYPRE_Int       patt_size;     /* Number of entries in current pattern */
      HYPRE_Int       patt_size_old; /* Number of entries in previous pattern */
      HYPRE_Int       i, j, k;       /* Loop variables */
      HYPRE_Complex   old_psi;       /* GAG' before k-th interation of aFSAI */
      HYPRE_Complex   new_psi;       /* GAG' after k-th interation of aFSAI */
      HYPRE_Complex   row_scale;     /* Scaling factor for G_temp */
      HYPRE_Complex  *G_temp_data;
      HYPRE_Complex  *A_subrow_data;


      /* Allocate and initialize local vector variables */
      G_temp    = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      A_subrow  = hypre_SeqVectorCreate(max_nnzrow_diag_G);
      kap_grad  = hypre_SeqVectorCreate(max_cand_size);
      A_sub     = hypre_SeqVectorCreate(max_nnzrow_diag_G * max_nnzrow_diag_G);
      pattern   = hypre_CTAlloc(HYPRE_Int, max_nnzrow_diag_G, HYPRE_MEMORY_HOST);
      kg_pos    = hypre_CTAlloc(HYPRE_Int, max_cand_size, HYPRE_MEMORY_HOST);
      kg_marker = hypre_CTAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);
      marker    = hypre_TAlloc(HYPRE_Int, num_rows_diag_A, HYPRE_MEMORY_HOST);

      hypre_SeqVectorInitialize_v2(G_temp, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_subrow, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(kap_grad, HYPRE_MEMORY_HOST);
      hypre_SeqVectorInitialize_v2(A_sub, HYPRE_MEMORY_HOST);
      hypre_Memset(marker, -1, num_rows_diag_A * sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

      /* Setting data variables for vectors */
      G_temp_data   = hypre_VectorData(G_temp);
      A_subrow_data = hypre_VectorData(A_subrow);

#ifdef HYPRE_USING_OPENMP
      #pragma omp for schedule(dynamic)
#endif
      for (i = 0; i < num_rows_diag_A; i++)
      {
         patt_size = 0;

         /* Set old_psi up front so we don't have to compute GAG' twice in the inner for-loop */
         new_psi = old_psi = A_a[A_i[i]];

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

            /* Update sizes */
            hypre_VectorSize(A_sub)    = patt_size * patt_size;
            hypre_VectorSize(A_subrow) = patt_size;
            hypre_VectorSize(G_temp)   = patt_size;

            if (patt_size == patt_size_old)
            {
               new_psi = old_psi;
               break;
            }
            else
            {
               /* Gather A[P, P] and -A[i, P] */
               for (j = 0; j < patt_size; j++)
               {
                  marker[pattern[j]] = j;
               }
               hypre_CSRMatrixExtractDenseMat(A_diag, A_sub, pattern, patt_size, marker);
               hypre_CSRMatrixExtractDenseRow(A_diag, A_subrow, marker, i);

               /* Solve A[P, P] G[i, P]' = -A[i, P] */
               hypre_DenseSPDSystemSolve(A_sub, A_subrow, G_temp);

               /* Determine psi_{k+1} = G_temp[i] * A[P, P] * G_temp[i]' */
               new_psi = A_a[A_i[i]];
               for (j = 0; j < patt_size; j++)
               {
                  new_psi += G_temp_data[j] * A_subrow_data[j];
               }

               /* Check psi reduction */
               if (hypre_cabs(new_psi - old_psi) < hypre_creal(kap_tolerance * old_psi))
               {
                  break;
               }
               else
               {
                  old_psi = new_psi;
               }
            }
         }

         /* Reset marker for building dense linear system */
         for (j = 0; j < patt_size; j++)
         {
            marker[pattern[j]] = -1;
         }

         /* Compute scaling factor */
         if (hypre_creal(new_psi) > 0 && hypre_cimag(new_psi) == 0)
         {
            row_scale = 1.0 / hypre_csqrt(new_psi);
         }
         else
         {
            hypre_sprintf(msg, "Warning: complex scaling factor found in row %d\n", i);
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);

            row_scale = 1.0 / hypre_cabs(A_a[A_i[i]]);
            hypre_VectorSize(G_temp) = patt_size = 0;
         }

         /* Pass values of G_temp into G */
         j = i * max_nnzrow_diag_G;
         G_j[j] = i;
         G_a[j] = row_scale;
         j++;
         for (k = 0; k < patt_size; k++)
         {
            G_j[j] = pattern[k];
            G_a[j++] = row_scale * G_temp_data[k];
            kg_marker[pattern[k]] = 0;
         }
         G_nnzcnt[i] = patt_size + 1;
      } /* omp for schedule(dynamic) */

      /* Free memory */
      hypre_SeqVectorDestroy(G_temp);
      hypre_SeqVectorDestroy(A_subrow);
      hypre_SeqVectorDestroy(kap_grad);
      hypre_SeqVectorDestroy(A_sub);
      hypre_TFree(kg_pos, HYPRE_MEMORY_HOST);
      hypre_TFree(pattern, HYPRE_MEMORY_HOST);
      hypre_TFree(marker, HYPRE_MEMORY_HOST);
      hypre_TFree(kg_marker, HYPRE_MEMORY_HOST);
   } /* end openmp region */
   HYPRE_ANNOTATE_REGION_END("%s", "MainLoop");

   /* Reorder array */
   G_i[0] = 0;
   for (i = 0; i < num_rows_diag_A; i++)
   {
      G_i[i + 1] = G_i[i] + G_nnzcnt[i];
      jj = i * max_nnzrow_diag_G;
      for (j = G_i[i]; j < G_i[i + 1]; j++)
      {
         G_j[j] = G_j[jj];
         G_a[j] = G_a[jj++];
      }
   }

   /* Free memory */
   hypre_TFree(twspace, HYPRE_MEMORY_HOST);
   hypre_TFree(G_nnzcnt, HYPRE_MEMORY_HOST);

   /* Update local number of nonzeros of G */
   hypre_CSRMatrixNumNonzeros(G_diag) = G_i[num_rows_diag_A];

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
   hypre_ParFSAIData       *fsai_data     = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int                max_steps     = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int                max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int                max_nnz_row   = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   HYPRE_Int                algo_type     = hypre_ParFSAIDataAlgoType(fsai_data);
   HYPRE_Int                print_level   = hypre_ParFSAIDataPrintLevel(fsai_data);
   HYPRE_Int                eig_max_iters = hypre_ParFSAIDataEigMaxIters(fsai_data);

   /* ParCSRMatrix A variables */
   MPI_Comm                 comm          = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt             num_rows_A    = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt             num_cols_A    = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt            *row_starts_A  = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_BigInt            *col_starts_A  = hypre_ParCSRMatrixColStarts(A);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix         *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int                num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);

   /* Work vectors */
   hypre_ParVector         *r_work;
   hypre_ParVector         *z_work;

   /* G variables */
   hypre_ParCSRMatrix      *G;
   HYPRE_Int                max_nnzrow_diag_G;   /* Max. number of nonzeros per row in G_diag */
   HYPRE_Int                max_nonzeros_diag_G; /* Max. number of nonzeros in G_diag */

   /* Sanity check */
   if (f && hypre_ParVectorNumVectors(f) > 1)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "FSAI doesn't support multicomponent vectors");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;

   /* Create and initialize work vectors used in the solve phase */
   r_work = hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);
   z_work = hypre_ParVectorCreate(comm, num_rows_A, row_starts_A);

   hypre_ParVectorInitialize(r_work);
   hypre_ParVectorInitialize(z_work);

   hypre_ParFSAIDataRWork(fsai_data) = r_work;
   hypre_ParFSAIDataZWork(fsai_data) = z_work;

   /* Create the matrix G */
   if (algo_type == 1 || algo_type == 2)
   {
      max_nnzrow_diag_G = max_steps * max_step_size + 1;
   }
   else
   {
      max_nnzrow_diag_G = max_nnz_row + 1;
   }
   max_nonzeros_diag_G = num_rows_diag_A * max_nnzrow_diag_G;
   G = hypre_ParCSRMatrixCreate(comm, num_rows_A, num_cols_A,
                                row_starts_A, col_starts_A,
                                0, max_nonzeros_diag_G, 0);
   hypre_ParFSAIDataGmat(fsai_data) = G;

   /* Initialize and compute lower triangular factor G */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_MemoryLocation  memloc_A = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1(memloc_A);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      hypre_FSAISetupDevice(fsai_vdata, A, f, u);
   }
   else
#endif
   {
      /* Initialize matrix */
      hypre_ParCSRMatrixInitialize(G);

      switch (algo_type)
      {
         case 1:
            // TODO: Change name to hypre_FSAISetupAdaptive
            hypre_FSAISetupNative(fsai_vdata, A, f, u);
            break;

         case 2:
            // TODO: Change name to hypre_FSAISetupAdaptiveOMPDynamic
            hypre_FSAISetupOMPDyn(fsai_vdata, A, f, u);
            break;

         default:
            hypre_FSAISetupNative(fsai_vdata, A, f, u);
            break;
      }
   }

   /* Compute G^T */
   G  = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrixTranspose(G, &hypre_ParFSAIDataGTmat(fsai_data), 1);

   /* Update omega if requested */
   if (eig_max_iters)
   {
      hypre_FSAIComputeOmega(fsai_vdata, A);
   }

   /* Print setup info */
   if (print_level == 1)
   {
      hypre_FSAIPrintStats(fsai_data, A);
   }
   else if (print_level > 2)
   {
      char filename[] = "FSAI.out.G.ij";
      hypre_ParCSRMatrixPrintIJ(G, 0, 0, filename);
   }

#if defined (DEBUG_FSAI)
#if !defined (HYPRE_USING_GPU) ||
   (defined (HYPRE_USING_GPU) && defined (HYPRE_USING_UNIFIED_MEMORY))
   hypre_FSAIDumpLocalLSDense(fsai_vdata, "fsai_dense_ls.out", A);
#endif
#endif

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIPrintStats
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIPrintStats( void *fsai_vdata,
                      hypre_ParCSRMatrix *A )
{
   /* Data structure variables */
   hypre_ParFSAIData      *fsai_data        = (hypre_ParFSAIData*) fsai_vdata;
   HYPRE_Int               algo_type        = hypre_ParFSAIDataAlgoType(fsai_data);
   HYPRE_Int               local_solve_type = hypre_ParFSAIDataLocalSolveType(fsai_data);
   HYPRE_Real              kap_tolerance    = hypre_ParFSAIDataKapTolerance(fsai_data);
   HYPRE_Int               max_steps        = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size    = hypre_ParFSAIDataMaxStepSize(fsai_data);
   HYPRE_Int               max_nnz_row      = hypre_ParFSAIDataMaxNnzRow(fsai_data);
   HYPRE_Int               num_levels       = hypre_ParFSAIDataNumLevels(fsai_data);
   HYPRE_Real              threshold        = hypre_ParFSAIDataThreshold(fsai_data);
   HYPRE_Int               eig_max_iters    = hypre_ParFSAIDataEigMaxIters(fsai_data);
   HYPRE_Real              density;

   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);

   /* Local variables */
   HYPRE_Int               nprocs;
   HYPRE_Int               my_id;

   hypre_MPI_Comm_size(hypre_ParCSRMatrixComm(A), &nprocs);
   hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);

   /* Compute density */
   hypre_ParCSRMatrixSetDNumNonzeros(G);
   hypre_ParCSRMatrixSetDNumNonzeros(A);
   density = hypre_ParCSRMatrixDNumNonzeros(G) /
             hypre_ParCSRMatrixDNumNonzeros(A);
   hypre_ParFSAIDataDensity(fsai_data) = density;

   if (!my_id)
   {
      hypre_printf("*************************\n");
      hypre_printf("* HYPRE FSAI Setup Info *\n");
      hypre_printf("*************************\n\n");

      hypre_printf("+---------------------------+\n");
      hypre_printf("| No. MPI tasks:     %6d |\n", nprocs);
      hypre_printf("| No. threads:       %6d |\n", hypre_NumThreads());
      hypre_printf("| Algorithm type:    %6d |\n", algo_type);
      hypre_printf("| Local solve type:  %6d |\n", local_solve_type);
      if (algo_type == 1 || algo_type == 2)
      {
         hypre_printf("| Max no. steps:     %6d |\n", max_steps);
         hypre_printf("| Max step size:     %6d |\n", max_step_size);
         hypre_printf("| Kap grad tol:    %8.1e |\n", kap_tolerance);
      }
      else
      {
         hypre_printf("| Max nnz. row:      %6d |\n", max_nnz_row);
         hypre_printf("| Number of levels:  %6d |\n", num_levels);
         hypre_printf("| Threshold:       %8.1e |\n", threshold);
      }
      hypre_printf("| Prec. density:   %8.3f |\n", density);
      hypre_printf("| Eig max iters:     %6d |\n", eig_max_iters);
      hypre_printf("| Omega factor:    %8.3f |\n", hypre_ParFSAIDataOmega(fsai_data));
      hypre_printf("+---------------------------+\n");

      hypre_printf("\n\n");
   }

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
hypre_FSAIComputeOmega( void               *fsai_vdata,
                        hypre_ParCSRMatrix *A )
{
   hypre_ParFSAIData    *fsai_data       = (hypre_ParFSAIData*) fsai_vdata;
   hypre_ParCSRMatrix   *G               = hypre_ParFSAIDataGmat(fsai_data);
   hypre_ParCSRMatrix   *GT              = hypre_ParFSAIDataGTmat(fsai_data);
   hypre_ParVector      *r_work          = hypre_ParFSAIDataRWork(fsai_data);
   hypre_ParVector      *z_work          = hypre_ParFSAIDataZWork(fsai_data);
   HYPRE_Int             eig_max_iters   = hypre_ParFSAIDataEigMaxIters(fsai_data);
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   hypre_ParVector      *eigvec;
   hypre_ParVector      *eigvec_old;

   HYPRE_Int             i;
   HYPRE_Real            norm, invnorm, lambda, omega;

   eigvec_old = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                      hypre_ParCSRMatrixGlobalNumRows(A),
                                      hypre_ParCSRMatrixRowStarts(A));
   eigvec = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                  hypre_ParCSRMatrixGlobalNumRows(A),
                                  hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(eigvec, memory_location);
   hypre_ParVectorInitialize_v2(eigvec_old, memory_location);

#if defined(HYPRE_USING_GPU)
   /* Make random number generation faster on GPUs */
   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      hypre_Vector  *eigvec_local = hypre_ParVectorLocalVector(eigvec);
      HYPRE_Complex *eigvec_data  = hypre_VectorData(eigvec_local);
      HYPRE_Int      eigvec_size  = hypre_VectorSize(eigvec_local);

      hypre_CurandUniform(eigvec_size, eigvec_data, 0, 0, 0, 0);
   }
   else
#endif
   {
      hypre_ParVectorSetRandomValues(eigvec, 256);
   }

   /* Power method iteration */
   for (i = 0; i < eig_max_iters; i++)
   {
      norm = hypre_ParVectorInnerProd(eigvec, eigvec);
      invnorm = 1.0 / hypre_sqrt(norm);
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
   lambda = hypre_sqrt(norm);

   /* Check lambda */
   if (lambda < HYPRE_REAL_EPSILON)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Found small lambda. Reseting it to one!");
      lambda = 1.0;
   }

   /* Free memory */
   hypre_ParVectorDestroy(eigvec_old);
   hypre_ParVectorDestroy(eigvec);

   /* Update omega */
   omega = 1.0 / lambda;
   hypre_FSAISetOmega(fsai_vdata, omega);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_FSAIDumpLocalLSDense
 *
 * Dump local linear systems to file. Matrices are written in dense format.
 * This functions serves for debugging.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FSAIDumpLocalLSDense( void               *fsai_vdata,
                            const char         *filename,
                            hypre_ParCSRMatrix *A )
{
   hypre_ParFSAIData      *fsai_data = (hypre_ParFSAIData*) fsai_vdata;

   /* Data structure variables */
   MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
   HYPRE_Int               max_steps = hypre_ParFSAIDataMaxSteps(fsai_data);
   HYPRE_Int               max_step_size = hypre_ParFSAIDataMaxStepSize(fsai_data);
   hypre_ParCSRMatrix     *G = hypre_ParFSAIDataGmat(fsai_data);
   hypre_CSRMatrix        *G_diag = hypre_ParCSRMatrixDiag(G);
   HYPRE_Int              *G_i = hypre_CSRMatrixI(G_diag);
   HYPRE_Int              *G_j = hypre_CSRMatrixJ(G_diag);
   HYPRE_Int               num_rows_diag_G = hypre_CSRMatrixNumRows(G_diag);

   /* CSRMatrix A_diag variables */
   hypre_CSRMatrix        *A_diag           = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int              *A_i              = hypre_CSRMatrixI(A_diag);
   HYPRE_Int              *A_j              = hypre_CSRMatrixJ(A_diag);
   HYPRE_Complex          *A_a              = hypre_CSRMatrixData(A_diag);

   FILE                   *fp;
   char                    new_filename[1024];
   HYPRE_Int               myid;
   HYPRE_Int               i, j, k, m, n;
   HYPRE_Int               ii, jj;
   HYPRE_Int               nnz, col, index;
   HYPRE_Int              *indices;
   HYPRE_Int              *marker;
   HYPRE_Real             *data;
   HYPRE_Int               data_size;
   HYPRE_Real              density;
   HYPRE_Int               width = 20; //6
   HYPRE_Int               prec  = 16; //2

   hypre_MPI_Comm_rank(comm, &myid);
   hypre_sprintf(new_filename, "%s.%05d", filename, myid);
   if ((fp = fopen(new_filename, "w")) == NULL)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Error: can't open output file %s\n");
      return hypre_error_flag;
   }

   /* Allocate memory */
   data_size = (max_steps * max_step_size) *
               (max_steps * max_step_size + 1);
   indices = hypre_CTAlloc(HYPRE_Int, data_size, HYPRE_MEMORY_HOST);
   data    = hypre_CTAlloc(HYPRE_Real, data_size, HYPRE_MEMORY_HOST);
   marker  = hypre_TAlloc(HYPRE_Int, num_rows_diag_G, HYPRE_MEMORY_HOST);
   hypre_Memset(marker, -1, num_rows_diag_G * sizeof(HYPRE_Int), HYPRE_MEMORY_HOST);

   /* Write header info */
   hypre_fprintf(fp, "num_linear_sys = %d\n", num_rows_diag_G);
   hypre_fprintf(fp, "max_data_size = %d\n", data_size);
   hypre_fprintf(fp, "max_num_steps = %d\n", hypre_ParFSAIDataMaxSteps(fsai_data));
   hypre_fprintf(fp, "max_step_size = %d\n", hypre_ParFSAIDataMaxStepSize(fsai_data));
   hypre_fprintf(fp, "max_step_size = %g\n", hypre_ParFSAIDataKapTolerance(fsai_data));
   hypre_fprintf(fp, "algo_type = %d\n\n", hypre_ParFSAIDataAlgoType(fsai_data));

   /* Write local full linear systems */
   for (i = 0; i < num_rows_diag_G; i++)
   {
      /* Build marker array */
      n = G_i[i + 1] - G_i[i] - 1;
      m = n + 1;
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         marker[G_j[j]] = j - G_i[i] - 1;
      }

      /* Gather matrix coefficients */
      nnz = 0;
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         for (k = A_i[G_j[j]]; k < A_i[G_j[j] + 1]; k++)
         {
            if ((col = marker[A_j[k]]) >= 0)
            {
               /* Add A(i,j) entry */
               index = (j - G_i[i] - 1) * n + col;
               data[index] = A_a[k];
               indices[nnz] = index;
               nnz++;
            }
         }
      }
      density = (n > 0) ? (HYPRE_Real) nnz / (n * n) : 0.0;

      /* Gather RHS coefficients */
      for (j = A_i[i]; j < A_i[i + 1]; j++)
      {
         if ((col = marker[A_j[j]]) >= 0)
         {
            index = (m - 1) * n + col;
            data[index] = A_a[j];
            indices[nnz] = index;
            nnz++;
         }
      }

      /* Write coefficients to file */
      hypre_fprintf(fp, "id = %d, (m, n) = (%d, %d), rho = %.3f\n", i, m, n, density);
      for (ii = 0; ii < n; ii++)
      {
         for (jj = 0; jj < n; jj++)
         {
            hypre_fprintf(fp, "%*.*f ", width, prec, data[ii * n + jj]);
         }
         hypre_fprintf(fp, "\n");
      }
      for (jj = 0; jj < n; jj++)
      {
         hypre_fprintf(fp, "%*.*f ", width, prec, data[ii * n + jj]);
      }
      hypre_fprintf(fp, "\n");


      /* Reset work arrays */
      for (j = (G_i[i] + 1); j < G_i[i + 1]; j++)
      {
         marker[G_j[j]] = -1;
      }

      for (k = 0; k < nnz; k++)
      {
         data[indices[k]] = 0.0;
      }
   }

   /* Close stream */
   fclose(fp);

   /* Free memory */
   hypre_TFree(indices, HYPRE_MEMORY_HOST);
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   hypre_TFree(data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
