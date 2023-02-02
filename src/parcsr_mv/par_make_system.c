/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

/* This routine takes as input 2 parcsr matrices L1 and L2 (and the
 corresponding initial guess and rhs), and creates the system M*[L1 0;
 0 L2] x = [ x1; x2] b = [b1; b2].  The entries of M are M = [m11 m12;
 m21 m22] and should be given as M_vals = [m11 m12 m21 m22]; So we
 return A = [ m11L1 m12L2; m21L1 m22L2]

 We assume that L1 and L2 are the same size, both square, and
 partitioned the same.  We also assume that m11 and m22 are nonzero.

 To Do:  This function could be easily extended to create a system
         with 3 or even N unknowns
*/


HYPRE_ParCSR_System_Problem *
HYPRE_Generate2DSystem(HYPRE_ParCSRMatrix H_L1, HYPRE_ParCSRMatrix H_L2,
                       HYPRE_ParVector H_b1, HYPRE_ParVector H_b2,
                       HYPRE_ParVector H_x1, HYPRE_ParVector H_x2,
                       HYPRE_Complex* M_vals)
{

   HYPRE_ParCSR_System_Problem  *sys_prob;

   hypre_ParCSRMatrix *A;
   hypre_ParCSRMatrix *L1 = (hypre_ParCSRMatrix*) H_L1;
   hypre_ParCSRMatrix *L2 = (hypre_ParCSRMatrix*) H_L2;
   hypre_CSRMatrix *A_diag;
   hypre_CSRMatrix *A_offd;

   hypre_ParVector *x, *b;

   hypre_ParVector *b1 = (hypre_ParVector*) H_b1;
   hypre_ParVector *b2 = (hypre_ParVector*) H_b2;

   hypre_ParVector *x1 = (hypre_ParVector*) H_x1;
   hypre_ParVector *x2 = (hypre_ParVector*) H_x2;

   HYPRE_Complex *b_data, *x_data;

   HYPRE_Int dim = 2;

   HYPRE_Complex m11, m12, m21, m22;

   MPI_Comm comm = hypre_ParCSRMatrixComm(L1);
   HYPRE_BigInt L_n = hypre_ParCSRMatrixGlobalNumRows(L1);
   HYPRE_BigInt n;
   HYPRE_Int num_procs, i;

   HYPRE_BigInt *L_row_starts = hypre_ParCSRMatrixRowStarts(L1);

   hypre_CSRMatrix *L1_diag = hypre_ParCSRMatrixDiag(L1);
   hypre_CSRMatrix *L1_offd = hypre_ParCSRMatrixOffd(L1);

   hypre_CSRMatrix *L2_diag = hypre_ParCSRMatrixDiag(L2);
   hypre_CSRMatrix *L2_offd = hypre_ParCSRMatrixOffd(L2);

   HYPRE_Complex   *L1_diag_data = hypre_CSRMatrixData(L1_diag);
   HYPRE_Int             *L1_diag_i = hypre_CSRMatrixI(L1_diag);
   HYPRE_Int             *L1_diag_j = hypre_CSRMatrixJ(L1_diag);

   HYPRE_Complex   *L2_diag_data = hypre_CSRMatrixData(L2_diag);
   HYPRE_Int             *L2_diag_i = hypre_CSRMatrixI(L2_diag);
   HYPRE_Int             *L2_diag_j = hypre_CSRMatrixJ(L2_diag);

   HYPRE_Complex   *L1_offd_data = hypre_CSRMatrixData(L1_offd);
   HYPRE_Int             *L1_offd_i = hypre_CSRMatrixI(L1_offd);
   HYPRE_Int             *L1_offd_j = hypre_CSRMatrixJ(L1_offd);

   HYPRE_Complex   *L2_offd_data = hypre_CSRMatrixData(L2_offd);
   HYPRE_Int             *L2_offd_i = hypre_CSRMatrixI(L2_offd);
   HYPRE_Int             *L2_offd_j = hypre_CSRMatrixJ(L2_offd);

   HYPRE_Int L1_num_cols_offd = hypre_CSRMatrixNumCols(L1_offd);
   HYPRE_Int L2_num_cols_offd = hypre_CSRMatrixNumCols(L2_offd);

   HYPRE_Int L1_nnz_diag = hypre_CSRMatrixNumNonzeros(L1_diag);
   HYPRE_Int L1_nnz_offd = hypre_CSRMatrixNumNonzeros(L1_offd);

   HYPRE_Int L2_nnz_diag = hypre_CSRMatrixNumNonzeros(L2_diag);
   HYPRE_Int L2_nnz_offd = hypre_CSRMatrixNumNonzeros(L2_offd);

   HYPRE_BigInt *L1_col_map_offd =  hypre_ParCSRMatrixColMapOffd(L1);
   HYPRE_BigInt *L2_col_map_offd =  hypre_ParCSRMatrixColMapOffd(L2);

   HYPRE_BigInt A_row_starts[2];
   HYPRE_BigInt A_col_starts[2];

   HYPRE_BigInt *A_col_map_offd = NULL;

   HYPRE_Int A_nnz_diag, A_nnz_offd, A_num_cols_offd;

   HYPRE_Int *A_diag_i, *A_diag_j, *A_offd_i, *A_offd_j;
   HYPRE_Complex *A_diag_data, *A_offd_data;

   /* initialize stuff */
   m11 = M_vals[0];
   m12 = M_vals[1];
   m21 = M_vals[2];
   m22 = M_vals[3];

   hypre_MPI_Comm_size(comm, &num_procs);

   sys_prob = hypre_CTAlloc(HYPRE_ParCSR_System_Problem,  1, HYPRE_MEMORY_HOST);

   /* global number of variables */
   n = L_n * (HYPRE_BigInt)dim;

   /* global row/col starts */
   for (i = 0; i < 2; i++)
   {
      A_row_starts[i] = L_row_starts[i] * (HYPRE_BigInt)dim;
      A_col_starts[i] = L_row_starts[i] * (HYPRE_BigInt)dim;
   }

   /***** first we will do the diag part ******/
   {
      HYPRE_Int L_num_rows, A_num_rows;
      HYPRE_Int num1, num2, A_j_count;
      HYPRE_Int k, L1_j_count, L2_j_count;


      L_num_rows = hypre_CSRMatrixNumRows(L1_diag);
      A_num_rows = L_num_rows * dim;

      /* assume m11 and m22 are nonzero */
      A_nnz_diag = L1_nnz_diag + L2_nnz_diag;
      if (m12) { A_nnz_diag +=  L2_nnz_diag; }
      if (m21) { A_nnz_diag +=  L1_nnz_diag; }

      A_diag_i    = hypre_CTAlloc(HYPRE_Int,  A_num_rows + 1, HYPRE_MEMORY_HOST);
      A_diag_j    = hypre_CTAlloc(HYPRE_Int,  A_nnz_diag, HYPRE_MEMORY_HOST);
      A_diag_data = hypre_CTAlloc(HYPRE_Complex,  A_nnz_diag, HYPRE_MEMORY_HOST);

      A_diag_i[0] = 0;

      A_j_count = 0;
      L1_j_count = 0;
      L2_j_count = 0;

      for (i = 0; i < L_num_rows; i++)
      {
         num1 = L1_diag_i[i + 1] - L1_diag_i[i];
         num2 = (L2_diag_i[i + 1] - L2_diag_i[i]);

         /* unknown 1*/
         if (m12 == 0.0)
         {
            A_diag_i[i * 2 + 1] = num1 + A_diag_i[i * 2];

            for (k = 0; k < num1; k++)
            {
               A_diag_j[A_j_count + k] = dim * L1_diag_j[L1_j_count + k];
               A_diag_data[A_j_count + k] = m11 * L1_diag_data[L1_j_count + k];
            }
            A_j_count += num1;
         }
         else /* m12 is nonzero */
         {
            A_diag_i[i * 2 + 1] = num1 + num2 + A_diag_i[i * 2];

            for (k = 0; k < num1; k++)
            {
               A_diag_j[A_j_count + k] = dim * L1_diag_j[L1_j_count + k];
               A_diag_data[A_j_count + k] = m11 * L1_diag_data[L1_j_count + k];
            }
            A_j_count += num1;

            for (k = 0; k < num2; k++)
            {
               A_diag_j[A_j_count + k] = 1 + dim * L2_diag_j[L2_j_count + k];
               A_diag_data[A_j_count + k] = m12 * L2_diag_data[L2_j_count + k];
            }
            A_j_count += num2;

            /* don't increment the j_count for L1 and L2 until
               after doing the next unknown */

         } /* end unknown 1 */
         /* unknown 2*/
         if (m21 == 0.0)
         {
            A_diag_i[i * 2 + 2] = num2 + A_diag_i[i * 2 + 1];

            for (k = 0; k < num2; k++)
            {
               A_diag_j[A_j_count + k] = 1 + dim * L2_diag_j[L2_j_count + k];
               A_diag_data[A_j_count + k] = m22 * L2_diag_data[L2_j_count + k];
            }
            A_j_count += num2;
         }
         else /* m21 is nonzero */
         {

            A_diag_i[i * 2 + 2] = num1 + num2 + A_diag_i[i * 2 + 1];

            for (k = 0; k < num2; k++)
            {
               A_diag_j[A_j_count + k] = 1 + dim * L2_diag_j[L2_j_count + k];
               A_diag_data[A_j_count + k] = m22 * L2_diag_data[L2_j_count + k];
            }
            A_j_count += num2;

            for (k = 0; k < num1; k++)
            {
               A_diag_j[A_j_count + k] = dim * L1_diag_j[L1_j_count + k];
               A_diag_data[A_j_count + k] = m21 * L1_diag_data[L1_j_count + k];
            }
            A_j_count += num1;


         } /* end unknown 2 */

         L1_j_count += num1;
         L2_j_count += num2;


      } /* end of for each row loop....*/
   }/* end of diag part of A*/


   /**** off-diag part of A ******/
   {
      HYPRE_Int L_num_rows, A_num_rows;
      HYPRE_Int *L1_map_to_new, *L2_map_to_new;
      HYPRE_BigInt ent1, ent2;
      HYPRE_Int tmp_i, num1, num2;
      HYPRE_Int L1_map_count, L2_map_count;
      HYPRE_Int k, L1_j_count, L2_j_count, A_j_count;

      L_num_rows = hypre_CSRMatrixNumRows(L1_offd);
      A_num_rows = L_num_rows * dim;

      A_nnz_offd = L1_nnz_offd + L2_nnz_offd;
      if (m12) { A_nnz_offd +=  L2_nnz_offd; }
      if (m21) { A_nnz_offd +=  L1_nnz_offd; }

      A_num_cols_offd = L1_num_cols_offd + L2_num_cols_offd;

      A_offd_i    = hypre_CTAlloc(HYPRE_Int,  A_num_rows + 1, HYPRE_MEMORY_HOST);
      A_offd_j    = hypre_CTAlloc(HYPRE_Int,  A_nnz_offd, HYPRE_MEMORY_HOST);
      A_offd_data = hypre_CTAlloc(HYPRE_Complex,  A_nnz_offd, HYPRE_MEMORY_HOST);


      A_col_map_offd =  hypre_CTAlloc(HYPRE_BigInt,  A_num_cols_offd, HYPRE_MEMORY_HOST);

      L1_map_to_new = hypre_CTAlloc(HYPRE_Int,  L1_num_cols_offd, HYPRE_MEMORY_HOST);
      L2_map_to_new = hypre_CTAlloc(HYPRE_Int,  L2_num_cols_offd, HYPRE_MEMORY_HOST);


      /* For offd, the j index is a local numbering and then the
         col_map is global - so first we will adjust the numbering of
         the 2 col maps and merge the two col. maps - these need to
         be in ascending order */

      L1_map_count = 0;
      L2_map_count = 0;
      for (i = 0; i < A_num_cols_offd; i++)
      {

         if (L1_map_count < L1_num_cols_offd && L2_map_count < L2_num_cols_offd)
         {
            ent1 = L1_col_map_offd[L1_map_count] * 2;
            ent2 = L2_col_map_offd[L2_map_count] * 2 + 1;
            if (ent1 < ent2)
            {
               A_col_map_offd[i] = ent1;
               L1_map_to_new[L1_map_count++] = i;
            }
            else
            {
               A_col_map_offd[i] = ent2;
               L2_map_to_new[L2_map_count++] = i;
            }
         }
         else if (L1_map_count >= L1_num_cols_offd)
         {
            ent2 = L2_col_map_offd[L2_map_count] * 2 + 1;
            A_col_map_offd[i] = ent2;
            L2_map_to_new[L2_map_count++] = i;
         }
         else if (L2_map_count >= L2_num_cols_offd)
         {
            ent1 = L1_col_map_offd[L1_map_count] * 2;
            A_col_map_offd[i] = ent1;
            L1_map_to_new[L1_map_count++] = i;
         }
         else
         {
            hypre_error(HYPRE_ERROR_GENERIC);
         }


      }

      /* now go through the rows */

      A_j_count = 0;
      L1_j_count = 0;
      L2_j_count = 0;

      A_offd_i[0] = 0;
      for (i = 0; i < L_num_rows; i++)
      {
         num1 = L1_offd_i[i + 1] - L1_offd_i[i];
         num2 = (L2_offd_i[i + 1] - L2_offd_i[i]);

         /* unknown 1*/
         if (m12 == 0.0)
         {
            A_offd_i[i * 2 + 1] = num1 + A_offd_i[i * 2];

            for (k = 0; k < num1; k++)
            {
               tmp_i = L1_offd_j[L1_j_count + k];
               A_offd_j[A_j_count + k] = L1_map_to_new[tmp_i];
               A_offd_data[A_j_count + k] = m11 * L1_offd_data[L1_j_count + k];
            }
            A_j_count += num1;


         }
         else /* m12 is nonzero */
         {
            A_offd_i[i * 2 + 1] = num1 + num2 + A_offd_i[i * 2];

            for (k = 0; k < num1; k++)
            {
               tmp_i = L1_offd_j[L1_j_count + k];
               A_offd_j[A_j_count + k] = L1_map_to_new[tmp_i];
               A_offd_data[A_j_count + k] = m11 * L1_offd_data[L1_j_count + k];
            }
            A_j_count += num1;

            for (k = 0; k < num2; k++)
            {
               tmp_i = L2_offd_j[L2_j_count + k];
               A_offd_j[A_j_count + k] =  L2_map_to_new[tmp_i];
               A_offd_data[A_j_count + k] = m12 * L2_offd_data[L2_j_count + k];
            }
            A_j_count += num2;

         } /* end unknown 1 */
         /* unknown 2*/
         if (m21 == 0.0)
         {
            A_offd_i[i * 2 + 2] = num2 + A_offd_i[i * 2 + 1];

            for (k = 0; k < num2; k++)
            {
               tmp_i = L2_offd_j[L2_j_count + k];
               A_offd_j[A_j_count + k] =  L2_map_to_new[tmp_i];
               A_offd_data[A_j_count + k] = m22 * L2_offd_data[L2_j_count + k];
            }
            A_j_count += num2;
         }
         else /* m21 is nonzero */
         {

            A_offd_i[i * 2 + 2] = num1 + num2 + A_offd_i[i * 2 + 1];

            for (k = 0; k < num2; k++)
            {
               tmp_i = L2_offd_j[L2_j_count + k];
               A_offd_j[A_j_count + k] =  L2_map_to_new[tmp_i];
               A_offd_data[A_j_count + k] = m22 * L2_offd_data[L2_j_count + k];
            }
            A_j_count += num2;

            for (k = 0; k < num1; k++)
            {
               tmp_i = L1_offd_j[L1_j_count + k];
               A_offd_j[A_j_count + k] = L1_map_to_new[tmp_i];
               A_offd_data[A_j_count + k] = m21 * L1_offd_data[L1_j_count + k];
            }
            A_j_count += num1;


         } /* end unknown 2 */

         L1_j_count += num1;
         L2_j_count += num2;


      } /* end of for each row loop....*/


      hypre_TFree(L1_map_to_new, HYPRE_MEMORY_HOST);
      hypre_TFree(L2_map_to_new, HYPRE_MEMORY_HOST);



   } /* end of offd part */

   /* create A*/
   {

      A = hypre_ParCSRMatrixCreate(comm, n, n,
                                   A_row_starts, A_col_starts, A_num_cols_offd,
                                   A_nnz_diag, A_nnz_offd);

      A_diag = hypre_ParCSRMatrixDiag(A);
      hypre_CSRMatrixData(A_diag) = A_diag_data;
      hypre_CSRMatrixI(A_diag) = A_diag_i;
      hypre_CSRMatrixJ(A_diag) = A_diag_j;

      A_offd = hypre_ParCSRMatrixOffd(A);
      hypre_CSRMatrixData(A_offd) = A_offd_data;
      hypre_CSRMatrixI(A_offd) = A_offd_i;
      hypre_CSRMatrixJ(A_offd) = A_offd_j;

      hypre_ParCSRMatrixColMapOffd(A) = A_col_map_offd;

      hypre_ParCSRMatrixSetNumNonzeros(A);


   }

   /* create b */
   {

      hypre_Vector *b1_local = hypre_ParVectorLocalVector(b1);
      hypre_Vector *b2_local = hypre_ParVectorLocalVector(b2);
      HYPRE_Int      size   = hypre_VectorSize(b1_local);
      HYPRE_Complex  *b1_data = hypre_VectorData(b1_local);
      HYPRE_Complex  *b2_data = hypre_VectorData(b2_local);

      b_data = hypre_CTAlloc(HYPRE_Complex,  size * 2, HYPRE_MEMORY_HOST);

      for (i = 0; i < size; i++)
      {
         b_data[i * 2] = b1_data[i];
         b_data[i * 2 + 1] = b2_data[i];
      }

      b = hypre_ParVectorCreate( comm, n, A_row_starts);
      hypre_ParVectorInitialize(b);

      hypre_TFree(hypre_VectorData(hypre_ParVectorLocalVector(b)), HYPRE_MEMORY_HOST);
      hypre_VectorData(hypre_ParVectorLocalVector(b)) = b_data;

      hypre_ParVectorSetDataOwner(b, 1);
   }

   /* create x */
   {
      hypre_Vector *x1_local = hypre_ParVectorLocalVector(x1);
      hypre_Vector *x2_local = hypre_ParVectorLocalVector(x2);
      HYPRE_Int      size   = hypre_VectorSize(x1_local);
      HYPRE_Complex  *x1_data = hypre_VectorData(x1_local);
      HYPRE_Complex  *x2_data = hypre_VectorData(x2_local);

      x_data = hypre_CTAlloc(HYPRE_Complex,  size * 2, HYPRE_MEMORY_HOST);

      for (i = 0; i < size; i++)
      {
         x_data[i * 2] = x1_data[i];
         x_data[i * 2 + 1] = x2_data[i];
      }

      x = hypre_ParVectorCreate( comm, n, A_row_starts);
      hypre_ParVectorInitialize(x);

      hypre_TFree(hypre_VectorData(hypre_ParVectorLocalVector(x)), HYPRE_MEMORY_HOST);
      hypre_VectorData(hypre_ParVectorLocalVector(x)) = x_data;

      hypre_ParVectorSetDataOwner(x, 1);
   }

   sys_prob->A = A;
   sys_prob->x = x;
   sys_prob->b = b;

   return sys_prob;
}


HYPRE_Int
HYPRE_Destroy2DSystem( HYPRE_ParCSR_System_Problem  *sys_prob)
{
   hypre_ParCSRMatrixDestroy(sys_prob->A);
   hypre_ParVectorDestroy(sys_prob->b);
   hypre_ParVectorDestroy(sys_prob->x);

   hypre_TFree(sys_prob, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}
