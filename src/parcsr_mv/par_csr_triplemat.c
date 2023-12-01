/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_mv.h"

#include "_hypre_utilities.h"
#include "../parcsr_mv/_hypre_parcsr_mv.h"

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatMatHost
 *
 * Host implementation of hypre_ParCSRMatMat (C = A * B)
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatMatHost( hypre_ParCSRMatrix  *A,
                        hypre_ParCSRMatrix  *B )
{
   MPI_Comm         comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_BigInt    *row_starts_A = hypre_ParCSRMatrixRowStarts(A);
   HYPRE_Int        num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int        num_rows_diag_A = hypre_CSRMatrixNumRows(A_diag);

   hypre_CSRMatrix *B_diag = hypre_ParCSRMatrixDiag(B);

   hypre_CSRMatrix *B_offd = hypre_ParCSRMatrixOffd(B);
   HYPRE_BigInt    *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);

   HYPRE_BigInt     first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);
   HYPRE_BigInt     last_col_diag_B;
   HYPRE_BigInt    *col_starts_B = hypre_ParCSRMatrixColStarts(B);
   HYPRE_Int        num_rows_diag_B = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int        num_cols_diag_B = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int        num_cols_offd_B = hypre_CSRMatrixNumCols(B_offd);

   hypre_ParCSRMatrix *C;
   HYPRE_BigInt    *col_map_offd_C = NULL;
   HYPRE_Int       *map_B_to_C = NULL;

   hypre_CSRMatrix *C_diag = NULL;

   hypre_CSRMatrix *C_offd = NULL;

   HYPRE_Int        num_cols_offd_C = 0;

   hypre_CSRMatrix *Bs_ext;

   hypre_CSRMatrix *Bext_diag;

   hypre_CSRMatrix *Bext_offd;

   hypre_CSRMatrix *AB_diag;
   hypre_CSRMatrix *AB_offd;
   HYPRE_Int        AB_offd_num_nonzeros;
   HYPRE_Int       *AB_offd_j;
   hypre_CSRMatrix *ABext_diag;
   hypre_CSRMatrix *ABext_offd;

   HYPRE_BigInt     n_rows_A, n_cols_A;
   HYPRE_BigInt     n_rows_B, n_cols_B;
   HYPRE_Int        cnt, i;
   HYPRE_Int        num_procs;
   HYPRE_Int        my_id;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   if (n_cols_A != n_rows_B || num_cols_diag_A != num_rows_diag_B)
   {
      hypre_error_in_arg(1);
      hypre_printf(" Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /*-----------------------------------------------------------------------
    *  Extract B_ext, i.e. portion of B that is stored on neighbor procs
    *  and needed locally for matrix matrix product
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);
   last_col_diag_B = first_col_diag_B + num_cols_diag_B - 1;

   if (num_procs > 1)
   {
      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Bs_ext = hypre_ParCSRMatrixExtractBExt(B, A, 1); /* contains communication
                                                          which should be explicitly included to allow for overlap */

      hypre_CSRMatrixSplit(Bs_ext, first_col_diag_B, last_col_diag_B, num_cols_offd_B, col_map_offd_B,
                           &num_cols_offd_C, &col_map_offd_C, &Bext_diag, &Bext_offd);

      hypre_CSRMatrixDestroy(Bs_ext);

      /* These are local and could be overlapped with communication */
      AB_diag = hypre_CSRMatrixMultiplyHost(A_diag, B_diag);
      AB_offd = hypre_CSRMatrixMultiplyHost(A_diag, B_offd);

      /* These require data from other processes */
      ABext_diag = hypre_CSRMatrixMultiplyHost(A_offd, Bext_diag);
      ABext_offd = hypre_CSRMatrixMultiplyHost(A_offd, Bext_offd);

      hypre_CSRMatrixDestroy(Bext_diag);
      hypre_CSRMatrixDestroy(Bext_offd);

      if (num_cols_offd_B)
      {
         map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B)
               {
                  break;
               }
            }
         }
      }
      AB_offd_num_nonzeros = hypre_CSRMatrixNumNonzeros(AB_offd);
      AB_offd_j = hypre_CSRMatrixJ(AB_offd);
      for (i = 0; i < AB_offd_num_nonzeros; i++)
      {
         AB_offd_j[i] = map_B_to_C[AB_offd_j[i]];
      }

      if (num_cols_offd_B)
      {
         hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST);
      }

      hypre_CSRMatrixNumCols(AB_diag) = num_cols_diag_B;
      hypre_CSRMatrixNumCols(ABext_diag) = num_cols_diag_B;
      hypre_CSRMatrixNumCols(AB_offd) = num_cols_offd_C;
      hypre_CSRMatrixNumCols(ABext_offd) = num_cols_offd_C;
      C_diag = hypre_CSRMatrixAdd(1.0, AB_diag, 1.0, ABext_diag);
      C_offd = hypre_CSRMatrixAdd(1.0, AB_offd, 1.0, ABext_offd);

      hypre_CSRMatrixDestroy(AB_diag);
      hypre_CSRMatrixDestroy(ABext_diag);
      hypre_CSRMatrixDestroy(AB_offd);
      hypre_CSRMatrixDestroy(ABext_offd);
   }
   else
   {
      C_diag = hypre_CSRMatrixMultiplyHost(A_diag, B_diag);
      C_offd = hypre_CSRMatrixCreate(num_rows_diag_A, 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, hypre_CSRMatrixMemoryLocation(C_diag));
   }

   /*-----------------------------------------------------------------------
    *  Allocate C_diag_data and C_diag_j arrays.
    *  Allocate C_offd_data and C_offd_j arrays.
    *-----------------------------------------------------------------------*/

   C = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_B, row_starts_A,
                                col_starts_B, num_cols_offd_C,
                                C_diag->num_nonzeros, C_offd->num_nonzeros);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   if (num_cols_offd_C)
   {
      hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;
   }

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatMat
 *
 * Computes C = A*B
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatMat( hypre_ParCSRMatrix  *A,
                    hypre_ParCSRMatrix  *B )
{
   hypre_ParCSRMatrix *C = NULL;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("Mat-Mat");

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                                                      hypre_ParCSRMatrixMemoryLocation(B) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_ParCSRMatMatDevice(A, B);
   }
   else
#endif
   {
      C = hypre_ParCSRMatMatHost(A, B);
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRTMatMatKTHost
 *
 * Host implementation of hypre_ParCSRTMatMatKT (C = A^T * B)
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRTMatMatKTHost( hypre_ParCSRMatrix  *A,
                           hypre_ParCSRMatrix  *B,
                           HYPRE_Int            keep_transpose)
{
   MPI_Comm             comm       = hypre_ParCSRMatrixComm(A);
   hypre_ParCSRCommPkg *comm_pkg_A = NULL;

   hypre_CSRMatrix     *A_diag  = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix     *A_offd  = hypre_ParCSRMatrixOffd(A);
   hypre_CSRMatrix     *B_diag  = hypre_ParCSRMatrixDiag(B);
   hypre_CSRMatrix     *B_offd  = hypre_ParCSRMatrixOffd(B);
   hypre_CSRMatrix     *AT_diag;
   hypre_CSRMatrix     *AT_offd;

   HYPRE_Int            num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int            num_cols_diag_A  = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int            num_rows_diag_B  = hypre_CSRMatrixNumRows(B_diag);
   HYPRE_Int            num_cols_diag_B  = hypre_CSRMatrixNumCols(B_diag);
   HYPRE_Int            num_cols_offd_B  = hypre_CSRMatrixNumCols(B_offd);
   HYPRE_BigInt         first_col_diag_B = hypre_ParCSRMatrixFirstColDiag(B);

   HYPRE_BigInt        *col_map_offd_B = hypre_ParCSRMatrixColMapOffd(B);
   HYPRE_BigInt        *col_starts_A   = hypre_ParCSRMatrixColStarts(A);
   HYPRE_BigInt        *col_starts_B   = hypre_ParCSRMatrixColStarts(B);

   hypre_ParCSRMatrix  *C;
   hypre_CSRMatrix     *C_diag = NULL;
   hypre_CSRMatrix     *C_offd = NULL;

   HYPRE_BigInt        *col_map_offd_C = NULL;
   HYPRE_Int           *map_B_to_C;
   HYPRE_BigInt         first_col_diag_C;
   HYPRE_BigInt         last_col_diag_C;
   HYPRE_Int            num_cols_offd_C = 0;

   HYPRE_BigInt         n_rows_A, n_cols_A;
   HYPRE_BigInt         n_rows_B, n_cols_B;
   HYPRE_Int            j_indx, cnt;
   HYPRE_Int            num_procs, my_id;

   n_rows_A = hypre_ParCSRMatrixGlobalNumRows(A);
   n_cols_A = hypre_ParCSRMatrixGlobalNumCols(A);
   n_rows_B = hypre_ParCSRMatrixGlobalNumRows(B);
   n_cols_B = hypre_ParCSRMatrixGlobalNumCols(B);

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if (n_rows_A != n_rows_B || num_rows_diag_A != num_rows_diag_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /*if (num_cols_diag_A == num_cols_diag_B) allsquare = 1;*/

   /* Compute AT_diag if necessary */
   if (!hypre_ParCSRMatrixDiagT(A))
   {
      hypre_CSRMatrixTranspose(A_diag, &AT_diag, 1);
   }
   else
   {
      AT_diag = hypre_ParCSRMatrixDiagT(A);
   }

   if (num_procs == 1)
   {
      C_diag = hypre_CSRMatrixMultiplyHost(AT_diag, B_diag);
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_A, 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, hypre_CSRMatrixMemoryLocation(C_diag));
      hypre_CSRMatrixNumRownnz(C_offd) = 0;
   }
   else
   {
      hypre_CSRMatrix  *C_tmp_diag = NULL;
      hypre_CSRMatrix  *C_tmp_offd = NULL;
      hypre_CSRMatrix  *C_int      = NULL;
      hypre_CSRMatrix  *C_ext      = NULL;
      hypre_CSRMatrix  *C_ext_diag = NULL;
      hypre_CSRMatrix  *C_ext_offd = NULL;
      hypre_CSRMatrix  *C_int_diag = NULL;
      hypre_CSRMatrix  *C_int_offd = NULL;

      HYPRE_Int         i;
      HYPRE_Int        *C_tmp_offd_i;
      HYPRE_Int        *C_tmp_offd_j;
      HYPRE_Int        *send_map_elmts_A;
      void             *request;

      /* Compute AT_offd if necessary */
      if (!hypre_ParCSRMatrixOffdT(A))
      {
         hypre_CSRMatrixTranspose(A_offd, &AT_offd, 1);
      }
      else
      {
         AT_offd = hypre_ParCSRMatrixOffdT(A);
      }

      C_int_diag = hypre_CSRMatrixMultiplyHost(AT_offd, B_diag);
      C_int_offd = hypre_CSRMatrixMultiplyHost(AT_offd, B_offd);

      hypre_ParCSRMatrixDiag(B) = C_int_diag;
      hypre_ParCSRMatrixOffd(B) = C_int_offd;

      C_int = hypre_MergeDiagAndOffd(B);

      hypre_ParCSRMatrixDiag(B) = B_diag;
      hypre_ParCSRMatrixOffd(B) = B_offd;

      if (!hypre_ParCSRMatrixCommPkg(A))
      {
         hypre_MatvecCommPkgCreate(A);
      }
      comm_pkg_A = hypre_ParCSRMatrixCommPkg(A);

      /* contains communication; should be explicitly included to allow for overlap */
      hypre_ExchangeExternalRowsInit(C_int, comm_pkg_A, &request);
      C_ext = hypre_ExchangeExternalRowsWait(request);

      hypre_CSRMatrixDestroy(C_int);
      hypre_CSRMatrixDestroy(C_int_diag);
      hypre_CSRMatrixDestroy(C_int_offd);

      C_tmp_diag = hypre_CSRMatrixMultiplyHost(AT_diag, B_diag);
      C_tmp_offd = hypre_CSRMatrixMultiplyHost(AT_diag, B_offd);

      if (!hypre_ParCSRMatrixOffdT(A))
      {
         if (keep_transpose)
         {
            hypre_ParCSRMatrixOffdT(A) = AT_offd;
         }
         else
         {
            hypre_CSRMatrixDestroy(AT_offd);
         }
      }

      /*-----------------------------------------------------------------------
       *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
       *  to obtain C_diag and C_offd
       *-----------------------------------------------------------------------*/

      /* split C_ext in local C_ext_diag and nonlocal part C_ext_offd,
         also generate new col_map_offd and adjust column indices accordingly */
      first_col_diag_C = first_col_diag_B;
      last_col_diag_C = first_col_diag_B + num_cols_diag_B - 1;

      if (C_ext)
      {
         hypre_CSRMatrixSplit(C_ext, first_col_diag_C, last_col_diag_C,
                              num_cols_offd_B, col_map_offd_B, &num_cols_offd_C, &col_map_offd_C,
                              &C_ext_diag, &C_ext_offd);

         hypre_CSRMatrixDestroy(C_ext);
         C_ext = NULL;
      }

      C_tmp_offd_i = hypre_CSRMatrixI(C_tmp_offd);
      C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);

      if (num_cols_offd_B)
      {
         map_B_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_B[cnt])
            {
               map_B_to_C[cnt++] = i;
               if (cnt == num_cols_offd_B)
               {
                  break;
               }
            }
         }
         for (i = 0; i < C_tmp_offd_i[hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
         {
            j_indx = C_tmp_offd_j[i];
            C_tmp_offd_j[i] = map_B_to_C[j_indx];
         }
         hypre_TFree(map_B_to_C, HYPRE_MEMORY_HOST);
      }

      /*-----------------------------------------------------------------------
       *  Need to compute C_diag = C_tmp_diag + C_ext_diag
       *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
       *-----------------------------------------------------------------------*/
      send_map_elmts_A = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_A);
      C_diag = hypre_CSRMatrixAddPartial(C_tmp_diag, C_ext_diag, send_map_elmts_A);
      hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;
      C_offd = hypre_CSRMatrixAddPartial(C_tmp_offd, C_ext_offd, send_map_elmts_A);

      hypre_CSRMatrixDestroy(C_tmp_diag);
      hypre_CSRMatrixDestroy(C_tmp_offd);
      hypre_CSRMatrixDestroy(C_ext_diag);
      hypre_CSRMatrixDestroy(C_ext_offd);
   }

   if (!hypre_ParCSRMatrixDiagT(A))
   {
      if (keep_transpose)
      {
         hypre_ParCSRMatrixDiagT(A) = AT_diag;
      }
      else
      {
         hypre_CSRMatrixDestroy(AT_diag);
      }
   }

   C = hypre_ParCSRMatrixCreate(comm, n_cols_A, n_cols_B, col_starts_A, col_starts_B,
                                num_cols_offd_C, C_diag->num_nonzeros, C_offd->num_nonzeros);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixOffd(C) = C_offd;

   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRTMatMatKT
 *
 * Multiplies two ParCSRMatrices transpose(A) and B and returns
 * the product in ParCSRMatrix C.
 *
 * If either AT_diag or AT_offd don't exist and the flag keep_transpose is
 * true, these local matrices are saved in the ParCSRMatrix A
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRTMatMatKT( hypre_ParCSRMatrix  *A,
                       hypre_ParCSRMatrix  *B,
                       HYPRE_Int            keep_transpose)
{
   hypre_GpuProfilingPushRange("Mat-T-Mat");

   hypre_ParCSRMatrix *C = NULL;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(A),
                                                      hypre_ParCSRMatrixMemoryLocation(B) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_ParCSRTMatMatKTDevice(A, B, keep_transpose);
   }
   else
#endif
   {
      C = hypre_ParCSRTMatMatKTHost(A, B, keep_transpose);
   }

   hypre_GpuProfilingPopRange();

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRTMatMat
 *
 * Computes "C = A^T * B" and discards the temporary local matrices generated
 * in the algorithm (keep_transpose = 0).
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRTMatMat( hypre_ParCSRMatrix  *A,
                     hypre_ParCSRMatrix  *B)
{
   return hypre_ParCSRTMatMatKT(A, B, 0);
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRAPKTHost
 *
 * Host implementation of hypre_ParCSRMatrixRAPKT
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixRAPKTHost( hypre_ParCSRMatrix *R,
                             hypre_ParCSRMatrix *A,
                             hypre_ParCSRMatrix *P,
                             HYPRE_Int           keep_transpose )
{
   MPI_Comm              comm             = hypre_ParCSRMatrixComm(A);

   hypre_ParCSRCommPkg  *comm_pkg_R       = hypre_ParCSRMatrixCommPkg(R);
   HYPRE_BigInt          n_rows_R         = hypre_ParCSRMatrixGlobalNumRows(R);
   HYPRE_BigInt          n_cols_R         = hypre_ParCSRMatrixGlobalNumCols(R);
   hypre_CSRMatrix      *R_diag           = hypre_ParCSRMatrixDiag(R);
   hypre_CSRMatrix      *RT_diag          = hypre_ParCSRMatrixDiagT(R);
   hypre_CSRMatrix      *R_offd           = hypre_ParCSRMatrixOffd(R);
   hypre_CSRMatrix      *RT_offd          = hypre_ParCSRMatrixOffdT(R);

   HYPRE_Int             num_rows_diag_R  = hypre_CSRMatrixNumRows(R_diag);
   HYPRE_Int             num_cols_diag_R  = hypre_CSRMatrixNumCols(R_diag);
   HYPRE_Int             num_cols_offd_R  = hypre_CSRMatrixNumCols(R_offd);
   HYPRE_BigInt         *col_starts_R     = hypre_ParCSRMatrixColStarts(R);

   hypre_CSRMatrix      *A_diag           = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd           = hypre_ParCSRMatrixOffd(A);
   HYPRE_BigInt          n_rows_A         = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt          n_cols_A         = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_BigInt         *row_starts_A     = hypre_ParCSRMatrixRowStarts(A);

   HYPRE_Int             num_rows_diag_A  = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int             num_cols_diag_A  = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int             num_cols_offd_A  = hypre_CSRMatrixNumCols(A_offd);

   HYPRE_BigInt          n_rows_P         = hypre_ParCSRMatrixGlobalNumRows(P);
   HYPRE_BigInt          n_cols_P         = hypre_ParCSRMatrixGlobalNumCols(P);
   HYPRE_BigInt         *col_map_offd_P   = hypre_ParCSRMatrixColMapOffd(P);
   hypre_CSRMatrix      *P_diag           = hypre_ParCSRMatrixDiag(P);
   hypre_CSRMatrix      *P_offd           = hypre_ParCSRMatrixOffd(P);

   HYPRE_BigInt          first_col_diag_P = hypre_ParCSRMatrixFirstColDiag(P);
   HYPRE_BigInt         *col_starts_P     = hypre_ParCSRMatrixColStarts(P);
   HYPRE_Int             num_rows_diag_P  = hypre_CSRMatrixNumRows(P_diag);
   HYPRE_Int             num_cols_diag_P  = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int             num_cols_offd_P  = hypre_CSRMatrixNumCols(P_offd);

   hypre_ParCSRMatrix   *Q;
   HYPRE_BigInt         *col_map_offd_Q = NULL;
   HYPRE_Int            *map_P_to_Q = NULL;

   hypre_CSRMatrix      *Q_diag = NULL;
   hypre_CSRMatrix      *Q_offd = NULL;

   HYPRE_Int             num_cols_offd_Q = 0;

   hypre_CSRMatrix      *Ps_ext;
   hypre_CSRMatrix      *Pext_diag;
   hypre_CSRMatrix      *Pext_offd;

   hypre_CSRMatrix      *AP_diag;
   hypre_CSRMatrix      *AP_offd;
   HYPRE_Int             AP_offd_num_nonzeros;
   HYPRE_Int            *AP_offd_j;
   hypre_CSRMatrix      *APext_diag = NULL;
   hypre_CSRMatrix      *APext_offd = NULL;

   hypre_ParCSRMatrix   *C;
   HYPRE_BigInt         *col_map_offd_C = NULL;
   HYPRE_Int            *map_Q_to_C;
   hypre_CSRMatrix      *C_diag = NULL;
   hypre_CSRMatrix      *C_offd = NULL;
   HYPRE_BigInt          first_col_diag_C;
   HYPRE_BigInt          last_col_diag_C;

   HYPRE_Int             num_cols_offd_C = 0;
   HYPRE_Int             j_indx;
   HYPRE_Int             num_procs, my_id;
   HYPRE_Int             cnt, i;

   hypre_MPI_Comm_size(comm, &num_procs);
   hypre_MPI_Comm_rank(comm, &my_id);

   if ( n_rows_R != n_rows_A || num_rows_diag_R != num_rows_diag_A ||
        n_cols_A != n_rows_P || num_cols_diag_A != num_rows_diag_P )
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, " Error! Incompatible matrix dimensions!\n");
      return NULL;
   }

   /* Compute RT_diag if necessary */
   if (!hypre_ParCSRMatrixDiagT(R))
   {
      hypre_CSRMatrixTranspose(R_diag, &RT_diag, 1);
   }
   else
   {
      RT_diag = hypre_ParCSRMatrixDiagT(R);
   }

   if (num_procs > 1)
   {
      HYPRE_BigInt     last_col_diag_P;
      hypre_CSRMatrix *C_tmp_diag = NULL;
      hypre_CSRMatrix *C_tmp_offd = NULL;
      hypre_CSRMatrix *C_int = NULL;
      hypre_CSRMatrix *C_ext = NULL;
      hypre_CSRMatrix *C_ext_diag = NULL;
      hypre_CSRMatrix *C_ext_offd = NULL;
      hypre_CSRMatrix *C_int_diag = NULL;
      hypre_CSRMatrix *C_int_offd = NULL;

      HYPRE_Int       *C_tmp_offd_i;
      HYPRE_Int       *C_tmp_offd_j;

      HYPRE_Int       *send_map_elmts_R;
      void            *request;

      /*---------------------------------------------------------------------
       * If there exists no CommPkg for A, a CommPkg is generated using
       * equally load balanced partitionings within
       * hypre_ParCSRMatrixExtractBExt
       *--------------------------------------------------------------------*/
      Ps_ext = hypre_ParCSRMatrixExtractBExt(P, A, 1); /* contains communication
                                                          which should be explicitly included to allow for overlap */
      if (num_cols_offd_A)
      {
         last_col_diag_P = first_col_diag_P + num_cols_diag_P - 1;
         hypre_CSRMatrixSplit(Ps_ext, first_col_diag_P, last_col_diag_P, num_cols_offd_P, col_map_offd_P,
                              &num_cols_offd_Q, &col_map_offd_Q, &Pext_diag, &Pext_offd);
         /* These require data from other processes */
         APext_diag = hypre_CSRMatrixMultiplyHost(A_offd, Pext_diag);
         APext_offd = hypre_CSRMatrixMultiplyHost(A_offd, Pext_offd);

         hypre_CSRMatrixDestroy(Pext_diag);
         hypre_CSRMatrixDestroy(Pext_offd);
      }
      else
      {
         num_cols_offd_Q = num_cols_offd_P;
         col_map_offd_Q = hypre_CTAlloc(HYPRE_BigInt, num_cols_offd_Q, HYPRE_MEMORY_HOST);
         for (i = 0; i < num_cols_offd_P; i++)
         {
            col_map_offd_Q[i] = col_map_offd_P[i];
         }
      }
      hypre_CSRMatrixDestroy(Ps_ext);

      /* These are local and could be overlapped with communication */
      AP_diag = hypre_CSRMatrixMultiplyHost(A_diag, P_diag);

      if (num_cols_offd_P)
      {
         AP_offd = hypre_CSRMatrixMultiplyHost(A_diag, P_offd);
         if (num_cols_offd_Q > num_cols_offd_P)
         {
            map_P_to_Q = hypre_CTAlloc(HYPRE_Int, num_cols_offd_P, HYPRE_MEMORY_HOST);

            cnt = 0;
            for (i = 0; i < num_cols_offd_Q; i++)
            {
               if (col_map_offd_Q[i] == col_map_offd_P[cnt])
               {
                  map_P_to_Q[cnt++] = i;
                  if (cnt == num_cols_offd_P)
                  {
                     break;
                  }
               }
            }
            AP_offd_num_nonzeros = hypre_CSRMatrixNumNonzeros(AP_offd);
            AP_offd_j = hypre_CSRMatrixJ(AP_offd);
            for (i = 0; i < AP_offd_num_nonzeros; i++)
            {
               AP_offd_j[i] = map_P_to_Q[AP_offd_j[i]];
            }

            hypre_TFree(map_P_to_Q, HYPRE_MEMORY_HOST);
            hypre_CSRMatrixNumCols(AP_offd) = num_cols_offd_Q;
         }
      }

      if (num_cols_offd_A) /* number of rows for Pext_diag */
      {
         Q_diag = hypre_CSRMatrixAdd(1.0, AP_diag, 1.0, APext_diag);
         hypre_CSRMatrixDestroy(AP_diag);
         hypre_CSRMatrixDestroy(APext_diag);
      }
      else
      {
         Q_diag = AP_diag;
      }

      if (num_cols_offd_P && num_cols_offd_A)
      {
         Q_offd = hypre_CSRMatrixAdd(1.0, AP_offd, 1.0, APext_offd);
         hypre_CSRMatrixDestroy(APext_offd);
         hypre_CSRMatrixDestroy(AP_offd);
      }
      else if (num_cols_offd_A)
      {
         Q_offd = APext_offd;
      }
      else if (num_cols_offd_P)
      {
         Q_offd = AP_offd;
      }
      else
      {
         Q_offd = hypre_CSRMatrixClone(A_offd, 1);
      }

      Q = hypre_ParCSRMatrixCreate(comm, n_rows_A, n_cols_P, row_starts_A,
                                   col_starts_P, num_cols_offd_Q,
                                   Q_diag->num_nonzeros, Q_offd->num_nonzeros);

      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(Q));
      hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(Q));
      hypre_ParCSRMatrixDiag(Q) = Q_diag;
      hypre_ParCSRMatrixOffd(Q) = Q_offd;
      hypre_ParCSRMatrixColMapOffd(Q) = col_map_offd_Q;

      C_tmp_diag = hypre_CSRMatrixMultiplyHost(RT_diag, Q_diag);
      if (num_cols_offd_Q)
      {
         C_tmp_offd = hypre_CSRMatrixMultiplyHost(RT_diag, Q_offd);
      }
      else
      {
         C_tmp_offd = hypre_CSRMatrixClone(Q_offd, 1);
         hypre_CSRMatrixNumRows(C_tmp_offd) = num_cols_diag_R;
      }

      if (num_cols_offd_R)
      {
         /* Compute RT_offd if necessary */
         if (!hypre_ParCSRMatrixOffdT(R))
         {
            hypre_CSRMatrixTranspose(R_offd, &RT_offd, 1);
         }
         else
         {
            RT_offd = hypre_ParCSRMatrixOffdT(R);
         }

         C_int_diag = hypre_CSRMatrixMultiplyHost(RT_offd, Q_diag);
         C_int_offd = hypre_CSRMatrixMultiplyHost(RT_offd, Q_offd);

         hypre_ParCSRMatrixDiag(Q) = C_int_diag;
         hypre_ParCSRMatrixOffd(Q) = C_int_offd;
         C_int = hypre_MergeDiagAndOffd(Q);
         hypre_ParCSRMatrixDiag(Q) = Q_diag;
         hypre_ParCSRMatrixOffd(Q) = Q_offd;
      }
      else
      {
         C_int = hypre_CSRMatrixCreate(0, 0, 0);
         hypre_CSRMatrixInitialize(C_int);
      }

      /* contains communication; should be explicitly included to allow for overlap */
      hypre_ExchangeExternalRowsInit(C_int, comm_pkg_R, &request);
      C_ext = hypre_ExchangeExternalRowsWait(request);

      hypre_CSRMatrixDestroy(C_int);
      if (num_cols_offd_R)
      {
         hypre_CSRMatrixDestroy(C_int_diag);
         hypre_CSRMatrixDestroy(C_int_offd);

         if (!hypre_ParCSRMatrixOffdT(R))
         {
            if (keep_transpose)
            {
               hypre_ParCSRMatrixOffdT(R) = RT_offd;
            }
            else
            {
               hypre_CSRMatrixDestroy(RT_offd);
            }
         }
      }

      /*-----------------------------------------------------------------------
       *  Add contents of C_ext to C_tmp_diag and C_tmp_offd
       *  to obtain C_diag and C_offd
       *-----------------------------------------------------------------------*/

      /* split C_ext in local C_ext_diag and nonlocal part C_ext_offd,
         also generate new col_map_offd and adjust column indices accordingly */

      if (C_ext)
      {
         first_col_diag_C = first_col_diag_P;
         last_col_diag_C = first_col_diag_P + num_cols_diag_P - 1;

         hypre_CSRMatrixSplit(C_ext, first_col_diag_C, last_col_diag_C,
                              num_cols_offd_Q, col_map_offd_Q, &num_cols_offd_C, &col_map_offd_C,
                              &C_ext_diag, &C_ext_offd);

         hypre_CSRMatrixDestroy(C_ext);
         C_ext = NULL;
         /*if (C_ext_offd->num_nonzeros == 0) C_ext_offd->num_cols = 0;*/
      }

      if (num_cols_offd_Q && C_tmp_offd->num_cols)
      {
         C_tmp_offd_i = hypre_CSRMatrixI(C_tmp_offd);
         C_tmp_offd_j = hypre_CSRMatrixJ(C_tmp_offd);

         map_Q_to_C = hypre_CTAlloc(HYPRE_Int, num_cols_offd_Q, HYPRE_MEMORY_HOST);

         cnt = 0;
         for (i = 0; i < num_cols_offd_C; i++)
         {
            if (col_map_offd_C[i] == col_map_offd_Q[cnt])
            {
               map_Q_to_C[cnt++] = i;
               if (cnt == num_cols_offd_Q)
               {
                  break;
               }
            }
         }
         for (i = 0; i < C_tmp_offd_i[hypre_CSRMatrixNumRows(C_tmp_offd)]; i++)
         {
            j_indx = C_tmp_offd_j[i];
            C_tmp_offd_j[i] = map_Q_to_C[j_indx];
         }
         hypre_TFree(map_Q_to_C, HYPRE_MEMORY_HOST);
      }
      hypre_CSRMatrixNumCols(C_tmp_offd) = num_cols_offd_C;
      hypre_ParCSRMatrixDestroy(Q);

      /*-----------------------------------------------------------------------
       *  Need to compute C_diag = C_tmp_diag + C_ext_diag
       *  and  C_offd = C_tmp_offd + C_ext_offd   !!!!
       *-----------------------------------------------------------------------*/

      send_map_elmts_R = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_R);
      if (C_ext_diag)
      {
         C_diag = hypre_CSRMatrixAddPartial(C_tmp_diag, C_ext_diag, send_map_elmts_R);
         hypre_CSRMatrixDestroy(C_tmp_diag);
         hypre_CSRMatrixDestroy(C_ext_diag);
      }
      else
      {
         C_diag = C_tmp_diag;
      }

      if (C_ext_offd)
      {
         C_offd = hypre_CSRMatrixAddPartial(C_tmp_offd, C_ext_offd, send_map_elmts_R);
         hypre_CSRMatrixDestroy(C_tmp_offd);
         hypre_CSRMatrixDestroy(C_ext_offd);
      }
      else
      {
         C_offd = C_tmp_offd;
      }
   }
   else
   {
      Q_diag = hypre_CSRMatrixMultiplyHost(A_diag, P_diag);
      C_diag = hypre_CSRMatrixMultiplyHost(RT_diag, Q_diag);
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_R, 0, 0);
      hypre_CSRMatrixInitialize_v2(C_offd, 0, hypre_CSRMatrixMemoryLocation(C_diag));
      hypre_CSRMatrixDestroy(Q_diag);
   }

   if (!hypre_ParCSRMatrixDiagT(R))
   {
      if (keep_transpose)
      {
         hypre_ParCSRMatrixDiagT(R) = RT_diag;
      }
      else
      {
         hypre_CSRMatrixDestroy(RT_diag);
      }
   }

   C = hypre_ParCSRMatrixCreate(comm, n_cols_R, n_cols_P, col_starts_R,
                                col_starts_P, num_cols_offd_C, 0, 0);

   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixDiag(C));
   hypre_CSRMatrixDestroy(hypre_ParCSRMatrixOffd(C));
   hypre_ParCSRMatrixDiag(C) = C_diag;

   if (C_offd)
   {
      hypre_ParCSRMatrixOffd(C) = C_offd;
   }
   else
   {
      C_offd = hypre_CSRMatrixCreate(num_cols_diag_R, 0, 0);
      hypre_CSRMatrixInitialize(C_offd);
      hypre_ParCSRMatrixOffd(C) = C_offd;
   }

   hypre_ParCSRMatrixColMapOffd(C) = col_map_offd_C;

   if (num_procs > 1)
   {
      /* hypre_GenerateRAPCommPkg(RAP, A); */
      hypre_MatvecCommPkgCreate(C);
   }

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRAPKT
 *
 * Computes "C = R * A * P".
 *
 * If either RT_diag or RT_offd don't exist and the flag keep_transpose is
 * true, these local matrices are saved in the ParCSRMatrix R
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixRAPKT( hypre_ParCSRMatrix  *R,
                         hypre_ParCSRMatrix  *A,
                         hypre_ParCSRMatrix  *P,
                         HYPRE_Int            keep_transpose)
{
   hypre_ParCSRMatrix *C = NULL;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("TripleMat-RAP");

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(R),
                                                      hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      C = hypre_ParCSRMatrixRAPKTDevice(R, A, P, keep_transpose);
   }
   else
#endif
   {
      C = hypre_ParCSRMatrixRAPKTHost(R, A, P, keep_transpose);
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixRAP
 *
 * Computes "C = R * A * P" and discards the temporary local matrices generated
 * in the algorithm (keep_transpose = 0).
 *--------------------------------------------------------------------------*/

hypre_ParCSRMatrix*
hypre_ParCSRMatrixRAP( hypre_ParCSRMatrix *R,
                       hypre_ParCSRMatrix *A,
                       hypre_ParCSRMatrix *P )
{
   return hypre_ParCSRMatrixRAPKT(R, A, P, 0);
}

/*--------------------------------------------------------------------------
 * OLD NOTES:
 * Sketch of John's code to build RAP
 *
 * Uses two integer arrays icg and ifg as marker arrays
 *
 *  icg needs to be of size n_fine; size of ia.
 *     A negative value of icg(i) indicates i is a f-point, otherwise
 *     icg(i) is the converts from fine to coarse grid orderings.
 *     Note that I belive the code assumes that if i<j and both are
 *     c-points, then icg(i) < icg(j).
 *  ifg needs to be of size n_coarse; size of irap
 *     I don't think it has meaning as either input or output.
 *
 * In the code, both the interpolation and restriction operator
 * are stored row-wise in the array b. If i is a f-point,
 * ib(i) points the row of the interpolation operator for point
 * i. If i is a c-point, ib(i) points the row of the restriction
 * operator for point i.
 *
 * In the CSR storage for rap, its guaranteed that the rows will
 * be ordered ( i.e. ic<jc -> irap(ic) < irap(jc)) but I don't
 * think there is a guarantee that the entries within a row will
 * be ordered in any way except that the diagonal entry comes first.
 *
 * As structured now, the code requires that the size of rap be
 * predicted up front. To avoid this, one could execute the code
 * twice, the first time would only keep track of icg ,ifg and ka.
 * Then you would know how much memory to allocate for rap and jrap.
 * The second time would fill in these arrays. Actually you might
 * be able to include the filling in of jrap into the first pass;
 * just overestimate its size (its an integer array) and cut it
 * back before the second time through. This would avoid some if tests
 * in the second pass.
 *
 * Questions
 *            1) parallel (PetSc) version?
 *            2) what if we don't store R row-wise and don't
 *               even want to store a copy of it in this form
 *               temporarily?
 *--------------------------------------------------------------------------*/
