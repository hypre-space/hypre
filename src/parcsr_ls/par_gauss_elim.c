/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "Common.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

/*-------------------------------------------------------------------------
 *
 *                      Gaussian Elimination
 *
 *------------------------------------------------------------------------ */

HYPRE_Int hypre_GaussElimSetup (hypre_ParAMGData *amg_data, HYPRE_Int level, HYPRE_Int relax_type)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SETUP] -= hypre_MPI_Wtime();
#endif

   /* Par Data Structure variables */
   hypre_ParCSRMatrix *A      = hypre_ParAMGDataAArray(amg_data)[level];
   hypre_CSRMatrix    *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix    *A_offd = hypre_ParCSRMatrixOffd(A);

   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int global_num_rows = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(A);
   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   MPI_Comm new_comm;

   /* Generate sub communicator: processes that have nonzero num_rows */
   hypre_GenerateSubComm(comm, num_rows, &new_comm);

   if (num_rows)
   {
      hypre_CSRMatrix *A_diag_host, *A_offd_host;
      if (hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(A_diag)) != hypre_MEMORY_HOST)
      {
         A_diag_host = hypre_CSRMatrixClone_v2(A_diag, 1, HYPRE_MEMORY_HOST);
      }
      else
      {
         A_diag_host = A_diag;
      }
      if (hypre_GetActualMemLocation(hypre_CSRMatrixMemoryLocation(A_offd)) != hypre_MEMORY_HOST)
      {
         A_offd_host = hypre_CSRMatrixClone_v2(A_offd, 1, HYPRE_MEMORY_HOST);
      }
      else
      {
         A_offd_host = A_offd;
      }

      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(A);
      HYPRE_Int  *A_diag_i    = hypre_CSRMatrixI(A_diag_host);
      HYPRE_Int  *A_offd_i    = hypre_CSRMatrixI(A_offd_host);
      HYPRE_Int  *A_diag_j    = hypre_CSRMatrixJ(A_diag_host);
      HYPRE_Int  *A_offd_j    = hypre_CSRMatrixJ(A_offd_host);
      HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag_host);
      HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd_host);

      HYPRE_Real *A_mat, *A_mat_local;
      HYPRE_Int *comm_info, *info, *displs;
      HYPRE_Int *mat_info, *mat_displs;
      HYPRE_Int new_num_procs, A_mat_local_size, i, jj, column;
      HYPRE_BigInt first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);

      hypre_MPI_Comm_size(new_comm, &new_num_procs);

      comm_info  = hypre_CTAlloc(HYPRE_Int, 2*new_num_procs+1, HYPRE_MEMORY_HOST);
      mat_info   = hypre_CTAlloc(HYPRE_Int, new_num_procs,     HYPRE_MEMORY_HOST);
      mat_displs = hypre_CTAlloc(HYPRE_Int, new_num_procs+1,   HYPRE_MEMORY_HOST);
      info = &comm_info[0];
      displs = &comm_info[new_num_procs];

      hypre_MPI_Allgather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, new_comm);

      displs[0] = 0;
      mat_displs[0] = 0;
      for (i = 0; i < new_num_procs; i++)
      {
         displs[i+1] = displs[i] + info[i];
         mat_displs[i+1] = global_num_rows * displs[i+1];
         mat_info[i] = global_num_rows * info[i];
      }

      hypre_ParAMGDataBVec(amg_data) = hypre_CTAlloc(HYPRE_Real, global_num_rows, HYPRE_MEMORY_HOST);

      A_mat_local_size = global_num_rows * num_rows;
      A_mat_local = hypre_CTAlloc(HYPRE_Real, A_mat_local_size,                HYPRE_MEMORY_HOST);
      A_mat       = hypre_CTAlloc(HYPRE_Real, global_num_rows*global_num_rows, HYPRE_MEMORY_HOST);

      /* load local matrix into A_mat_local */
      for (i = 0; i < num_rows; i++)
      {
         for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
         {
             /* need col major */
             column = A_diag_j[jj]+first_row_index;
             A_mat_local[i*global_num_rows + column] = A_diag_data[jj];
         }
         for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
         {
             /* need col major */
             column = col_map_offd[A_offd_j[jj]];
             A_mat_local[i*global_num_rows + column] = A_offd_data[jj];
         }
      }

      hypre_MPI_Allgatherv(A_mat_local, A_mat_local_size, HYPRE_MPI_REAL, A_mat, mat_info,
                           mat_displs, HYPRE_MPI_REAL, new_comm);

      if (relax_type == 99)
      {
         HYPRE_Real *AT_mat = hypre_CTAlloc(HYPRE_Real, global_num_rows*global_num_rows, HYPRE_MEMORY_HOST);
         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = 0; jj < global_num_rows; jj++)
            {
               AT_mat[i*global_num_rows + jj] = A_mat[i + jj*global_num_rows];
            }
         }
         hypre_ParAMGDataAMat(amg_data) = AT_mat;
         hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
      }
      else if (relax_type == 9)
      {
         hypre_ParAMGDataAMat(amg_data) = A_mat;
      }
      else if (relax_type == 199)
      {
         HYPRE_Real *AT_mat = hypre_TAlloc(HYPRE_Real, global_num_rows*global_num_rows, HYPRE_MEMORY_HOST);
         HYPRE_Real *Ainv   = hypre_TAlloc(HYPRE_Real, num_rows*global_num_rows,        HYPRE_MEMORY_HOST);
         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = 0; jj < global_num_rows; jj++)
            {
               AT_mat[i*global_num_rows + jj] = A_mat[i + jj*global_num_rows];
            }
         }
         HYPRE_Int *ipiv, info, query = -1, lwork;
         HYPRE_Real lwork_opt, *work;
         ipiv = hypre_TAlloc(HYPRE_Int, global_num_rows, HYPRE_MEMORY_HOST);
         hypre_dgetrf(&global_num_rows, &global_num_rows, AT_mat, &global_num_rows, ipiv, &info);
         hypre_assert(info == 0);
         hypre_dgetri(&global_num_rows, AT_mat, &global_num_rows, ipiv, &lwork_opt, &query, &info);
         hypre_assert(info == 0);
         lwork = lwork_opt;
         work = hypre_TAlloc(HYPRE_Real, lwork, HYPRE_MEMORY_HOST);
         hypre_dgetri(&global_num_rows, AT_mat, &global_num_rows, ipiv, work, &lwork, &info);
         hypre_assert(info == 0);

         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = 0; jj < num_rows; jj++)
            {
               Ainv[i*num_rows+jj] = AT_mat[i*global_num_rows+jj+first_row_index];
            }
         }

         hypre_TFree(ipiv,   HYPRE_MEMORY_HOST);
         hypre_TFree(A_mat,  HYPRE_MEMORY_HOST);
         hypre_TFree(AT_mat, HYPRE_MEMORY_HOST);
         hypre_TFree(work,   HYPRE_MEMORY_HOST);

         hypre_ParAMGDataAInv(amg_data) = Ainv;
      }

      hypre_ParAMGDataCommInfo(amg_data) = comm_info;
      hypre_ParAMGDataNewComm(amg_data)  = new_comm;

      hypre_TFree(mat_info,    HYPRE_MEMORY_HOST);
      hypre_TFree(mat_displs,  HYPRE_MEMORY_HOST);
      hypre_TFree(A_mat_local, HYPRE_MEMORY_HOST);

      if (A_diag_host != A_diag)
      {
         hypre_CSRMatrixDestroy(A_diag_host);
      }

      if (A_offd_host != A_offd)
      {
         hypre_CSRMatrixDestroy(A_offd_host);
      }
   }

   hypre_ParAMGDataGSSetup(amg_data) = 1;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SETUP] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}

/* relax_type = 9, 99, 199, see par_relax.c for 19 and 98 */
HYPRE_Int hypre_GaussElimSolve (hypre_ParAMGData *amg_data, HYPRE_Int level, HYPRE_Int relax_type)
{
#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SOLVE] -= hypre_MPI_Wtime();
#endif

   hypre_ParCSRMatrix *A = hypre_ParAMGDataAArray(amg_data)[level];
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int n = hypre_CSRMatrixNumRows(A_diag);
   HYPRE_Int error_flag = 0;

   if (hypre_ParAMGDataGSSetup(amg_data) == 0)
   {
      hypre_GaussElimSetup(amg_data, level, relax_type);
   }

   if (n)
   {
      MPI_Comm new_comm = hypre_ParAMGDataNewComm(amg_data);
      hypre_ParVector *f = hypre_ParAMGDataFArray(amg_data)[level];
      hypre_ParVector *u = hypre_ParAMGDataUArray(amg_data)[level];
      HYPRE_Real *b_vec  = hypre_ParAMGDataBVec(amg_data);
      HYPRE_Real *f_data = hypre_VectorData(hypre_ParVectorLocalVector(f));
      HYPRE_Real *u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
      HYPRE_Int *comm_info = hypre_ParAMGDataCommInfo(amg_data);
      HYPRE_Int *displs, *info;
      HYPRE_Int n_global = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(A);
      HYPRE_Int new_num_procs;
      HYPRE_Int first_row_index = (HYPRE_Int) hypre_ParCSRMatrixFirstRowIndex(A);
      HYPRE_Int one_i = 1;

      hypre_MPI_Comm_size(new_comm, &new_num_procs);
      info = &comm_info[0];
      displs = &comm_info[new_num_procs];

      HYPRE_Real *f_data_host, *u_data_host;

      if (hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(f)) != hypre_MEMORY_HOST)
      {
         f_data_host = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);

         hypre_TMemcpy(f_data_host, f_data, HYPRE_Real, n, HYPRE_MEMORY_HOST, hypre_ParVectorMemoryLocation(f));
      }
      else
      {
         f_data_host = f_data;
      }

      if (hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(u)) != hypre_MEMORY_HOST)
      {
         u_data_host = hypre_TAlloc(HYPRE_Real, n, HYPRE_MEMORY_HOST);
      }
      else
      {
         u_data_host = u_data;
      }

      hypre_MPI_Allgatherv (f_data_host, n, HYPRE_MPI_REAL, b_vec, info,
                            displs, HYPRE_MPI_REAL, new_comm);

      if (f_data_host != f_data)
      {
         hypre_TFree(f_data_host, HYPRE_MEMORY_HOST);
      }

      if (relax_type == 9 || relax_type == 99)
      {
         HYPRE_Real *A_mat = hypre_ParAMGDataAMat(amg_data);
         HYPRE_Real *A_tmp;
         HYPRE_Int   i, my_info;

         A_tmp = hypre_CTAlloc(HYPRE_Real, n_global*n_global, HYPRE_MEMORY_HOST);
         for (i = 0; i < n_global*n_global; i++)
         {
            A_tmp[i] = A_mat[i];
         }

         if (relax_type == 9)
         {
            hypre_gselim(A_tmp, b_vec, n_global, error_flag);
         }
         else if (relax_type == 99) /* use pivoting */
         {
            HYPRE_Int *piv = hypre_CTAlloc(HYPRE_Int, n_global, HYPRE_MEMORY_HOST);

            /* write over A with LU */
            hypre_dgetrf(&n_global, &n_global, A_tmp, &n_global, piv, &my_info);

            /* now b_vec = inv(A)*b_vec */
            hypre_dgetrs("N", &n_global, &one_i, A_tmp, &n_global, piv, b_vec, &n_global, &my_info);

            hypre_TFree(piv, HYPRE_MEMORY_HOST);
         }

         for (i = 0; i < n; i++)
         {
            u_data_host[i] = b_vec[first_row_index+i];
         }

         hypre_TFree(A_tmp, HYPRE_MEMORY_HOST);
      }
      else if (relax_type == 199)
      {
         HYPRE_Real *Ainv = hypre_ParAMGDataAInv(amg_data);

         char cN = 'N';
         HYPRE_Real one = 1.0, zero = 0.0;
         hypre_dgemv(&cN, &n, &n_global, &one, Ainv, &n, b_vec, &one_i, &zero, u_data_host, &one_i);
      }

      if (u_data_host != u_data)
      {
         hypre_TMemcpy(u_data, u_data_host, HYPRE_Real, n, hypre_ParVectorMemoryLocation(u), HYPRE_MEMORY_HOST);
         hypre_TFree(u_data_host, HYPRE_MEMORY_HOST);
      }
   }

   if (error_flag)
   {
      hypre_error(HYPRE_ERROR_GENERIC);
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SOLVE] += hypre_MPI_Wtime();
#endif

   return hypre_error_flag;
}































#if 0
#include "HYPRE_config.h"
#ifndef HYPRE_SEQUENTIAL
#define HYPRE_SEQUENTIAL
#endif
#include "_hypre_utilities.h"
#include "_hypre_blas.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define BLOCK_SIZE 512

__global__ void
hypreCUDAKernel_dgemv(HYPRE_Int   m,
                      HYPRE_Int   n,
                      HYPRE_Int   lda,
                      HYPRE_Real *a,
                      HYPRE_Real *x,
                      HYPRE_Real *y)
{
   __shared__ HYPRE_Real sh_x[BLOCK_SIZE];

   HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>();
   HYPRE_Int tid = hypre_cuda_get_thread_id<1>();

   HYPRE_Real y_row = 0.0;

   for (HYPRE_Int k = 0; k < n; k += BLOCK_SIZE)
   {
      if (k + tid < n)
      {
         sh_x[tid] = read_only_load(&x[k+tid]);
      }

      __syncthreads();

      if (row < m)
      {
#pragma unroll
         for (HYPRE_Int j = 0; j < BLOCK_SIZE; j++)
         {
            const HYPRE_Int col = k + j;
            if (col < n)
            {
               y_row += a[row + col*lda] * sh_x[j];
            }
         }
      }

      __syncthreads();
   }

   if (row < m)
   {
      y[row] = y_row;
   }
}

HYPRE_Int hypre_dgemv_device(HYPRE_Int m, HYPRE_Int n, HYPRE_Int lda, HYPRE_Real *a, HYPRE_Real *x, HYPRE_Real *y)
{
   dim3 bDim(BLOCK_SIZE, 1, 1);
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(m, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_dgemv, gDim, bDim, m, n, lda, a, x, y );

   return hypre_error_flag;
}

#endif // defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#endif
