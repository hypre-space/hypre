/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_blas.h"
#include "_hypre_lapack.h"

/*--------------------------------------------------------------------------
 * hypre_GaussElimSetup
 *
 * Gaussian elimination setup routine.
 *
 * Local matrices/vectors formed via MPI collectives defined on new
 * sub-communicators:
 *
 *   - 9: hypre's internal Gaussian elimination on the host.
 *   - 99: LU factorization with pivoting via LAPACK.
 *   - 199: explicit inverse A_inv = U^{-1}*L^{-1}.
 *
 * Local matrices/vectors formed via hypre_DataExchange:
 *
 *   - 19: hypre's internal Gaussian elimination on the host.
 *   - 98: LU factorization with pivoting.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GaussElimSetup(hypre_ParAMGData *amg_data,
                     HYPRE_Int         level,
                     HYPRE_Int         solver_type)
{
   /* Par Data Structure variables */
   hypre_ParCSRMatrix   *A               = hypre_ParAMGDataAArray(amg_data)[level];
   MPI_Comm              comm            = hypre_ParCSRMatrixComm(A);
   HYPRE_Int             num_rows        = hypre_ParCSRMatrixNumRows(A);
   HYPRE_Int             global_num_rows = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_BigInt          first_row_index = hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_BigInt         *col_map_offd    = hypre_ParCSRMatrixColMapOffd(A);
   hypre_CSRMatrix      *A_diag          = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *A_offd          = hypre_ParCSRMatrixOffd(A);
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   /* Local variables */
   HYPRE_Int             global_size     = global_num_rows * global_num_rows;
   HYPRE_Complex        *A_mat           = NULL;
   HYPRE_Complex        *AT_mat;
   HYPRE_Complex        *Ainv;
   HYPRE_Int            *Apiv;
   HYPRE_Int             i, jj, col;
   HYPRE_Int             ierr = 0;

   /*-----------------------------------------------------------------
    *  Sanity checks
    *-----------------------------------------------------------------*/

   /* Check for relaxation type */
   if (solver_type != 9  && solver_type != 99 && solver_type != 199 &&
       solver_type != 19 && solver_type != 98)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Wrong relaxation type!");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    *  Gaussian elimination setup
    *-----------------------------------------------------------------*/

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SETUP] -= hypre_MPI_Wtime();
#endif

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("GaussElimSetup");

   if (solver_type == 9 || solver_type == 99 || solver_type == 199)
   {
      hypre_CSRMatrix      *A_diag_host;
      hypre_CSRMatrix      *A_offd_host;
      MPI_Comm              new_comm;

      /* Generate sub communicator - processes that have nonzero num_rows */
      hypre_GenerateSubComm(comm, num_rows, &new_comm);
      hypre_ParAMGDataNewComm(amg_data) = new_comm;

      if (num_rows)
      {
         A_diag_host = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
                        hypre_CSRMatrixClone_v2(A_diag, 1, HYPRE_MEMORY_HOST) : A_diag;
         A_offd_host = (hypre_GetActualMemLocation(memory_location) == hypre_MEMORY_DEVICE) ?
                        hypre_CSRMatrixClone_v2(A_offd, 1, HYPRE_MEMORY_HOST) : A_offd;

         HYPRE_Int      *A_diag_i    = hypre_CSRMatrixI(A_diag_host);
         HYPRE_Int      *A_offd_i    = hypre_CSRMatrixI(A_offd_host);
         HYPRE_Int      *A_diag_j    = hypre_CSRMatrixJ(A_diag_host);
         HYPRE_Int      *A_offd_j    = hypre_CSRMatrixJ(A_offd_host);
         HYPRE_Complex  *A_diag_data = hypre_CSRMatrixData(A_diag_host);
         HYPRE_Complex  *A_offd_data = hypre_CSRMatrixData(A_offd_host);

         HYPRE_Complex  *A_mat_local;
         HYPRE_Int      *comm_info, *info, *displs;
         HYPRE_Int      *mat_info, *mat_displs;
         HYPRE_Int       new_num_procs, A_mat_local_size;

         hypre_MPI_Comm_size(new_comm, &new_num_procs);

         comm_info  = hypre_CTAlloc(HYPRE_Int, 2 * new_num_procs + 1, HYPRE_MEMORY_HOST);
         mat_info   = hypre_CTAlloc(HYPRE_Int, new_num_procs, HYPRE_MEMORY_HOST);
         mat_displs = hypre_CTAlloc(HYPRE_Int, new_num_procs + 1, HYPRE_MEMORY_HOST);
         info       = &comm_info[0];
         displs     = &comm_info[new_num_procs];

         hypre_ParAMGDataCommInfo(amg_data) = comm_info;
         hypre_ParAMGDataBVec(amg_data)     = hypre_CTAlloc(HYPRE_Real, global_num_rows,
                                                            HYPRE_MEMORY_HOST);

         hypre_MPI_Allgather(&num_rows, 1, HYPRE_MPI_INT, info, 1, HYPRE_MPI_INT, new_comm);

         displs[0] = 0;
         mat_displs[0] = 0;
         for (i = 0; i < new_num_procs; i++)
         {
            displs[i + 1] = displs[i] + info[i];
            mat_displs[i + 1] = global_num_rows * displs[i + 1];
            mat_info[i] = global_num_rows * info[i];
         }

         A_mat_local_size = global_num_rows * num_rows;
         A_mat_local = hypre_CTAlloc(HYPRE_Real, A_mat_local_size, HYPRE_MEMORY_HOST);
         A_mat       = hypre_CTAlloc(HYPRE_Real, global_size, HYPRE_MEMORY_HOST);

         /*---------------------------------------------------------------
          *  Load local matrix into A_mat_local.
          *---------------------------------------------------------------*/

         for (i = 0; i < num_rows; i++)
         {
            for (jj = A_diag_i[i]; jj < A_diag_i[i + 1]; jj++)
            {
               /* need col major */
               col = A_diag_j[jj] + first_row_index;
               A_mat_local[i * global_num_rows + col] = A_diag_data[jj];
            }

            for (jj = A_offd_i[i]; jj < A_offd_i[i + 1]; jj++)
            {
               /* need col major */
               col = col_map_offd[A_offd_j[jj]];
               A_mat_local[i * global_num_rows + col] = A_offd_data[jj];
            }
         }

         hypre_MPI_Allgatherv(A_mat_local, A_mat_local_size, HYPRE_MPI_REAL, A_mat, mat_info,
                              mat_displs, HYPRE_MPI_REAL, new_comm);

         /* Set dense matrix */
         if (solver_type == 9)
         {
            hypre_ParAMGDataAMat(amg_data) = A_mat;
         }
         else
         {
            AT_mat = hypre_CTAlloc(HYPRE_Real, global_size, HYPRE_MEMORY_HOST);
            for (i = 0; i < global_num_rows; i++)
            {
               for (jj = 0; jj < global_num_rows; jj++)
               {
                  AT_mat[i * global_num_rows + jj] = A_mat[i + jj * global_num_rows];
               }
            }
            hypre_ParAMGDataAMat(amg_data) = AT_mat;
            hypre_TFree(A_mat, HYPRE_MEMORY_HOST);
         }

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
      else
      {
         /* Skip setup if this rank has no rows. */
         hypre_ParAMGDataGSSetup(amg_data) = 1;
         return hypre_error_flag;
      }
   }
   else /* if (solver_type == 19 || solver_type = 98) */
   {
      /* Generate CSR matrix from ParCSRMatrix A */
      hypre_CSRMatrix   *A_CSR = hypre_ParCSRMatrixToCSRMatrixAll(A, HYPRE_MEMORY_HOST);

      if (num_rows)
      {
         HYPRE_Int      *A_CSR_i    = hypre_CSRMatrixI(A_CSR);
         HYPRE_Int      *A_CSR_j    = hypre_CSRMatrixJ(A_CSR);
         HYPRE_Complex  *A_CSR_data = hypre_CSRMatrixData(A_CSR);

         /*---------------------------------------------------------------
          *  Load CSR matrix into A_mat.
          *---------------------------------------------------------------*/

         /* Allocate memory */
         A_mat = hypre_CTAlloc(HYPRE_Real, global_size, HYPRE_MEMORY_HOST);
         hypre_ParAMGDataAMat(amg_data) = A_mat;

         /* TODO (VPM): Add OpenMP support */
         for (i = 0; i < global_num_rows; i++)
         {
            for (jj = A_CSR_i[i]; jj < A_CSR_i[i + 1]; jj++)
            {
               /* need col major */
               col = A_CSR_j[jj];
               A_mat[i + global_num_rows * col] = (HYPRE_Real) A_CSR_data[jj];
            }
         }
      }
      else
      {
         /* Free memory */
         hypre_CSRMatrixDestroy(A_CSR);

         /* Skip setup if this rank has no rows. */
         hypre_ParAMGDataGSSetup(amg_data) = 1;
         return hypre_error_flag;
      }

      /* Free memory */
      hypre_CSRMatrixDestroy(A_CSR);
   }

   /*-----------------------------------------------------------------
    *  Factorization phase
    *-----------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU)
   HYPRE_MemoryLocation   memory_location = hypre_ParCSRMatrixMemoryLocation(par_A);
   HYPRE_ExecutionPolicy  exec            = hypre_GetExecPolicy1(memory_location);

   if (exec == HYPRE_EXEC_DEVICE)
   {
      /* TODO (VPM) */
   }
   else
#endif
   {
      if (solver_type != 9 && solver_type != 19)
      {
         /* Create array to store pivots */
         Apiv = hypre_TAlloc(HYPRE_Int, global_num_rows, HYPRE_MEMORY_HOST);
         hypre_ParAMGDataAPiv(amg_data) = Apiv;

         /* Perform factorization */
         hypre_dgetrf(&global_num_rows, &global_num_rows,
                      hypre_ParAMGDataAMat(amg_data),
                      &global_num_rows, Apiv, &ierr);
         if (ierr != 0)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Problem with dgetrf!");
            return hypre_error_flag;
         }

         /* Compute explicit inverse */
         if (solver_type == 199)
         {
            HYPRE_Int     query = -1, lwork;
            HYPRE_Real    lwork_opt, *work;

            Ainv = hypre_TAlloc(HYPRE_Real, num_rows * global_num_rows, HYPRE_MEMORY_HOST);

            /* hypre_dgetri query phase */
            hypre_dgetri(&global_num_rows, hypre_ParAMGDataAMat(amg_data),
                         &global_num_rows, Apiv, &lwork_opt, &query, &ierr);
            if (ierr != 0)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Problem with dgetri (query)!");
               return hypre_error_flag;
            }

            lwork = (HYPRE_Int) lwork_opt;
            work  = hypre_TAlloc(HYPRE_Real, lwork, HYPRE_MEMORY_HOST);
            hypre_dgetri(&global_num_rows, hypre_ParAMGDataAMat(amg_data),
                         &global_num_rows, Apiv, work, &lwork, &ierr);
            if (ierr != 0)
            {
               hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Problem with dgetri!");
               return hypre_error_flag;
            }

            /* Store explicit dense inverse in Ainv */
            AT_mat = hypre_ParAMGDataAMat(amg_data);
            for (i = 0; i < global_num_rows; i++)
            {
               for (jj = 0; jj < num_rows; jj++)
               {
                  Ainv[i * num_rows + jj] = AT_mat[i * global_num_rows + jj + first_row_index];
               }
            }

            hypre_TFree(AT_mat, HYPRE_MEMORY_HOST);
            hypre_TFree(work,   HYPRE_MEMORY_HOST);

            hypre_ParAMGDataAInv(amg_data) = Ainv;
            hypre_ParAMGDataAMat(amg_data) = NULL;
         }
      }
      else if (solver_type == 9 || solver_type == 19)
      {
         /* Work space for performing gaussian elimination */
         Ainv = hypre_TAlloc(HYPRE_Real, global_size, HYPRE_MEMORY_HOST);
         hypre_ParAMGDataAInv(amg_data) = Ainv;
      }
   }

   /*-----------------------------------------------------------------
    *  Finalize
    *-----------------------------------------------------------------*/

   hypre_ParAMGDataGSSetup(amg_data) = 1;

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SETUP] += hypre_MPI_Wtime();
#endif
   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GaussElimSolve
 *
 * Gaussian elimination solve. See hypre_GaussElimSetup for comments.
 *
 * TODO (VPM): remove (u/f)_data_h. Communicate device buffers instead.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GaussElimSolve(hypre_ParAMGData *amg_data,
                     HYPRE_Int         level,
                     HYPRE_Int         solver_type)
{
   hypre_ParCSRMatrix  * A               = hypre_ParAMGDataAArray(amg_data)[level];
   HYPRE_Int             first_row_index = (HYPRE_Int) hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int             global_num_rows = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int             num_rows        = hypre_ParCSRMatrixNumRows(A);
   HYPRE_MemoryLocation  memory_location = hypre_ParCSRMatrixMemoryLocation(A);

   HYPRE_Real           *b_vec           = hypre_ParAMGDataBVec(amg_data);
   hypre_ParVector      *f               = hypre_ParAMGDataFArray(amg_data)[level];
   HYPRE_Real           *f_data          = hypre_VectorData(hypre_ParVectorLocalVector(f));
   HYPRE_Real           *f_data_h        = NULL;
   hypre_Vector         *f_vector        = NULL;

   hypre_ParVector      *u               = hypre_ParAMGDataUArray(amg_data)[level];
   HYPRE_Real           *u_data          = hypre_VectorData(hypre_ParVectorLocalVector(u));
   HYPRE_Real           *u_data_h        = NULL;

   /* Coarse solver data */
   HYPRE_Int            *A_piv           = hypre_ParAMGDataAPiv(amg_data);
   HYPRE_Real           *A_mat           = hypre_ParAMGDataAMat(amg_data);
   HYPRE_Real           *A_inv           = hypre_ParAMGDataAInv(amg_data);

   /* Constants */
   HYPRE_Int             one_i           = 1;
   HYPRE_Real            one             = 1.0;
   HYPRE_Real            zero            = 0.0;

   /* Local variables */
   MPI_Comm              new_comm        = hypre_ParAMGDataNewComm(amg_data);
   HYPRE_Int            *comm_info       = hypre_ParAMGDataCommInfo(amg_data);
   HYPRE_Int             global_size     = global_num_rows * global_num_rows;
   HYPRE_Int            *displs, *info;
   HYPRE_Int             new_num_procs;
   HYPRE_Int             ierr  = 0;

   /*-----------------------------------------------------------------
    *  Sanity checks
    *-----------------------------------------------------------------*/

   /* Call setup if not done before */
   if (hypre_ParAMGDataGSSetup(amg_data) == 0)
   {
      hypre_GaussElimSetup(amg_data, level, solver_type);
   }

   /* Check for relaxation type */
   if (solver_type != 9  && solver_type != 99 && solver_type != 199 &&
       solver_type != 19 && solver_type != 98)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Wrong relaxation type!");
      return hypre_error_flag;
   }

   /* Exit if no rows in this rank */
   if (!num_rows)
   {
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    *  Gather RHS phase
    *-----------------------------------------------------------------*/

   if (solver_type == 9 || solver_type == 99 || solver_type == 199)
   {
      hypre_MPI_Comm_size(new_comm, &new_num_procs);
      info   = &comm_info[0];
      displs = &comm_info[new_num_procs];

      if (hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(f)) != hypre_MEMORY_HOST)
      {
         f_data_h = hypre_TAlloc(HYPRE_Real, num_rows, HYPRE_MEMORY_HOST);

         hypre_TMemcpy(f_data_h, f_data, HYPRE_Real, num_rows, HYPRE_MEMORY_HOST,
                       hypre_ParVectorMemoryLocation(f));
      }
      else
      {
         f_data_h = f_data;
      }

      /* solver_type 199 needs a work space for the solution vector */
      if (solver_type == 199)
      {
         if (hypre_GetActualMemLocation(hypre_ParVectorMemoryLocation(f)) != hypre_MEMORY_HOST)
         {
            u_data_h = hypre_TAlloc(HYPRE_Real, num_rows, HYPRE_MEMORY_HOST);
         }
         else
         {
            u_data_h = u_data;
         }
      }

      /* TODO (VPM): Add GPU-aware MPI support to buffers */
      hypre_MPI_Allgatherv(f_data_h, num_rows, HYPRE_MPI_REAL, b_vec, info,
                           displs, HYPRE_MPI_REAL, new_comm);

      if (f_data_h != f_data)
      {
         hypre_TFree(f_data_h, HYPRE_MEMORY_HOST);
      }
   }
   else /* if (solver_type == 19 || solver_type == 99) */
   {
      f_vector = hypre_ParVectorToVectorAll(f, HYPRE_MEMORY_HOST);
      f_data   = hypre_VectorData(f_vector);
   }

   /*-----------------------------------------------------------------
    *  Gaussian elimination solve
    *-----------------------------------------------------------------*/

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SOLVE] -= hypre_MPI_Wtime();
#endif
   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("GaussElimSolve");

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy  exec = hypre_GetExecPolicy1(memory_location);

   if (exec == HYPRE_EXEC_DEVICE && (solver_type != 9 && solver_type != 19))
   {
      /* TODO (VPM) */
   }
   else
#endif
   {
      if (solver_type == 9 || solver_type == 19)
      {
         hypre_TMemcpy(A_inv, A_mat, HYPRE_Real, global_size,
                       HYPRE_MEMORY_HOST, HYPRE_MEMORY_HOST);
         b_vec = (solver_type == 9) ? b_vec : f_data;

         /* Run hypre's internal gaussian elimination */
         hypre_gselim(A_inv, b_vec, global_num_rows, ierr);
         if (ierr != 0)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Problem with hypre_gselim!");
         }

         hypre_TMemcpy(u_data, b_vec + first_row_index, HYPRE_Real, num_rows,
                       memory_location, HYPRE_MEMORY_HOST);
      }
      else if (solver_type == 98 || solver_type == 99)
      {
         b_vec = (solver_type == 99) ? b_vec : f_data;

         /* Run LAPACK's triangular solver */
         hypre_dgetrs("N", &global_num_rows, &one_i, A_mat,
                      &global_num_rows, A_piv, b_vec,
                      &global_num_rows, &ierr);
         if (ierr != 0)
         {
            hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Problem with hypre_dgetrs!");
         }

         hypre_TMemcpy(u_data, b_vec + first_row_index, HYPRE_Real, num_rows,
                       memory_location, HYPRE_MEMORY_HOST);
      }
      else /* if (solver_type == 199)*/
      {
         hypre_dgemv("N", &num_rows, &global_num_rows, &one,
                     A_inv, &num_rows, b_vec, &one_i, &zero,
                     u_data_h, &one_i);

         if (u_data_h != u_data)
         {
            hypre_TMemcpy(u_data, u_data_h + first_row_index, HYPRE_Real, num_rows,
                          memory_location, HYPRE_MEMORY_HOST);
         }
      }
   }

#ifdef HYPRE_PROFILE
   hypre_profile_times[HYPRE_TIMER_ID_GS_ELIM_SOLVE] += hypre_MPI_Wtime();
#endif

   /* Free memory - TODO (VPM): do we need to create and destroy f_vector at every solve call? */
   hypre_SeqVectorDestroy(f_vector);
   if (u_data_h != u_data)
   {
      hypre_TFree(u_data_h, HYPRE_MEMORY_HOST);
   }

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
