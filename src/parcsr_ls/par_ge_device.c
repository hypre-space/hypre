/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

/*--------------------------------------------------------------------------
 * hypre_GaussElimSetupDevice
 *
 * Gaussian elimination setup routine on the device. This uses MAGMA by
 * default or any of the vendor math libraries (cuBLAS, rocSOLVER) when
 * MAGMA is not available.
 *
 * See hypre_GaussElimSetup for more info.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GaussElimSetupDevice(hypre_ParAMGData *amg_data,
                           HYPRE_Int         level,
                           HYPRE_Int         solver_type)
{
   /* Input data */
   hypre_ParCSRMatrix  *par_A           = hypre_ParAMGDataAArray(amg_data)[level];
   HYPRE_Int            global_num_rows = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(par_A);
   HYPRE_Int            num_rows        = hypre_ParCSRMatrixNumRows(par_A);
   HYPRE_Int           *A_piv           = hypre_ParAMGDataAPiv(amg_data);
   HYPRE_Real          *A_mat           = hypre_ParAMGDataAMat(amg_data);
   HYPRE_Real          *A_work          = hypre_ParAMGDataAWork(amg_data);
   HYPRE_Int            global_size     = global_num_rows * global_num_rows;

   /* Local variables */
   HYPRE_Int            buffer_size     = 0;
   HYPRE_Int            ierr            = 0;
   HYPRE_Int           *d_ierr          = NULL;
   char                 msg[1024];

   /* Sanity checks */
   if (!num_rows || !global_size)
   {
      return hypre_error_flag;
   }

   if (global_size < 0)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Detected overflow!");
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    *  Compute the factorization A = L*U
    *-----------------------------------------------------------------*/

#if defined(HYPRE_USING_MAGMA)
   HYPRE_MAGMA_CALL(hypre_magma_getrf_nat(global_num_rows,
                                          global_num_rows,
                                          A_mat,
                                          global_num_rows,
                                          A_piv,
                                          &ierr));

#elif defined(HYPRE_USING_CUSOLVER)
   /* Allocate space for device error code */
   d_ierr = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);

   /* Compute buffer size */
   HYPRE_CUSOLVER_CALL(hypre_cusolver_dngetrf_bs(hypre_HandleVendorSolverHandle(hypre_handle()),
                                                 global_num_rows,
                                                 global_num_rows,
                                                 A_mat,
                                                 global_num_rows,
                                                 &buffer_size));

   /* We use A_work as workspace */
   if (buffer_size > global_size)
   {
      A_work = hypre_TReAlloc_v2(A_work, HYPRE_Real, global_size, HYPRE_Real, buffer_size,
                                 HYPRE_MEMORY_DEVICE);
      hypre_ParAMGDataAWork(amg_data) = A_work;
   }

   /* Factorize */
   HYPRE_CUSOLVER_CALL(hypre_cusolver_dngetrf(hypre_HandleVendorSolverHandle(hypre_handle()),
                                              global_num_rows,
                                              global_num_rows,
                                              A_mat,
                                              global_num_rows,
                                              A_work,
                                              A_piv,
                                              d_ierr));

   /* Move error code to host */
   hypre_TMemcpy(&ierr, d_ierr, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);

#elif defined(HYPRE_USING_ROCSOLVER)

   /**************
    * TODO (VPM) *
    **************/

#else
   /* Silence declared but never referenced warnings */
   (A_piv  += 0);
   (A_mat  += 0);
   (A_work += 0);
   (buffer_size *= 1);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "Missing dependency library for running gaussian elimination!");
#endif

   /* Free memory */
   hypre_TFree(d_ierr, HYPRE_MEMORY_DEVICE);

   if (ierr < 0)
   {
      hypre_sprintf(msg, "Problem with getrf's %d-th input argument", -ierr);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }
   else if (ierr > 0)
   {
      hypre_sprintf(msg, "Found that U(%d, %d) = 0", ierr, ierr);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   /*-----------------------------------------------------------------
    *  Compute the explicit inverse: A^{-1} = inv(A)
    *-----------------------------------------------------------------*/

   if (solver_type == 198 || solver_type == 199)
   {
#if defined(HYPRE_USING_MAGMA)
      /* Determine workspace size */
      buffer_size = global_num_rows * hypre_magma_getri_nb(global_num_rows);

      /* We use A_work as workspace */
      if (buffer_size > global_size)
      {
         A_work = hypre_TReAlloc_v2(A_work, HYPRE_Real, global_size, HYPRE_Real, buffer_size,
                                    HYPRE_MEMORY_DEVICE);
         hypre_ParAMGDataAWork(amg_data) = A_work;
      }

      HYPRE_MAGMA_CALL(hypre_magma_getri_gpu(global_num_rows,
                                             A_mat,
                                             global_num_rows,
                                             A_piv,
                                             A_work,
                                             buffer_size,
                                             &ierr));

#elif defined(HYPRE_USING_CUSOLVER)
      /* Allocate space for device error code */
      d_ierr = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);

      /* Create identity dense matrix */
      hypre_Memset((void*) A_work, 0,
                   (size_t) global_size * sizeof(HYPRE_Real),
                   HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL(for_each,
                        thrust::make_counting_iterator(0),
                        thrust::make_counting_iterator(global_num_rows),
                        hypreFunctor_DenseMatrixIdentity(global_num_rows, A_work));

      /* Compute inverse */
      HYPRE_CUSOLVER_CALL(hypre_cusolver_dngetrs(hypre_HandleVendorSolverHandle(hypre_handle()),
                                                 CUBLAS_OP_N,
                                                 global_num_rows,
                                                 global_num_rows,
                                                 A_mat,
                                                 global_num_rows,
                                                 A_piv,
                                                 A_work,
                                                 global_num_rows,
                                                 d_ierr));

      /* Store the inverse in A_mat */
      hypre_TMemcpy(A_mat, A_work, HYPRE_Real, global_size,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      /* Free memory */
      hypre_TFree(d_ierr, HYPRE_MEMORY_DEVICE);

#elif defined(HYPRE_USING_ROCSOLVER)

      /**************
       * TODO (VPM) *
       **************/

#else
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Missing dependency library for running gaussian elimination!");
#endif
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GaussElimSolveDevice
 *
 * Gaussian elimination solve routine on the device.
 *
 * See hypre_GaussElimSolve and hypre_GaussElimSetupDevice for more info.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GaussElimSolveDevice(hypre_ParAMGData *amg_data,
                           HYPRE_Int         level,
                           HYPRE_Int         solver_type)
{
   /* Input variables */
   hypre_ParCSRMatrix   *A                  = hypre_ParAMGDataAArray(amg_data)[level];
   HYPRE_Int             global_num_rows    = (HYPRE_Int) hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int             first_row_index    = (HYPRE_Int) hypre_ParCSRMatrixFirstRowIndex(A);
   HYPRE_Int             num_rows           = hypre_ParCSRMatrixNumRows(A);
   HYPRE_MemoryLocation  memory_location    = hypre_ParCSRMatrixMemoryLocation(A);
   HYPRE_MemoryLocation  ge_memory_location = hypre_ParAMGDataGEMemoryLocation(amg_data);

   hypre_ParVector      *par_u              = hypre_ParAMGDataUArray(amg_data)[level];
   HYPRE_Real           *u_data             = hypre_VectorData(hypre_ParVectorLocalVector(par_u));

   HYPRE_Real           *b_vec              = hypre_ParAMGDataBVec(amg_data);
   HYPRE_Real           *u_vec              = hypre_ParAMGDataUVec(amg_data);
   HYPRE_Real           *A_mat              = hypre_ParAMGDataAMat(amg_data);
   HYPRE_Int            *A_piv              = hypre_ParAMGDataAPiv(amg_data);

   /* Local variables */
   HYPRE_Real           *work;
   HYPRE_Int            *d_ierr = NULL;
   HYPRE_Int             ierr   = 0;
   HYPRE_Int             i_one  = 1;
   HYPRE_Complex         d_one  = 1.0;
   HYPRE_Complex         zero   = 0.0;
   char                  msg[1024];

   /* Sanity check */
   if (!num_rows || !global_num_rows)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_MAGMA)
   if (solver_type == 98 || solver_type == 99)
   {
      HYPRE_MAGMA_CALL(hypre_magma_getrs_gpu(MagmaNoTrans,
                                             global_num_rows,
                                             i_one,
                                             A_mat,
                                             global_num_rows,
                                             A_piv,
                                             b_vec,
                                             global_num_rows,
                                             &ierr));
   }
   else if (solver_type == 198 || solver_type == 199)
   {
      HYPRE_MAGMA_VCALL(hypre_magma_gemv(MagmaNoTrans,
                                         global_num_rows,
                                         global_num_rows,
                                         d_one,
                                         A_mat,
                                         global_num_rows,
                                         b_vec,
                                         i_one,
                                         zero,
                                         u_vec,
                                         i_one,
                                         hypre_HandleMagmaQueue(hypre_handle())));
   }

#elif defined(HYPRE_USING_CUSOLVER) && defined(HYPRE_USING_CUBLAS)
   if (solver_type == 98 || solver_type == 99)
   {
      d_ierr = hypre_CTAlloc(HYPRE_Int, 1, HYPRE_MEMORY_DEVICE);
      HYPRE_CUSOLVER_CALL(hypre_cusolver_dngetrs(hypre_HandleVendorSolverHandle(hypre_handle()),
                                                 CUBLAS_OP_N,
                                                 global_num_rows,
                                                 d_one,
                                                 A_mat,
                                                 global_num_rows,
                                                 A_piv,
                                                 b_vec,
                                                 global_num_rows,
                                                 d_ierr));
      hypre_TMemcpy(&ierr, d_ierr, HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   }
   else if (solver_type == 198 || solver_type == 199)
   {
      HYPRE_CUBLAS_CALL(hypre_cublas_gemv(hypre_HandleCublasHandle(hypre_handle()),
                                          CUBLAS_OP_N,
                                          global_num_rows,
                                          global_num_rows,
                                          &d_one,
                                          A_mat,
                                          global_num_rows,
                                          b_vec,
                                          i_one,
                                          &zero,
                                          u_vec,
                                          i_one));
   }
#elif defined(HYPRE_USING_ROCSOLVER)

   /**************
    * TODO (VPM) *
    **************/

#else
   /* Silence declared but never referenced warnings */
   (A_mat += 0);
   (A_piv += 0);
   (i_one *= 1);
   (d_one *= 1.0);
   (zero  *= zero);

   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "Missing dependency library for running gaussian elimination!");
#endif

   /* Check error code */
   if (ierr < 0)
   {
      hypre_sprintf(msg, "Problem with getrs' %d-th input argument", -ierr);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, msg);
      return hypre_error_flag;
   }

   /* Copy solution vector to proper variable */
   work = (solver_type == 198 || solver_type == 199) ? u_vec : b_vec;
   hypre_TMemcpy(u_data, work + first_row_index, HYPRE_Complex, num_rows,
                 memory_location, ge_memory_location);

   /* Free memory */
   hypre_TFree(d_ierr, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) */
