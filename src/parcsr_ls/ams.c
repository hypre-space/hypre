/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"
#include "float.h"
#include "ams.h"
#include "_hypre_utilities.hpp"

/*--------------------------------------------------------------------------
 * hypre_ParCSRRelax
 *
 * Relaxation on the ParCSR matrix A with right-hand side f and
 * initial guess u. Possible values for relax_type are:
 *
 * 1 = l1-scaled (or weighted) Jacobi
 * 2 = l1-scaled block Gauss-Seidel/SSOR
 * 3 = Kaczmarz
 * 4 = truncated version of 2 (Remark 6.2 in smoothers paper)
 * x = BoomerAMG relaxation with relax_type = |x|
 * (16 = Cheby)
 *
 * The default value of relax_type is 2.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRRelax( hypre_ParCSRMatrix *A,              /* matrix to relax with */
                   hypre_ParVector    *f,              /* right-hand side */
                   HYPRE_Int           relax_type,     /* relaxation type */
                   HYPRE_Int           relax_times,    /* number of sweeps */
                   HYPRE_Real         *l1_norms,       /* l1 norms of the rows of A */
                   HYPRE_Real          relax_weight,   /* damping coefficient (usually <= 1) */
                   HYPRE_Real          omega,          /* SOR parameter (usually in (0,2) */
                   HYPRE_Real          max_eig_est,    /* for cheby smoothers */
                   HYPRE_Real          min_eig_est,
                   HYPRE_Int           cheby_order,
                   HYPRE_Real          cheby_fraction,
                   hypre_ParVector    *u,              /* initial/updated approximation */
                   hypre_ParVector    *v,              /* temporary vector */
                   hypre_ParVector    *z               /* temporary vector */ )
{
   HYPRE_Int sweep;

   for (sweep = 0; sweep < relax_times; sweep++)
   {
      if (relax_type == 1) /* l1-scaled Jacobi */
      {
         hypre_BoomerAMGRelax(A, f, NULL, 7, 0, relax_weight, 1.0, l1_norms, u, v, z);
      }
      else if (relax_type == 2 || relax_type == 4) /* offd-l1-scaled block GS */
      {
         /* !!! Note: relax_weight and omega flipped !!! */
         hypre_BoomerAMGRelaxHybridSOR(A, f, NULL, 0, omega,
                                       relax_weight, l1_norms, u, v, z, 1, 1, 0, 1);
      }
      else if (relax_type == 3) /* Kaczmarz */
      {
         hypre_BoomerAMGRelax(A, f, NULL, 20, 0, relax_weight, omega, l1_norms, u, v, z);
      }
      else /* call BoomerAMG relaxation */
      {
         if (relax_type == 16)
         {
            hypre_ParCSRRelax_Cheby(A, f, max_eig_est, min_eig_est, cheby_fraction, cheby_order, 1,
                                    0, u, v, z);
         }
         else
         {
            hypre_BoomerAMGRelax(A, f, NULL, hypre_abs(relax_type), 0, relax_weight,
                                 omega, l1_norms, u, v, z);
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInRangeOf
 *
 * Return a vector that belongs to the range of a given matrix.
 *--------------------------------------------------------------------------*/

hypre_ParVector *hypre_ParVectorInRangeOf(hypre_ParCSRMatrix *A)
{
   hypre_ParVector *x;

   x = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize(x);
   hypre_ParVectorOwnsData(x) = 1;

   return x;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInDomainOf
 *
 * Return a vector that belongs to the domain of a given matrix.
 *--------------------------------------------------------------------------*/

hypre_ParVector *hypre_ParVectorInDomainOf(hypre_ParCSRMatrix *A)
{
   hypre_ParVector *x;

   x = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumCols(A),
                             hypre_ParCSRMatrixColStarts(A));
   hypre_ParVectorInitialize(x);
   hypre_ParVectorOwnsData(x) = 1;

   return x;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorBlockSplit
 *
 * Extract the dim sub-vectors x_0,...,x_{dim-1} composing a parallel
 * block vector x. It is assumed that &x[i] = [x_0[i],...,x_{dim-1}[i]].
 *--------------------------------------------------------------------------*/
#if defined(HYPRE_USING_GPU)
template<HYPRE_Int dir>
__global__ void
hypreGPUKernel_ParVectorBlockSplitGather(hypre_DeviceItem &item,
                                         HYPRE_Int   size,
                                         HYPRE_Int   dim,
                                         HYPRE_Real *x0,
                                         HYPRE_Real *x1,
                                         HYPRE_Real *x2,
                                         HYPRE_Real *x)
{
   const HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i >= size * dim)
   {
      return;
   }

   HYPRE_Real *xx[3];

   xx[0] = x0;
   xx[1] = x1;
   xx[2] = x2;

   const HYPRE_Int d = i % dim;
   const HYPRE_Int k = i / dim;

   if (dir == 0)
   {
      xx[d][k] = x[i];
   }
   else if (dir == 1)
   {
      x[i] = xx[d][k];
   }
}
#endif

HYPRE_Int
hypre_ParVectorBlockSplit(hypre_ParVector *x,
                          hypre_ParVector *x_[3],
                          HYPRE_Int dim)
{
   HYPRE_Int i, d, size_;
   HYPRE_Real *x_data, *x_data_[3];

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParVectorMemoryLocation(x) );
#endif

   size_ = hypre_VectorSize(hypre_ParVectorLocalVector(x_[0]));

   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   for (d = 0; d < dim; d++)
   {
      x_data_[d] = hypre_VectorData(hypre_ParVectorLocalVector(x_[d]));
   }

#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(size_ * dim, "thread", bDim);
      HYPRE_GPU_LAUNCH( hypreGPUKernel_ParVectorBlockSplitGather<0>, gDim, bDim,
                        size_, dim, x_data_[0], x_data_[1], x_data_[2], x_data);
   }
   else
#endif
   {
      for (i = 0; i < size_; i++)
      {
         for (d = 0; d < dim; d++)
         {
            x_data_[d][i] = x_data[dim * i + d];
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorBlockGather
 *
 * Compose a parallel block vector x from dim given sub-vectors
 * x_0,...,x_{dim-1}, such that &x[i] = [x_0[i],...,x_{dim-1}[i]].
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorBlockGather(hypre_ParVector *x,
                           hypre_ParVector *x_[3],
                           HYPRE_Int dim)
{
   HYPRE_Int i, d, size_;
   HYPRE_Real *x_data, *x_data_[3];

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParVectorMemoryLocation(x) );
#endif

   size_ = hypre_VectorSize(hypre_ParVectorLocalVector(x_[0]));

   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   for (d = 0; d < dim; d++)
   {
      x_data_[d] = hypre_VectorData(hypre_ParVectorLocalVector(x_[d]));
   }

#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(size_ * dim, "thread", bDim);
      HYPRE_GPU_LAUNCH( hypreGPUKernel_ParVectorBlockSplitGather<1>, gDim, bDim,
                        size_, dim, x_data_[0], x_data_[1], x_data_[2], x_data);
   }
   else
#endif
   {
      for (i = 0; i < size_; i++)
      {
         for (d = 0; d < dim; d++)
         {
            x_data[dim * i + d] = x_data_[d][i];
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_BoomerAMGBlockSolve
 *
 * Apply the block-diagonal solver diag(B) to the system diag(A) x = b.
 * Here B is a given BoomerAMG solver for A, while x and b are "block"
 * parallel vectors.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_BoomerAMGBlockSolve(void *B,
                                    hypre_ParCSRMatrix *A,
                                    hypre_ParVector *b,
                                    hypre_ParVector *x)
{
   HYPRE_Int d, dim = 1;

   hypre_ParVector *b_[3] = {NULL, NULL, NULL};
   hypre_ParVector *x_[3] = {NULL, NULL, NULL};

   dim = hypre_ParVectorGlobalSize(x) / hypre_ParCSRMatrixGlobalNumRows(A);

   if (dim == 1)
   {
      hypre_BoomerAMGSolve(B, A, b, x);
      return hypre_error_flag;
   }

   for (d = 0; d < dim; d++)
   {
      b_[d] = hypre_ParVectorInRangeOf(A);
      x_[d] = hypre_ParVectorInRangeOf(A);
   }

   hypre_ParVectorBlockSplit(b, b_, dim);
   hypre_ParVectorBlockSplit(x, x_, dim);

   for (d = 0; d < dim; d++)
   {
      hypre_BoomerAMGSolve(B, A, b_[d], x_[d]);
   }

   hypre_ParVectorBlockGather(x, x_, dim);

   for (d = 0; d < dim; d++)
   {
      hypre_ParVectorDestroy(b_[d]);
      hypre_ParVectorDestroy(x_[d]);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixFixZeroRows
 *
 * For every zero row in the matrix: set the diagonal element to 1.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRMatrixFixZeroRowsHost(hypre_ParCSRMatrix *A)
{
   HYPRE_Int i, j;
   HYPRE_Real l1_norm;
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int *A_diag_I = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *A_offd_I = hypre_CSRMatrixI(A_offd);
   HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   /* a row will be considered zero if its l1 norm is less than eps */
   HYPRE_Real eps = 0.0; /* DBL_EPSILON * 1e+4; */

   for (i = 0; i < num_rows; i++)
   {
      l1_norm = 0.0;
      for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
      {
         l1_norm += hypre_abs(A_diag_data[j]);
      }
      if (num_cols_offd)
         for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
         {
            l1_norm += hypre_abs(A_offd_data[j]);
         }

      if (l1_norm <= eps)
      {
         for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
            if (A_diag_J[j] == i)
            {
               A_diag_data[j] = 1.0;
            }
            else
            {
               A_diag_data[j] = 0.0;
            }
         if (num_cols_offd)
            for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
            {
               A_offd_data[j] = 0.0;
            }
      }
   }

   return hypre_error_flag;
}

#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_ParCSRMatrixFixZeroRows( hypre_DeviceItem    &item,
                                        HYPRE_Int      nrows,
                                        HYPRE_Int     *A_diag_i,
                                        HYPRE_Int     *A_diag_j,
                                        HYPRE_Complex *A_diag_data,
                                        HYPRE_Int     *A_offd_i,
                                        HYPRE_Complex *A_offd_data,
                                        HYPRE_Int      num_cols_offd)
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Real eps = 0.0; /* DBL_EPSILON * 1e+4; */
   HYPRE_Real l1_norm = 0.0;
   HYPRE_Int p1 = 0, q1, p2 = 0, q2 = 0;

   if (lane < 2)
   {
      p1 = read_only_load(A_diag_i + row_i + lane);
      if (num_cols_offd)
      {
         p2 = read_only_load(A_offd_i + row_i + lane);
      }
   }

   q1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 0);
   if (num_cols_offd)
   {
      q2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (HYPRE_Int j = p1 + lane; j < q1; j += HYPRE_WARP_SIZE)
   {
      l1_norm += hypre_abs(A_diag_data[j]);
   }

   for (HYPRE_Int j = p2 + lane; j < q2; j += HYPRE_WARP_SIZE)
   {
      l1_norm += hypre_abs(A_offd_data[j]);
   }

   l1_norm = warp_allreduce_sum(item, l1_norm);

   if (l1_norm <= eps)
   {
      for (HYPRE_Int j = p1 + lane; j < q1; j += HYPRE_WARP_SIZE)
      {
         if (row_i == read_only_load(&A_diag_j[j]))
         {
            A_diag_data[j] = 1.0;
         }
         else
         {
            A_diag_data[j] = 0.0;
         }
      }

      for (HYPRE_Int j = p2 + lane; j < q2; j += HYPRE_WARP_SIZE)
      {
         A_offd_data[j] = 0.0;
      }
   }
}

HYPRE_Int hypre_ParCSRMatrixFixZeroRowsDevice(hypre_ParCSRMatrix *A)
{
   HYPRE_Int        nrows         = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix *A_diag        = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real      *A_diag_data   = hypre_CSRMatrixData(A_diag);
   HYPRE_Int       *A_diag_i      = hypre_CSRMatrixI(A_diag);
   HYPRE_Int       *A_diag_j      = hypre_CSRMatrixJ(A_diag);
   hypre_CSRMatrix *A_offd        = hypre_ParCSRMatrixOffd(A);
   HYPRE_Real      *A_offd_data   = hypre_CSRMatrixData(A_offd);
   HYPRE_Int       *A_offd_i      = hypre_CSRMatrixI(A_offd);
   HYPRE_Int        num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   HYPRE_GPU_LAUNCH(hypreGPUKernel_ParCSRMatrixFixZeroRows, gDim, bDim,
                    nrows, A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_data, num_cols_offd);

   //hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}
#endif

HYPRE_Int hypre_ParCSRMatrixFixZeroRows(hypre_ParCSRMatrix *A)
{
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      return hypre_ParCSRMatrixFixZeroRowsDevice(A);
   }
   else
#endif
   {
      return hypre_ParCSRMatrixFixZeroRowsHost(A);
   }
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRComputeL1Norms
 *
 * Compute the l1 norms of the rows of a given matrix, depending on
 * the option parameter:
 *
 * option 1 = Compute the l1 norm of the rows
 * option 2 = Compute the l1 norm of the (processor) off-diagonal
 *            part of the rows plus the diagonal of A
 * option 3 = Compute the l2 norm^2 of the rows
 * option 4 = Truncated version of option 2 based on Remark 6.2 in "Multigrid
 *            Smoothers for Ultra-Parallel Computing"
 *
 * The above computations are done in a CF manner, whenever the provided
 * cf_marker is not NULL.
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_USING_SYCL)
struct l1_norm_op1
#else
struct l1_norm_op1 : public thrust::binary_function<HYPRE_Complex, HYPRE_Complex, HYPRE_Complex>
#endif
{
   __host__ __device__
   HYPRE_Complex operator()(const HYPRE_Complex &x, const HYPRE_Complex &y) const
   {
      return x <= 4.0 / 3.0 * y ? y : x;
   }
};
#endif

#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_USING_SYCL)
struct l1_norm_op6
#else
struct l1_norm_op6 : public thrust::binary_function<HYPRE_Complex, HYPRE_Complex, HYPRE_Complex>
#endif
{
   __host__ __device__
   HYPRE_Complex operator()(const HYPRE_Complex &d, const HYPRE_Complex &l) const
   {
      return (l + d + sqrt(l * l + d * d)) * 0.5;
   }
};
#endif

HYPRE_Int hypre_ParCSRComputeL1Norms(hypre_ParCSRMatrix  *A,
                                     HYPRE_Int            option,
                                     HYPRE_Int           *cf_marker,
                                     HYPRE_Real         **l1_norm_ptr)
{
   HYPRE_Int i;
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);
   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   HYPRE_MemoryLocation memory_location_l1 = hypre_ParCSRMatrixMemoryLocation(A);

   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( memory_location_l1 );

   if (exec == HYPRE_EXEC_HOST)
   {
      HYPRE_Int num_threads = hypre_NumThreads();
      if (num_threads > 1)
      {
         return hypre_ParCSRComputeL1NormsThreads(A, option, num_threads, cf_marker, l1_norm_ptr);
      }
   }

   HYPRE_Real *l1_norm = hypre_TAlloc(HYPRE_Real, num_rows, memory_location_l1);

   HYPRE_MemoryLocation memory_location_tmp =
      exec == HYPRE_EXEC_HOST ? HYPRE_MEMORY_HOST : HYPRE_MEMORY_DEVICE;

   HYPRE_Real *diag_tmp = NULL;

   HYPRE_Int *cf_marker_offd = NULL;

   /* collect the cf marker data from other procs */
   if (cf_marker != NULL)
   {
      HYPRE_Int num_sends;
      HYPRE_Int *int_buf_data = NULL;

      hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      hypre_ParCSRCommHandle *comm_handle;

      if (num_cols_offd)
      {
         cf_marker_offd = hypre_CTAlloc(HYPRE_Int, num_cols_offd, memory_location_tmp);
      }
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends))
      {
         int_buf_data = hypre_CTAlloc(HYPRE_Int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends),
                                      memory_location_tmp);
      }
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         hypre_ParCSRCommPkgCopySendMapElmtsToDevice(comm_pkg);
#if defined(HYPRE_USING_SYCL)
         hypreSycl_gather( hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                           hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                                                                                             num_sends),
                           cf_marker,
                           int_buf_data );
#else
         HYPRE_THRUST_CALL( gather,
                            hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg),
                            hypre_ParCSRCommPkgDeviceSendMapElmts(comm_pkg) + hypre_ParCSRCommPkgSendMapStart(comm_pkg,
                                  num_sends),
                            cf_marker,
                            int_buf_data );
#endif

#if defined(HYPRE_USING_THRUST_NOSYNC)
         /* RL: make sure int_buf_data is ready before issuing GPU-GPU MPI */
         if (hypre_GetGpuAwareMPI())
         {
            hypre_ForceSyncComputeStream(hypre_handle());
         }
#endif
      }
      else
#endif
      {
         HYPRE_Int index = 0;
         HYPRE_Int start;
         HYPRE_Int j;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
            {
               int_buf_data[index++] = cf_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
            }
         }
      }

      comm_handle = hypre_ParCSRCommHandleCreate_v2(11, comm_pkg, memory_location_tmp, int_buf_data,
                                                    memory_location_tmp, cf_marker_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, memory_location_tmp);
   }

   if (option == 1)
   {
      /* Set the l1 norm of the diag part */
      hypre_CSRMatrixComputeRowSum(A_diag, cf_marker, cf_marker, l1_norm, 1, 1.0, "set");

      /* Add the l1 norm of the offd part */
      if (num_cols_offd)
      {
         hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker_offd, l1_norm, 1, 1.0, "add");
      }
   }
   else if (option == 2)
   {
      /* Set the abs(diag) element */
      hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 1);
      /* Add the l1 norm of the offd part */
      if (num_cols_offd)
      {
         hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker, l1_norm, 1, 1.0, "add");
      }
   }
   else if (option == 3)
   {
      /* Set the CF l2 norm of the diag part */
      hypre_CSRMatrixComputeRowSum(A_diag, NULL, NULL, l1_norm, 2, 1.0, "set");
      /* Add the CF l2 norm of the offd part */
      if (num_cols_offd)
      {
         hypre_CSRMatrixComputeRowSum(A_offd, NULL, NULL, l1_norm, 2, 1.0, "add");
      }
   }
   else if (option == 4)
   {
      /* Set the abs(diag) element */
      hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 1);

      diag_tmp = hypre_TAlloc(HYPRE_Real, num_rows, memory_location_tmp);
      hypre_TMemcpy(diag_tmp, l1_norm, HYPRE_Real, num_rows, memory_location_tmp, memory_location_l1);

      /* Add the scaled l1 norm of the offd part */
      if (num_cols_offd)
      {
         hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker_offd, l1_norm, 1, 0.5, "add");
      }

      /* Truncate according to Remark 6.2 */
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::transform, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm, l1_norm_op1() );
#else
         HYPRE_THRUST_CALL( transform, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm, l1_norm_op1() );
#endif
      }
      else
#endif
      {
         for (i = 0; i < num_rows; i++)
         {
            if (l1_norm[i] <= 4.0 / 3.0 * diag_tmp[i])
            {
               l1_norm[i] = diag_tmp[i];
            }
         }
      }
   }
   else if (option == 5) /*stores diagonal of A for Jacobi using matvec, rlx 7 */
   {
      /* Set the diag element */
      hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 0);

#if defined(HYPRE_USING_GPU)
      if ( exec == HYPRE_EXEC_DEVICE)
      {
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::replace_if, l1_norm, l1_norm + num_rows, [] (const auto & x) {return !x;},
         1.0 );
#else
         thrust::identity<HYPRE_Complex> identity;
         HYPRE_THRUST_CALL( replace_if, l1_norm, l1_norm + num_rows, thrust::not1(identity), 1.0 );
#endif
      }
      else
#endif
      {
         for (i = 0; i < num_rows; i++)
         {
            if (l1_norm[i] == 0.0)
            {
               l1_norm[i] = 1.0;
            }
         }
      }

      *l1_norm_ptr = l1_norm;

      return hypre_error_flag;
   }
   else if (option == 6)
   {
      /* Set the abs(diag) element */
      hypre_CSRMatrixExtractDiagonal(A_diag, l1_norm, 1);
      /* Add the scaled l1 norm of the offd part */
      if (num_cols_offd)
      {
         diag_tmp = hypre_TAlloc(HYPRE_Real, num_rows, memory_location_tmp);
         hypre_CSRMatrixComputeRowSum(A_offd, cf_marker, cf_marker_offd, diag_tmp, 1, 1.0, "set");
#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::transform, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm, l1_norm_op6() );
#else
            HYPRE_THRUST_CALL( transform, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm, l1_norm_op6() );
#endif
         }
         else
#endif
         {
            for (i = 0; i < num_rows; i++)
            {
               l1_norm[i] = 0.5 * (diag_tmp[i] + l1_norm[i] +
                                   hypre_sqrt(hypre_squared(diag_tmp[i]) + hypre_squared(l1_norm[i])));
            }
         }
      }
   }

   /* Handle negative definite matrices */
   if (!diag_tmp)
   {
      diag_tmp = hypre_TAlloc(HYPRE_Real, num_rows, memory_location_tmp);
   }

   /* Set the diag element */
   hypre_CSRMatrixExtractDiagonal(A_diag, diag_tmp, 0);

#if defined(HYPRE_USING_GPU)
   if (exec == HYPRE_EXEC_DEVICE)
   {
#if defined(HYPRE_USING_SYCL)
      hypreSycl_transform_if( l1_norm, l1_norm + num_rows, diag_tmp, l1_norm,
                              std::negate<HYPRE_Real>(),
                              is_negative<HYPRE_Real>() );
      bool any_zero = 0.0 == HYPRE_ONEDPL_CALL( std::reduce, l1_norm, l1_norm + num_rows, 1.0,
                                                oneapi::dpl::minimum<HYPRE_Real>() );
#else
      HYPRE_THRUST_CALL( transform_if, l1_norm, l1_norm + num_rows, diag_tmp, l1_norm,
                         thrust::negate<HYPRE_Real>(),
                         is_negative<HYPRE_Real>() );
      //bool any_zero = HYPRE_THRUST_CALL( any_of, l1_norm, l1_norm + num_rows, thrust::not1(thrust::identity<HYPRE_Complex>()) );
      bool any_zero = 0.0 == HYPRE_THRUST_CALL( reduce, l1_norm, l1_norm + num_rows, 1.0,
                                                thrust::minimum<HYPRE_Real>() );
#endif
      if ( any_zero )
      {
         hypre_error_in_arg(1);
      }
   }
   else
#endif
   {
      for (i = 0; i < num_rows; i++)
      {
         if (diag_tmp[i] < 0.0)
         {
            l1_norm[i] = -l1_norm[i];
         }
      }

      for (i = 0; i < num_rows; i++)
      {
         /* if (hypre_abs(l1_norm[i]) < DBL_EPSILON) */
         if (hypre_abs(l1_norm[i]) == 0.0)
         {
            hypre_error_in_arg(1);
            break;
         }
      }
   }

   hypre_TFree(cf_marker_offd, memory_location_tmp);
   hypre_TFree(diag_tmp, memory_location_tmp);

   *l1_norm_ptr = l1_norm;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixSetDiagRows
 *
 * For every row containing only a diagonal element: set it to d.
 *--------------------------------------------------------------------------*/
#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_ParCSRMatrixSetDiagRows(hypre_DeviceItem    &item,
                                       HYPRE_Int      nrows,
                                       HYPRE_Int     *A_diag_I,
                                       HYPRE_Int     *A_diag_J,
                                       HYPRE_Complex *A_diag_data,
                                       HYPRE_Int     *A_offd_I,
                                       HYPRE_Int      num_cols_offd,
                                       HYPRE_Real     d)
{
   const HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (i >= nrows)
   {
      return;
   }

   HYPRE_Int j = read_only_load(&A_diag_I[i]);

   if ( (read_only_load(&A_diag_I[i + 1]) == j + 1) && (read_only_load(&A_diag_J[j]) == i) &&
        (!num_cols_offd || (read_only_load(&A_offd_I[i + 1]) == read_only_load(&A_offd_I[i]))) )
   {
      A_diag_data[j] = d;
   }
}
#endif

HYPRE_Int hypre_ParCSRMatrixSetDiagRows(hypre_ParCSRMatrix *A, HYPRE_Real d)
{
   HYPRE_Int i, j;
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int *A_diag_I = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *A_offd_I = hypre_CSRMatrixI(A_offd);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
   if (exec == HYPRE_EXEC_DEVICE)
   {
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_rows, "thread", bDim);
      HYPRE_GPU_LAUNCH( hypreGPUKernel_ParCSRMatrixSetDiagRows, gDim, bDim,
                        num_rows, A_diag_I, A_diag_J, A_diag_data, A_offd_I, num_cols_offd, d);
   }
   else
#endif
   {
      for (i = 0; i < num_rows; i++)
      {
         j = A_diag_I[i];
         if ((A_diag_I[i + 1] == j + 1) && (A_diag_J[j] == i) &&
             (!num_cols_offd || (A_offd_I[i + 1] == A_offd_I[i])))
         {
            A_diag_data[j] = d;
         }
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSCreate
 *
 * Allocate the AMS solver structure.
 *--------------------------------------------------------------------------*/

void * hypre_AMSCreate(void)
{
   hypre_AMSData *ams_data;

   ams_data = hypre_CTAlloc(hypre_AMSData,  1, HYPRE_MEMORY_HOST);

   /* Default parameters */

   ams_data -> dim = 3;                /* 3D problem */
   ams_data -> maxit = 20;             /* perform at most 20 iterations */
   ams_data -> tol = 1e-6;             /* convergence tolerance */
   ams_data -> print_level = 1;        /* print residual norm at each step */
   ams_data -> cycle_type = 1;         /* a 3-level multiplicative solver */
   ams_data -> A_relax_type = 2;       /* offd-l1-scaled GS */
   ams_data -> A_relax_times = 1;      /* one relaxation sweep */
   ams_data -> A_relax_weight = 1.0;   /* damping parameter */
   ams_data -> A_omega = 1.0;          /* SSOR coefficient */
   ams_data -> A_cheby_order = 2;      /* Cheby: order (1 -4 are vaild) */
   ams_data -> A_cheby_fraction = .3;  /* Cheby: fraction of spectrum to smooth */

   ams_data -> B_G_coarsen_type = 10;  /* HMIS coarsening */
   ams_data -> B_G_agg_levels = 1;     /* Levels of aggressive coarsening */
   ams_data -> B_G_relax_type = 3;     /* hybrid G-S/Jacobi */
   ams_data -> B_G_theta = 0.25;       /* strength threshold */
   ams_data -> B_G_interp_type = 0;    /* interpolation type */
   ams_data -> B_G_Pmax = 0;           /* max nonzero elements in interp. rows */
   ams_data -> B_Pi_coarsen_type = 10; /* HMIS coarsening */
   ams_data -> B_Pi_agg_levels = 1;    /* Levels of aggressive coarsening */
   ams_data -> B_Pi_relax_type = 3;    /* hybrid G-S/Jacobi */
   ams_data -> B_Pi_theta = 0.25;      /* strength threshold */
   ams_data -> B_Pi_interp_type = 0;   /* interpolation type */
   ams_data -> B_Pi_Pmax = 0;          /* max nonzero elements in interp. rows */
   ams_data -> beta_is_zero = 0;       /* the problem has a mass term */

   /* By default, do l1-GS smoothing on the coarsest grid */
   ams_data -> B_G_coarse_relax_type  = 8;
   ams_data -> B_Pi_coarse_relax_type = 8;

   /* The rest of the fields are initialized using the Set functions */

   ams_data -> A    = NULL;
   ams_data -> G    = NULL;
   ams_data -> A_G  = NULL;
   ams_data -> B_G  = 0;
   ams_data -> Pi   = NULL;
   ams_data -> A_Pi = NULL;
   ams_data -> B_Pi = 0;
   ams_data -> x    = NULL;
   ams_data -> y    = NULL;
   ams_data -> z    = NULL;
   ams_data -> Gx   = NULL;
   ams_data -> Gy   = NULL;
   ams_data -> Gz   = NULL;

   ams_data -> r0  = NULL;
   ams_data -> g0  = NULL;
   ams_data -> r1  = NULL;
   ams_data -> g1  = NULL;
   ams_data -> r2  = NULL;
   ams_data -> g2  = NULL;
   ams_data -> zz  = NULL;

   ams_data -> Pix    = NULL;
   ams_data -> Piy    = NULL;
   ams_data -> Piz    = NULL;
   ams_data -> A_Pix  = NULL;
   ams_data -> A_Piy  = NULL;
   ams_data -> A_Piz  = NULL;
   ams_data -> B_Pix  = 0;
   ams_data -> B_Piy  = 0;
   ams_data -> B_Piz  = 0;

   ams_data -> interior_nodes       = NULL;
   ams_data -> G0                   = NULL;
   ams_data -> A_G0                 = NULL;
   ams_data -> B_G0                 = 0;
   ams_data -> projection_frequency = 5;

   ams_data -> A_l1_norms = NULL;
   ams_data -> A_max_eig_est = 0;
   ams_data -> A_min_eig_est = 0;

   ams_data -> owns_Pi   = 1;
   ams_data -> owns_A_G  = 0;
   ams_data -> owns_A_Pi = 0;

   return (void *) ams_data;
}

/*--------------------------------------------------------------------------
 * hypre_AMSDestroy
 *
 * Deallocate the AMS solver structure. Note that the input data (given
 * through the Set functions) is not destroyed.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSDestroy(void *solver)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   if (!ams_data)
   {
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (ams_data -> owns_A_G)
      if (ams_data -> A_G)
      {
         hypre_ParCSRMatrixDestroy(ams_data -> A_G);
      }
   if (!ams_data -> beta_is_zero)
      if (ams_data -> B_G)
      {
         HYPRE_BoomerAMGDestroy(ams_data -> B_G);
      }

   if (ams_data -> owns_Pi && ams_data -> Pi)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> Pi);
   }
   if (ams_data -> owns_A_Pi)
      if (ams_data -> A_Pi)
      {
         hypre_ParCSRMatrixDestroy(ams_data -> A_Pi);
      }
   if (ams_data -> B_Pi)
   {
      HYPRE_BoomerAMGDestroy(ams_data -> B_Pi);
   }

   if (ams_data -> owns_Pi && ams_data -> Pix)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> Pix);
   }
   if (ams_data -> A_Pix)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> A_Pix);
   }
   if (ams_data -> B_Pix)
   {
      HYPRE_BoomerAMGDestroy(ams_data -> B_Pix);
   }
   if (ams_data -> owns_Pi && ams_data -> Piy)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> Piy);
   }
   if (ams_data -> A_Piy)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> A_Piy);
   }
   if (ams_data -> B_Piy)
   {
      HYPRE_BoomerAMGDestroy(ams_data -> B_Piy);
   }
   if (ams_data -> owns_Pi && ams_data -> Piz)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> Piz);
   }
   if (ams_data -> A_Piz)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> A_Piz);
   }
   if (ams_data -> B_Piz)
   {
      HYPRE_BoomerAMGDestroy(ams_data -> B_Piz);
   }

   if (ams_data -> r0)
   {
      hypre_ParVectorDestroy(ams_data -> r0);
   }
   if (ams_data -> g0)
   {
      hypre_ParVectorDestroy(ams_data -> g0);
   }
   if (ams_data -> r1)
   {
      hypre_ParVectorDestroy(ams_data -> r1);
   }
   if (ams_data -> g1)
   {
      hypre_ParVectorDestroy(ams_data -> g1);
   }
   if (ams_data -> r2)
   {
      hypre_ParVectorDestroy(ams_data -> r2);
   }
   if (ams_data -> g2)
   {
      hypre_ParVectorDestroy(ams_data -> g2);
   }
   if (ams_data -> zz)
   {
      hypre_ParVectorDestroy(ams_data -> zz);
   }

   if (ams_data -> G0)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> A);
   }
   if (ams_data -> G0)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> G0);
   }
   if (ams_data -> A_G0)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> A_G0);
   }
   if (ams_data -> B_G0)
   {
      HYPRE_BoomerAMGDestroy(ams_data -> B_G0);
   }

   hypre_SeqVectorDestroy(ams_data -> A_l1_norms);

   /* G, x, y ,z, Gx, Gy and Gz are not destroyed */

   if (ams_data)
   {
      hypre_TFree(ams_data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetDimension
 *
 * Set problem dimension (2 or 3). By default we assume dim = 3.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetDimension(void *solver,
                                HYPRE_Int dim)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   if (dim != 1 && dim != 2 && dim != 3)
   {
      hypre_error_in_arg(2);
   }

   ams_data -> dim = dim;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetDiscreteGradient
 *
 * Set the discrete gradient matrix G.
 * This function should be called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetDiscreteGradient(void *solver,
                                       hypre_ParCSRMatrix *G)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> G = G;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetCoordinateVectors
 *
 * Set the x, y and z coordinates of the vertices in the mesh.
 *
 * Either SetCoordinateVectors or SetEdgeConstantVectors should be
 * called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetCoordinateVectors(void *solver,
                                        hypre_ParVector *x,
                                        hypre_ParVector *y,
                                        hypre_ParVector *z)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> x = x;
   ams_data -> y = y;
   ams_data -> z = z;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetEdgeConstantVectors
 *
 * Set the vectors Gx, Gy and Gz which give the representations of
 * the constant vector fields (1,0,0), (0,1,0) and (0,0,1) in the
 * edge element basis.
 *
 * Either SetCoordinateVectors or SetEdgeConstantVectors should be
 * called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetEdgeConstantVectors(void *solver,
                                          hypre_ParVector *Gx,
                                          hypre_ParVector *Gy,
                                          hypre_ParVector *Gz)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> Gx = Gx;
   ams_data -> Gy = Gy;
   ams_data -> Gz = Gz;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetInterpolations
 *
 * Set the (components of) the Nedelec interpolation matrix Pi=[Pix,Piy,Piz].
 *
 * This function is generally intended to be used only for high-order Nedelec
 * discretizations (in the lowest order case, Pi is constructed internally in
 * AMS from the discreet gradient matrix and the coordinates of the vertices),
 * though it can also be used in the lowest-order case or for other types of
 * discretizations (e.g. ones based on the second family of Nedelec elements).
 *
 * By definition, Pi is the matrix representation of the linear operator that
 * interpolates (high-order) vector nodal finite elements into the (high-order)
 * Nedelec space. The component matrices are defined as Pix phi = Pi (phi,0,0)
 * and similarly for Piy and Piz. Note that all these operators depend on the
 * choice of the basis and degrees of freedom in the high-order spaces.
 *
 * The column numbering of Pi should be node-based, i.e. the x/y/z components of
 * the first node (vertex or high-order dof) should be listed first, followed by
 * the x/y/z components of the second node and so on (see the documentation of
 * HYPRE_BoomerAMGSetDofFunc).
 *
 * If used, this function should be called before hypre_AMSSetup() and there is
 * no need to provide the vertex coordinates. Furthermore, only one of the sets
 * {Pi} and {Pix,Piy,Piz} needs to be specified (though it is OK to provide
 * both).  If Pix is NULL, then scalar Pi-based AMS cycles, i.e. those with
 * cycle_type > 10, will be unavailable.  Similarly, AMS cycles based on
 * monolithic Pi (cycle_type < 10) require that Pi is not NULL.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetInterpolations(void *solver,
                                     hypre_ParCSRMatrix *Pi,
                                     hypre_ParCSRMatrix *Pix,
                                     hypre_ParCSRMatrix *Piy,
                                     hypre_ParCSRMatrix *Piz)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> Pi = Pi;
   ams_data -> Pix = Pix;
   ams_data -> Piy = Piy;
   ams_data -> Piz = Piz;
   ams_data -> owns_Pi = 0;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetAlphaPoissonMatrix
 *
 * Set the matrix corresponding to the Poisson problem with coefficient
 * alpha (the curl-curl term coefficient in the Maxwell problem).
 *
 * If this function is called, the coarse space solver on the range
 * of Pi^T is a block-diagonal version of A_Pi. If this function is not
 * called, the coarse space solver on the range of Pi^T is constructed
 * as Pi^T A Pi in hypre_AMSSetup().
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetAlphaPoissonMatrix(void *solver,
                                         hypre_ParCSRMatrix *A_Pi)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> A_Pi = A_Pi;

   /* Penalize the eliminated degrees of freedom */
   hypre_ParCSRMatrixSetDiagRows(A_Pi, HYPRE_REAL_MAX);

   /* Make sure that the first entry in each row is the diagonal one. */
   /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A_Pi)); */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetBetaPoissonMatrix
 *
 * Set the matrix corresponding to the Poisson problem with coefficient
 * beta (the mass term coefficient in the Maxwell problem).
 *
 * This function call is optional - if not given, the Poisson matrix will
 * be computed in hypre_AMSSetup(). If the given matrix is NULL, we assume
 * that beta is 0 and use two-level (instead of three-level) methods.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetBetaPoissonMatrix(void *solver,
                                        hypre_ParCSRMatrix *A_G)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> A_G = A_G;
   if (!A_G)
   {
      ams_data -> beta_is_zero = 1;
   }
   else
   {
      /* Penalize the eliminated degrees of freedom */
      hypre_ParCSRMatrixSetDiagRows(A_G, HYPRE_REAL_MAX);

      /* Make sure that the first entry in each row is the diagonal one. */
      /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(A_G)); */
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetInteriorNodes
 *
 * Set the list of nodes which are interior to the zero-conductivity region.
 * A node is interior if interior_nodes[i] == 1.0.
 *
 * Should be called before hypre_AMSSetup()!
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetInteriorNodes(void *solver,
                                    hypre_ParVector *interior_nodes)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> interior_nodes = interior_nodes;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetProjectionFrequency
 *
 * How often to project the r.h.s. onto the compatible sub-space Ker(G0^T),
 * when iterating with the solver.
 *
 * The default value is every 5th iteration.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetProjectionFrequency(void *solver,
                                          HYPRE_Int projection_frequency)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> projection_frequency = projection_frequency;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetMaxIter
 *
 * Set the maximum number of iterations in the three-level method.
 * The default value is 20. To use the AMS solver as a preconditioner,
 * set maxit to 1, tol to 0.0 and print_level to 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetMaxIter(void *solver,
                              HYPRE_Int maxit)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> maxit = maxit;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetTol
 *
 * Set the convergence tolerance (if the method is used as a solver).
 * The default value is 1e-6.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetTol(void *solver,
                          HYPRE_Real tol)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> tol = tol;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetCycleType
 *
 * Choose which three-level solver to use. Possible values are:
 *
 *   1 = 3-level multipl. solver (01210)      <-- small solution time
 *   2 = 3-level additive solver (0+1+2)
 *   3 = 3-level multipl. solver (02120)
 *   4 = 3-level additive solver (010+2)
 *   5 = 3-level multipl. solver (0102010)    <-- small solution time
 *   6 = 3-level additive solver (1+020)
 *   7 = 3-level multipl. solver (0201020)    <-- small number of iterations
 *   8 = 3-level additive solver (0(1+2)0)    <-- small solution time
 *   9 = 3-level multipl. solver (01210) with discrete divergence
 *  11 = 5-level multipl. solver (013454310)  <-- small solution time, memory
 *  12 = 5-level additive solver (0+1+3+4+5)
 *  13 = 5-level multipl. solver (034515430)  <-- small solution time, memory
 *  14 = 5-level additive solver (01(3+4+5)10)
 *  20 = 2-level multipl. solver (0[12]0)
 *
 *   0 = a Hiptmair-like smoother (010)
 *
 * The default value is 1.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetCycleType(void *solver,
                                HYPRE_Int cycle_type)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> cycle_type = cycle_type;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetPrintLevel
 *
 * Control how much information is printed during the solution iterations.
 * The defaut values is 1 (print residual norm at each step).
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetPrintLevel(void *solver,
                                 HYPRE_Int print_level)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> print_level = print_level;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetSmoothingOptions
 *
 * Set relaxation parameters for A. Default values: 2, 1, 1.0, 1.0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetSmoothingOptions(void *solver,
                                       HYPRE_Int A_relax_type,
                                       HYPRE_Int A_relax_times,
                                       HYPRE_Real A_relax_weight,
                                       HYPRE_Real A_omega)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> A_relax_type = A_relax_type;
   ams_data -> A_relax_times = A_relax_times;
   ams_data -> A_relax_weight = A_relax_weight;
   ams_data -> A_omega = A_omega;
   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_AMSSetChebySmoothingOptions
 *  AB: note: this could be added to the above,
 *      but I didn't want to change parameter list)
 * Set parameters for chebyshev smoother for A. Default values: 2,.3.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMSSetChebySmoothingOptions(void       *solver,
                                  HYPRE_Int   A_cheby_order,
                                  HYPRE_Real  A_cheby_fraction)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> A_cheby_order =  A_cheby_order;
   ams_data -> A_cheby_fraction =  A_cheby_fraction;

   return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_AMSSetAlphaAMGOptions
 *
 * Set AMG parameters for B_Pi. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetAlphaAMGOptions(void *solver,
                                      HYPRE_Int B_Pi_coarsen_type,
                                      HYPRE_Int B_Pi_agg_levels,
                                      HYPRE_Int B_Pi_relax_type,
                                      HYPRE_Real B_Pi_theta,
                                      HYPRE_Int B_Pi_interp_type,
                                      HYPRE_Int B_Pi_Pmax)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> B_Pi_coarsen_type = B_Pi_coarsen_type;
   ams_data -> B_Pi_agg_levels = B_Pi_agg_levels;
   ams_data -> B_Pi_relax_type = B_Pi_relax_type;
   ams_data -> B_Pi_theta = B_Pi_theta;
   ams_data -> B_Pi_interp_type = B_Pi_interp_type;
   ams_data -> B_Pi_Pmax = B_Pi_Pmax;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetAlphaAMGCoarseRelaxType
 *
 * Set the AMG coarsest level relaxation for B_Pi. Default value: 8.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetAlphaAMGCoarseRelaxType(void *solver,
                                              HYPRE_Int B_Pi_coarse_relax_type)
{
   hypre_AMSData *ams_data =  (hypre_AMSData *)solver;
   ams_data -> B_Pi_coarse_relax_type = B_Pi_coarse_relax_type;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetBetaAMGOptions
 *
 * Set AMG parameters for B_G. Default values: 10, 1, 3, 0.25, 0, 0.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetBetaAMGOptions(void *solver,
                                     HYPRE_Int B_G_coarsen_type,
                                     HYPRE_Int B_G_agg_levels,
                                     HYPRE_Int B_G_relax_type,
                                     HYPRE_Real B_G_theta,
                                     HYPRE_Int B_G_interp_type,
                                     HYPRE_Int B_G_Pmax)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> B_G_coarsen_type = B_G_coarsen_type;
   ams_data -> B_G_agg_levels = B_G_agg_levels;
   ams_data -> B_G_relax_type = B_G_relax_type;
   ams_data -> B_G_theta = B_G_theta;
   ams_data -> B_G_interp_type = B_G_interp_type;
   ams_data -> B_G_Pmax = B_G_Pmax;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetBetaAMGCoarseRelaxType
 *
 * Set the AMG coarsest level relaxation for B_G. Default value: 8.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSetBetaAMGCoarseRelaxType(void *solver,
                                             HYPRE_Int B_G_coarse_relax_type)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   ams_data -> B_G_coarse_relax_type = B_G_coarse_relax_type;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSComputePi
 *
 * Construct the Pi interpolation matrix, which maps the space of vector
 * linear finite elements to the space of edge finite elements.
 *
 * The construction is based on the fact that Pi = [Pi_x, Pi_y, Pi_z],
 * where each block has the same sparsity structure as G, and the entries
 * can be computed from the vectors Gx, Gy, Gz.
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_AMSComputePi_copy1(hypre_DeviceItem &item,
                                  HYPRE_Int  nnz,
                                  HYPRE_Int  dim,
                                  HYPRE_Int *j_in,
                                  HYPRE_Int *j_out)
{
   const HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < nnz)
   {
      const HYPRE_Int j = dim * i;

      for (HYPRE_Int d = 0; d < dim; d++)
      {
         j_out[j + d] = dim * read_only_load(&j_in[i]) + d;
      }
   }
}

__global__ void
hypreGPUKernel_AMSComputePi_copy2(hypre_DeviceItem &item,
                                  HYPRE_Int   nrows,
                                  HYPRE_Int   dim,
                                  HYPRE_Int  *i_in,
                                  HYPRE_Real *data_in,
                                  HYPRE_Real *Gx_data,
                                  HYPRE_Real *Gy_data,
                                  HYPRE_Real *Gz_data,
                                  HYPRE_Real *data_out)
{
   const HYPRE_Int i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   const HYPRE_Int lane_id = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int j = 0, istart, iend;
   HYPRE_Real t, G[3], *Gdata[3];

   Gdata[0] = Gx_data;
   Gdata[1] = Gy_data;
   Gdata[2] = Gz_data;

   if (lane_id < 2)
   {
      j = read_only_load(i_in + i + lane_id);
   }

   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);

   if (lane_id < dim)
   {
      t = read_only_load(Gdata[lane_id] + i);
   }

   for (HYPRE_Int d = 0; d < dim; d++)
   {
      G[d] = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, t, d);
   }

   for (j = istart + lane_id; j < iend; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Real v = data_in ? hypre_abs(read_only_load(&data_in[j])) * 0.5 : 1.0;
      const HYPRE_Int k = j * dim;

      for (HYPRE_Int d = 0; d < dim; d++)
      {
         data_out[k + d] = v * G[d];
      }
   }
}

#endif

HYPRE_Int
hypre_AMSComputePi(hypre_ParCSRMatrix *A,
                   hypre_ParCSRMatrix *G,
                   hypre_ParVector *Gx,
                   hypre_ParVector *Gy,
                   hypre_ParVector *Gz,
                   HYPRE_Int dim,
                   hypre_ParCSRMatrix **Pi_ptr)
{
   HYPRE_UNUSED_VAR(A);

   hypre_ParCSRMatrix *Pi;

   /* Compute Pi = [Pi_x, Pi_y, Pi_z] */
   {
      HYPRE_Int i, j, d;

      HYPRE_Real *Gx_data, *Gy_data = NULL, *Gz_data = NULL;

      MPI_Comm comm = hypre_ParCSRMatrixComm(G);
      HYPRE_BigInt *col_starts_G = hypre_ParCSRMatrixColStarts(G);
      HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(G);
      HYPRE_BigInt global_num_cols = dim * hypre_ParCSRMatrixGlobalNumCols(G);
      HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(G);
      HYPRE_BigInt col_starts[2] = {(HYPRE_BigInt)dim * col_starts_G[0],
                                    (HYPRE_BigInt)dim * col_starts_G[1]
                                   };
      HYPRE_Int num_cols_offd = dim * hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(G));
      HYPRE_Int num_nonzeros_diag = dim * hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(G));
      HYPRE_Int num_nonzeros_offd = dim * hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(G));

      Pi = hypre_ParCSRMatrixCreate(comm,
                                    global_num_rows,
                                    global_num_cols,
                                    row_starts,
                                    col_starts,
                                    num_cols_offd,
                                    num_nonzeros_diag,
                                    num_nonzeros_offd);

      hypre_ParCSRMatrixOwnsData(Pi) = 1;
      hypre_ParCSRMatrixInitialize(Pi);

      Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(Gx));
      if (dim >= 2)
      {
         Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(Gy));
      }
      if (dim == 3)
      {
         Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(Gz));
      }

#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(G),
                                                         hypre_ParCSRMatrixMemoryLocation(Pi) );
#endif

      /* Fill-in the diagonal part */
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         HYPRE_Int *G_diag_I = hypre_CSRMatrixI(G_diag);
         HYPRE_Int *G_diag_J = hypre_CSRMatrixJ(G_diag);
         HYPRE_Real *G_diag_data = hypre_CSRMatrixData(G_diag);

         HYPRE_Int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         HYPRE_Int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pi_diag = hypre_ParCSRMatrixDiag(Pi);
         HYPRE_Int *Pi_diag_I = hypre_CSRMatrixI(Pi_diag);
         HYPRE_Int *Pi_diag_J = hypre_CSRMatrixJ(Pi_diag);
         HYPRE_Real *Pi_diag_data = hypre_CSRMatrixData(Pi_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntScalen( G_diag_I, G_diag_nrows + 1, Pi_diag_I, dim );

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nnz, "thread", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_diag_nnz, dim, G_diag_J, Pi_diag_J );

            gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, Gz_data,
                              Pi_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pi_diag_I[i] = dim * G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  Pi_diag_J[dim * i + d] = dim * G_diag_J[i] + d;
               }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pi_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 2)
                  {
                     *Pi_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 3)
                  {
                     *Pi_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         HYPRE_Int *G_offd_I = hypre_CSRMatrixI(G_offd);
         HYPRE_Int *G_offd_J = hypre_CSRMatrixJ(G_offd);
         HYPRE_Real *G_offd_data = hypre_CSRMatrixData(G_offd);

         HYPRE_Int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         HYPRE_Int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         HYPRE_Int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pi_offd = hypre_ParCSRMatrixOffd(Pi);
         HYPRE_Int *Pi_offd_I = hypre_CSRMatrixI(Pi_offd);
         HYPRE_Int *Pi_offd_J = hypre_CSRMatrixJ(Pi_offd);
         HYPRE_Real *Pi_offd_data = hypre_CSRMatrixData(Pi_offd);

         HYPRE_BigInt *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         HYPRE_BigInt *Pi_cmap = hypre_ParCSRMatrixColMapOffd(Pi);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            if (G_offd_ncols)
            {
               hypreDevice_IntScalen( G_offd_I, G_offd_nrows + 1, Pi_offd_I, dim );
            }

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nnz, "thread", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_offd_nnz, dim, G_offd_J, Pi_offd_J );

            gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy2, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, Gz_data,
                              Pi_offd_data );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pi_offd_I[i] = dim * G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  Pi_offd_J[dim * i + d] = dim * G_offd_J[i] + d;
               }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pi_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 2)
                  {
                     *Pi_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 3)
                  {
                     *Pi_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
            for (d = 0; d < dim; d++)
            {
               Pi_cmap[dim * i + d] = (HYPRE_BigInt)dim * G_cmap[i] + (HYPRE_BigInt)d;
            }
      }
   }

   *Pi_ptr = Pi;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSComputePixyz
 *
 * Construct the components Pix, Piy, Piz of the interpolation matrix Pi,
 * which maps the space of vector linear finite elements to the space of
 * edge finite elements.
 *
 * The construction is based on the fact that each component has the same
 * sparsity structure as G, and the entries can be computed from the vectors
 * Gx, Gy, Gz.
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_AMSComputePixyz_copy(hypre_DeviceItem &item,
                                    HYPRE_Int   nrows,
                                    HYPRE_Int   dim,
                                    HYPRE_Int  *i_in,
                                    HYPRE_Real *data_in,
                                    HYPRE_Real *Gx_data,
                                    HYPRE_Real *Gy_data,
                                    HYPRE_Real *Gz_data,
                                    HYPRE_Real *data_x_out,
                                    HYPRE_Real *data_y_out,
                                    HYPRE_Real *data_z_out )
{
   const HYPRE_Int i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   const HYPRE_Int lane_id = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int j = 0, istart, iend;
   HYPRE_Real t, G[3], *Gdata[3], *Odata[3];

   Gdata[0] = Gx_data;
   Gdata[1] = Gy_data;
   Gdata[2] = Gz_data;

   Odata[0] = data_x_out;
   Odata[1] = data_y_out;
   Odata[2] = data_z_out;

   if (lane_id < 2)
   {
      j = read_only_load(i_in + i + lane_id);
   }

   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);

   if (lane_id < dim)
   {
      t = read_only_load(Gdata[lane_id] + i);
   }

   for (HYPRE_Int d = 0; d < dim; d++)
   {
      G[d] = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, t, d);
   }

   for (j = istart + lane_id; j < iend; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Real v = data_in ? hypre_abs(read_only_load(&data_in[j])) * 0.5 : 1.0;

      for (HYPRE_Int d = 0; d < dim; d++)
      {
         Odata[d][j] = v * G[d];
      }
   }
}
#endif

HYPRE_Int
hypre_AMSComputePixyz(hypre_ParCSRMatrix *A,
                      hypre_ParCSRMatrix *G,
                      hypre_ParVector *Gx,
                      hypre_ParVector *Gy,
                      hypre_ParVector *Gz,
                      HYPRE_Int dim,
                      hypre_ParCSRMatrix **Pix_ptr,
                      hypre_ParCSRMatrix **Piy_ptr,
                      hypre_ParCSRMatrix **Piz_ptr)
{
   HYPRE_UNUSED_VAR(A);

   hypre_ParCSRMatrix *Pix, *Piy, *Piz = NULL;

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(G) );
#endif

   /* Compute Pix, Piy, Piz  */
   {
      HYPRE_Int i, j;

      HYPRE_Real *Gx_data, *Gy_data, *Gz_data;

      MPI_Comm comm = hypre_ParCSRMatrixComm(G);
      HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(G);
      HYPRE_BigInt global_num_cols = hypre_ParCSRMatrixGlobalNumCols(G);
      HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(G);
      HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(G);
      HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(G));
      HYPRE_Int num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(G));
      HYPRE_Int num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(G));

      Pix = hypre_ParCSRMatrixCreate(comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     num_nonzeros_diag,
                                     num_nonzeros_offd);
      hypre_ParCSRMatrixOwnsData(Pix) = 1;
      hypre_ParCSRMatrixInitialize(Pix);

      if (dim >= 2)
      {
         Piy = hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         hypre_ParCSRMatrixOwnsData(Piy) = 1;
         hypre_ParCSRMatrixInitialize(Piy);
      }

      if (dim == 3)
      {
         Piz = hypre_ParCSRMatrixCreate(comm,
                                        global_num_rows,
                                        global_num_cols,
                                        row_starts,
                                        col_starts,
                                        num_cols_offd,
                                        num_nonzeros_diag,
                                        num_nonzeros_offd);
         hypre_ParCSRMatrixOwnsData(Piz) = 1;
         hypre_ParCSRMatrixInitialize(Piz);
      }

      Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(Gx));
      if (dim >= 2)
      {
         Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(Gy));
      }
      if (dim == 3)
      {
         Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(Gz));
      }

      /* Fill-in the diagonal part */
      if (dim == 3)
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         HYPRE_Int *G_diag_I = hypre_CSRMatrixI(G_diag);
         HYPRE_Int *G_diag_J = hypre_CSRMatrixJ(G_diag);
         HYPRE_Real *G_diag_data = hypre_CSRMatrixData(G_diag);

         HYPRE_Int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         HYPRE_Int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pix_diag = hypre_ParCSRMatrixDiag(Pix);
         HYPRE_Int *Pix_diag_I = hypre_CSRMatrixI(Pix_diag);
         HYPRE_Int *Pix_diag_J = hypre_CSRMatrixJ(Pix_diag);
         HYPRE_Real *Pix_diag_data = hypre_CSRMatrixData(Pix_diag);

         hypre_CSRMatrix *Piy_diag = hypre_ParCSRMatrixDiag(Piy);
         HYPRE_Int *Piy_diag_I = hypre_CSRMatrixI(Piy_diag);
         HYPRE_Int *Piy_diag_J = hypre_CSRMatrixJ(Piy_diag);
         HYPRE_Real *Piy_diag_data = hypre_CSRMatrixData(Piy_diag);

         hypre_CSRMatrix *Piz_diag = hypre_ParCSRMatrixDiag(Piz);
         HYPRE_Int *Piz_diag_I = hypre_CSRMatrixI(Piz_diag);
         HYPRE_Int *Piz_diag_J = hypre_CSRMatrixJ(Piz_diag);
         HYPRE_Real *Piz_diag_data = hypre_CSRMatrixData(Piz_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_I, G_diag_I, G_diag_I),
                               G_diag_nrows + 1,
                               oneapi::dpl::make_zip_iterator(Pix_diag_I, Piy_diag_I, Piz_diag_I) );

            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_J, G_diag_J, G_diag_J),
                               G_diag_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_diag_J, Piy_diag_J, Piz_diag_J) );
#else
            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_I, G_diag_I, G_diag_I)),
                               G_diag_nrows + 1,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_I, Piy_diag_I, Piz_diag_I)) );

            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_J, G_diag_J, G_diag_J)),
                               G_diag_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_J, Piy_diag_J, Piz_diag_J)) );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, Gz_data,
                              Pix_diag_data, Piy_diag_data, Piz_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = G_diag_I[i];
               Piy_diag_I[i] = G_diag_I[i];
               Piz_diag_I[i] = G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
            {
               Pix_diag_J[i] = G_diag_J[i];
               Piy_diag_J[i] = G_diag_J[i];
               Piz_diag_J[i] = G_diag_J[i];
            }

            for (i = 0; i < G_diag_nrows; i++)
            {
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  *Piy_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
                  *Piz_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gz_data[i];
               }
            }
         }
      }
      else if (dim == 2)
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         HYPRE_Int *G_diag_I = hypre_CSRMatrixI(G_diag);
         HYPRE_Int *G_diag_J = hypre_CSRMatrixJ(G_diag);
         HYPRE_Real *G_diag_data = hypre_CSRMatrixData(G_diag);

         HYPRE_Int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         HYPRE_Int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pix_diag = hypre_ParCSRMatrixDiag(Pix);
         HYPRE_Int *Pix_diag_I = hypre_CSRMatrixI(Pix_diag);
         HYPRE_Int *Pix_diag_J = hypre_CSRMatrixJ(Pix_diag);
         HYPRE_Real *Pix_diag_data = hypre_CSRMatrixData(Pix_diag);

         hypre_CSRMatrix *Piy_diag = hypre_ParCSRMatrixDiag(Piy);
         HYPRE_Int *Piy_diag_I = hypre_CSRMatrixI(Piy_diag);
         HYPRE_Int *Piy_diag_J = hypre_CSRMatrixJ(Piy_diag);
         HYPRE_Real *Piy_diag_data = hypre_CSRMatrixData(Piy_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_I, G_diag_I),
                               G_diag_nrows + 1,
                               oneapi::dpl::make_zip_iterator(Pix_diag_I, Piy_diag_I) );

            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_diag_J, G_diag_J),
                               G_diag_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_diag_J, Piy_diag_J) );
#else
            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_I, G_diag_I)),
                               G_diag_nrows + 1,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_I, Piy_diag_I)) );

            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_diag_J, G_diag_J)),
                               G_diag_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_diag_J, Piy_diag_J)) );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, NULL,
                              Pix_diag_data, Piy_diag_data, NULL );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = G_diag_I[i];
               Piy_diag_I[i] = G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
            {
               Pix_diag_J[i] = G_diag_J[i];
               Piy_diag_J[i] = G_diag_J[i];
            }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  *Piy_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
               }
         }
      }
      else
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         HYPRE_Int *G_diag_I = hypre_CSRMatrixI(G_diag);
         HYPRE_Int *G_diag_J = hypre_CSRMatrixJ(G_diag);
         HYPRE_Real *G_diag_data = hypre_CSRMatrixData(G_diag);

         HYPRE_Int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         HYPRE_Int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *Pix_diag = hypre_ParCSRMatrixDiag(Pix);
         HYPRE_Int *Pix_diag_I = hypre_CSRMatrixI(Pix_diag);
         HYPRE_Int *Pix_diag_J = hypre_CSRMatrixJ(Pix_diag);
         HYPRE_Real *Pix_diag_data = hypre_CSRMatrixData(Pix_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            HYPRE_ONEDPL_CALL( std::copy_n,
                               G_diag_I,
                               G_diag_nrows + 1,
                               Pix_diag_I );

            HYPRE_ONEDPL_CALL( std::copy_n,
                               G_diag_J,
                               G_diag_nnz,
                               Pix_diag_J );
#else
            HYPRE_THRUST_CALL( copy_n,
                               G_diag_I,
                               G_diag_nrows + 1,
                               Pix_diag_I );

            HYPRE_THRUST_CALL( copy_n,
                               G_diag_J,
                               G_diag_nnz,
                               Pix_diag_J );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, NULL, NULL,
                              Pix_diag_data, NULL, NULL );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               Pix_diag_I[i] = G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
            {
               Pix_diag_J[i] = G_diag_J[i];
            }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *Pix_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
               }
         }
      }


      /* Fill-in the off-diagonal part */
      if (dim == 3)
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         HYPRE_Int *G_offd_I = hypre_CSRMatrixI(G_offd);
         HYPRE_Int *G_offd_J = hypre_CSRMatrixJ(G_offd);
         HYPRE_Real *G_offd_data = hypre_CSRMatrixData(G_offd);

         HYPRE_Int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         HYPRE_Int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         HYPRE_Int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pix_offd = hypre_ParCSRMatrixOffd(Pix);
         HYPRE_Int *Pix_offd_I = hypre_CSRMatrixI(Pix_offd);
         HYPRE_Int *Pix_offd_J = hypre_CSRMatrixJ(Pix_offd);
         HYPRE_Real *Pix_offd_data = hypre_CSRMatrixData(Pix_offd);

         hypre_CSRMatrix *Piy_offd = hypre_ParCSRMatrixOffd(Piy);
         HYPRE_Int *Piy_offd_I = hypre_CSRMatrixI(Piy_offd);
         HYPRE_Int *Piy_offd_J = hypre_CSRMatrixJ(Piy_offd);
         HYPRE_Real *Piy_offd_data = hypre_CSRMatrixData(Piy_offd);

         hypre_CSRMatrix *Piz_offd = hypre_ParCSRMatrixOffd(Piz);
         HYPRE_Int *Piz_offd_I = hypre_CSRMatrixI(Piz_offd);
         HYPRE_Int *Piz_offd_J = hypre_CSRMatrixJ(Piz_offd);
         HYPRE_Real *Piz_offd_data = hypre_CSRMatrixData(Piz_offd);

         HYPRE_BigInt *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         HYPRE_BigInt *Pix_cmap = hypre_ParCSRMatrixColMapOffd(Pix);
         HYPRE_BigInt *Piy_cmap = hypre_ParCSRMatrixColMapOffd(Piy);
         HYPRE_BigInt *Piz_cmap = hypre_ParCSRMatrixColMapOffd(Piz);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            if (G_offd_ncols)
            {
               HYPRE_ONEDPL_CALL( std::copy_n,
                                  oneapi::dpl::make_zip_iterator(G_offd_I, G_offd_I, G_offd_I),
                                  G_offd_nrows + 1,
                                  oneapi::dpl::make_zip_iterator(Pix_offd_I, Piy_offd_I, Piz_offd_I) );
            }

            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_offd_J, G_offd_J, G_offd_J),
                               G_offd_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_offd_J, Piy_offd_J, Piz_offd_J) );
#else
            if (G_offd_ncols)
            {
               HYPRE_THRUST_CALL( copy_n,
                                  thrust::make_zip_iterator(thrust::make_tuple(G_offd_I, G_offd_I, G_offd_I)),
                                  G_offd_nrows + 1,
                                  thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_I, Piy_offd_I, Piz_offd_I)) );
            }

            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_offd_J, G_offd_J, G_offd_J)),
                               G_offd_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_J, Piy_offd_J, Piz_offd_J)) );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, Gz_data,
                              Pix_offd_data, Piy_offd_data, Piz_offd_data );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = G_offd_I[i];
                  Piy_offd_I[i] = G_offd_I[i];
                  Piz_offd_I[i] = G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
            {
               Pix_offd_J[i] = G_offd_J[i];
               Piy_offd_J[i] = G_offd_J[i];
               Piz_offd_J[i] = G_offd_J[i];
            }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  *Piy_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
                  *Piz_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gz_data[i];
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
            Piy_cmap[i] = G_cmap[i];
            Piz_cmap[i] = G_cmap[i];
         }
      }
      else if (dim == 2)
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         HYPRE_Int *G_offd_I = hypre_CSRMatrixI(G_offd);
         HYPRE_Int *G_offd_J = hypre_CSRMatrixJ(G_offd);
         HYPRE_Real *G_offd_data = hypre_CSRMatrixData(G_offd);

         HYPRE_Int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         HYPRE_Int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         HYPRE_Int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pix_offd = hypre_ParCSRMatrixOffd(Pix);
         HYPRE_Int *Pix_offd_I = hypre_CSRMatrixI(Pix_offd);
         HYPRE_Int *Pix_offd_J = hypre_CSRMatrixJ(Pix_offd);
         HYPRE_Real *Pix_offd_data = hypre_CSRMatrixData(Pix_offd);

         hypre_CSRMatrix *Piy_offd = hypre_ParCSRMatrixOffd(Piy);
         HYPRE_Int *Piy_offd_I = hypre_CSRMatrixI(Piy_offd);
         HYPRE_Int *Piy_offd_J = hypre_CSRMatrixJ(Piy_offd);
         HYPRE_Real *Piy_offd_data = hypre_CSRMatrixData(Piy_offd);

         HYPRE_BigInt *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         HYPRE_BigInt *Pix_cmap = hypre_ParCSRMatrixColMapOffd(Pix);
         HYPRE_BigInt *Piy_cmap = hypre_ParCSRMatrixColMapOffd(Piy);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            if (G_offd_ncols)
            {
               HYPRE_ONEDPL_CALL( std::copy_n,
                                  oneapi::dpl::make_zip_iterator(G_offd_I, G_offd_I),
                                  G_offd_nrows + 1,
                                  oneapi::dpl::make_zip_iterator(Pix_offd_I, Piy_offd_I) );
            }

            HYPRE_ONEDPL_CALL( std::copy_n,
                               oneapi::dpl::make_zip_iterator(G_offd_J, G_offd_J),
                               G_offd_nnz,
                               oneapi::dpl::make_zip_iterator(Pix_offd_J, Piy_offd_J) );
#else
            if (G_offd_ncols)
            {
               HYPRE_THRUST_CALL( copy_n,
                                  thrust::make_zip_iterator(thrust::make_tuple(G_offd_I, G_offd_I)),
                                  G_offd_nrows + 1,
                                  thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_I, Piy_offd_I)) );
            }

            HYPRE_THRUST_CALL( copy_n,
                               thrust::make_zip_iterator(thrust::make_tuple(G_offd_J, G_offd_J)),
                               G_offd_nnz,
                               thrust::make_zip_iterator(thrust::make_tuple(Pix_offd_J, Piy_offd_J)) );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, NULL,
                              Pix_offd_data, Piy_offd_data, NULL );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = G_offd_I[i];
                  Piy_offd_I[i] = G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
            {
               Pix_offd_J[i] = G_offd_J[i];
               Piy_offd_J[i] = G_offd_J[i];
            }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  *Piy_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
            Piy_cmap[i] = G_cmap[i];
         }
      }
      else
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         HYPRE_Int *G_offd_I = hypre_CSRMatrixI(G_offd);
         HYPRE_Int *G_offd_J = hypre_CSRMatrixJ(G_offd);
         HYPRE_Real *G_offd_data = hypre_CSRMatrixData(G_offd);

         HYPRE_Int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         HYPRE_Int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         HYPRE_Int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *Pix_offd = hypre_ParCSRMatrixOffd(Pix);
         HYPRE_Int *Pix_offd_I = hypre_CSRMatrixI(Pix_offd);
         HYPRE_Int *Pix_offd_J = hypre_CSRMatrixJ(Pix_offd);
         HYPRE_Real *Pix_offd_data = hypre_CSRMatrixData(Pix_offd);

         HYPRE_BigInt *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         HYPRE_BigInt *Pix_cmap = hypre_ParCSRMatrixColMapOffd(Pix);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
#if defined(HYPRE_USING_SYCL)
            if (G_offd_ncols)
            {
               HYPRE_ONEDPL_CALL( std::copy_n,
                                  G_offd_I,
                                  G_offd_nrows + 1,
                                  Pix_offd_I );
            }

            HYPRE_ONEDPL_CALL( std::copy_n,
                               G_offd_J,
                               G_offd_nnz,
                               Pix_offd_J );
#else
            if (G_offd_ncols)
            {
               HYPRE_THRUST_CALL( copy_n,
                                  G_offd_I,
                                  G_offd_nrows + 1,
                                  Pix_offd_I );
            }

            HYPRE_THRUST_CALL( copy_n,
                               G_offd_J,
                               G_offd_nnz,
                               Pix_offd_J );
#endif

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePixyz_copy, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, NULL, NULL,
                              Pix_offd_data, NULL, NULL );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  Pix_offd_I[i] = G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
            {
               Pix_offd_J[i] = G_offd_J[i];
            }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *Pix_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
         {
            Pix_cmap[i] = G_cmap[i];
         }
      }
   }

   *Pix_ptr = Pix;
   if (dim >= 2)
   {
      *Piy_ptr = Piy;
   }
   if (dim == 3)
   {
      *Piz_ptr = Piz;
   }

   return hypre_error_flag;
}

#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_AMSComputeGPi_copy2(hypre_DeviceItem &item,
                                   HYPRE_Int   nrows,
                                   HYPRE_Int   dim,
                                   HYPRE_Int  *i_in,
                                   HYPRE_Real *data_in,
                                   HYPRE_Real *Gx_data,
                                   HYPRE_Real *Gy_data,
                                   HYPRE_Real *Gz_data,
                                   HYPRE_Real *data_out)
{
   const HYPRE_Int i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (i >= nrows)
   {
      return;
   }

   const HYPRE_Int lane_id = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int j = 0, istart, iend;
   HYPRE_Real t, G[3], *Gdata[3];

   Gdata[0] = Gx_data;
   Gdata[1] = Gy_data;
   Gdata[2] = Gz_data;

   if (lane_id < 2)
   {
      j = read_only_load(i_in + i + lane_id);
   }

   istart = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 0);
   iend   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, j, 1);

   if (lane_id < dim - 1)
   {
      t = read_only_load(Gdata[lane_id] + i);
   }

   for (HYPRE_Int d = 0; d < dim - 1; d++)
   {
      G[d] = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, t, d);
   }

   for (j = istart + lane_id; j < iend; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Real u = read_only_load(&data_in[j]);
      const HYPRE_Real v = hypre_abs(u) * 0.5;
      const HYPRE_Int k = j * dim;

      data_out[k] = u;
      for (HYPRE_Int d = 0; d < dim - 1; d++)
      {
         data_out[k + d + 1] = v * G[d];
      }
   }
}
#endif

/*--------------------------------------------------------------------------
 * hypre_AMSComputeGPi
 *
 * Construct the matrix [G,Pi] which can be considered an interpolation
 * matrix from S_h^4 (4 copies of the scalar linear finite element space)
 * to the edge finite elements space.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMSComputeGPi(hypre_ParCSRMatrix *A,
                    hypre_ParCSRMatrix *G,
                    hypre_ParVector *Gx,
                    hypre_ParVector *Gy,
                    hypre_ParVector *Gz,
                    HYPRE_Int dim,
                    hypre_ParCSRMatrix **GPi_ptr)
{
   HYPRE_UNUSED_VAR(A);

   hypre_ParCSRMatrix *GPi;

   /* Take into account G */
   dim++;

   /* Compute GPi = [Pi_x, Pi_y, Pi_z, G] */
   {
      HYPRE_Int i, j, d;

      HYPRE_Real *Gx_data, *Gy_data = NULL, *Gz_data = NULL;

      MPI_Comm comm = hypre_ParCSRMatrixComm(G);
      HYPRE_BigInt *col_starts_G = hypre_ParCSRMatrixColStarts(G);
      HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(G);
      HYPRE_BigInt global_num_cols = dim * hypre_ParCSRMatrixGlobalNumCols(G);
      HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(G);
      HYPRE_BigInt col_starts[2] = {(HYPRE_BigInt)dim * col_starts_G[0],
                                    (HYPRE_BigInt)dim * col_starts_G[1]
                                   };
      HYPRE_Int num_cols_offd = dim * hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(G));
      HYPRE_Int num_nonzeros_diag = dim * hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(G));
      HYPRE_Int num_nonzeros_offd = dim * hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(G));

      GPi = hypre_ParCSRMatrixCreate(comm,
                                     global_num_rows,
                                     global_num_cols,
                                     row_starts,
                                     col_starts,
                                     num_cols_offd,
                                     num_nonzeros_diag,
                                     num_nonzeros_offd);

      hypre_ParCSRMatrixOwnsData(GPi) = 1;
      hypre_ParCSRMatrixInitialize(GPi);

      Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(Gx));
      if (dim >= 3)
      {
         Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(Gy));
      }
      if (dim == 4)
      {
         Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(Gz));
      }

#if defined(HYPRE_USING_GPU)
      HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy2( hypre_ParCSRMatrixMemoryLocation(G),
                                                         hypre_ParCSRMatrixMemoryLocation(GPi) );
#endif

      /* Fill-in the diagonal part */
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         HYPRE_Int *G_diag_I = hypre_CSRMatrixI(G_diag);
         HYPRE_Int *G_diag_J = hypre_CSRMatrixJ(G_diag);
         HYPRE_Real *G_diag_data = hypre_CSRMatrixData(G_diag);

         HYPRE_Int G_diag_nrows = hypre_CSRMatrixNumRows(G_diag);
         HYPRE_Int G_diag_nnz = hypre_CSRMatrixNumNonzeros(G_diag);

         hypre_CSRMatrix *GPi_diag = hypre_ParCSRMatrixDiag(GPi);
         HYPRE_Int *GPi_diag_I = hypre_CSRMatrixI(GPi_diag);
         HYPRE_Int *GPi_diag_J = hypre_CSRMatrixJ(GPi_diag);
         HYPRE_Real *GPi_diag_data = hypre_CSRMatrixData(GPi_diag);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            hypreDevice_IntScalen( G_diag_I, G_diag_nrows + 1, GPi_diag_I, dim );

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nnz, "thread", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_diag_nnz, dim, G_diag_J, GPi_diag_J );

            gDim = hypre_GetDefaultDeviceGridDimension(G_diag_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputeGPi_copy2, gDim, bDim,
                              G_diag_nrows, dim, G_diag_I, G_diag_data, Gx_data, Gy_data, Gz_data,
                              GPi_diag_data );
         }
         else
#endif
         {
            for (i = 0; i < G_diag_nrows + 1; i++)
            {
               GPi_diag_I[i] = dim * G_diag_I[i];
            }

            for (i = 0; i < G_diag_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  GPi_diag_J[dim * i + d] = dim * G_diag_J[i] + d;
               }

            for (i = 0; i < G_diag_nrows; i++)
               for (j = G_diag_I[i]; j < G_diag_I[i + 1]; j++)
               {
                  *GPi_diag_data++ = G_diag_data[j];
                  *GPi_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 3)
                  {
                     *GPi_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 4)
                  {
                     *GPi_diag_data++ = hypre_abs(G_diag_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }
      }

      /* Fill-in the off-diagonal part */
      {
         hypre_CSRMatrix *G_offd = hypre_ParCSRMatrixOffd(G);
         HYPRE_Int *G_offd_I = hypre_CSRMatrixI(G_offd);
         HYPRE_Int *G_offd_J = hypre_CSRMatrixJ(G_offd);
         HYPRE_Real *G_offd_data = hypre_CSRMatrixData(G_offd);

         HYPRE_Int G_offd_nrows = hypre_CSRMatrixNumRows(G_offd);
         HYPRE_Int G_offd_ncols = hypre_CSRMatrixNumCols(G_offd);
         HYPRE_Int G_offd_nnz = hypre_CSRMatrixNumNonzeros(G_offd);

         hypre_CSRMatrix *GPi_offd = hypre_ParCSRMatrixOffd(GPi);
         HYPRE_Int *GPi_offd_I = hypre_CSRMatrixI(GPi_offd);
         HYPRE_Int *GPi_offd_J = hypre_CSRMatrixJ(GPi_offd);
         HYPRE_Real *GPi_offd_data = hypre_CSRMatrixData(GPi_offd);

         HYPRE_BigInt *G_cmap = hypre_ParCSRMatrixColMapOffd(G);
         HYPRE_BigInt *GPi_cmap = hypre_ParCSRMatrixColMapOffd(GPi);

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            if (G_offd_ncols)
            {
               hypreDevice_IntScalen( G_offd_I, G_offd_nrows + 1, GPi_offd_I, dim );
            }

            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nnz, "thread", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputePi_copy1, gDim, bDim,
                              G_offd_nnz, dim, G_offd_J, GPi_offd_J );

            gDim = hypre_GetDefaultDeviceGridDimension(G_offd_nrows, "warp", bDim);

            HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSComputeGPi_copy2, gDim, bDim,
                              G_offd_nrows, dim, G_offd_I, G_offd_data, Gx_data, Gy_data, Gz_data,
                              GPi_offd_data );
         }
         else
#endif
         {
            if (G_offd_ncols)
               for (i = 0; i < G_offd_nrows + 1; i++)
               {
                  GPi_offd_I[i] = dim * G_offd_I[i];
               }

            for (i = 0; i < G_offd_nnz; i++)
               for (d = 0; d < dim; d++)
               {
                  GPi_offd_J[dim * i + d] = dim * G_offd_J[i] + d;
               }

            for (i = 0; i < G_offd_nrows; i++)
               for (j = G_offd_I[i]; j < G_offd_I[i + 1]; j++)
               {
                  *GPi_offd_data++ = G_offd_data[j];
                  *GPi_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gx_data[i];
                  if (dim >= 3)
                  {
                     *GPi_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gy_data[i];
                  }
                  if (dim == 4)
                  {
                     *GPi_offd_data++ = hypre_abs(G_offd_data[j]) * 0.5 * Gz_data[i];
                  }
               }
         }

         for (i = 0; i < G_offd_ncols; i++)
            for (d = 0; d < dim; d++)
            {
               GPi_cmap[dim * i + d] = dim * G_cmap[i] + d;
            }
      }

   }

   *GPi_ptr = GPi;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSetup
 *
 * Construct the AMS solver components.
 *
 * The following functions need to be called before hypre_AMSSetup():
 * - hypre_AMSSetDimension() (if solving a 2D problem)
 * - hypre_AMSSetDiscreteGradient()
 * - hypre_AMSSetCoordinateVectors() or hypre_AMSSetEdgeConstantVectors
 *--------------------------------------------------------------------------*/
#if defined(HYPRE_USING_GPU)
__global__ void
hypreGPUKernel_FixInterNodes( hypre_DeviceItem    &item,
                              HYPRE_Int      nrows,
                              HYPRE_Int     *G0t_diag_i,
                              HYPRE_Complex *G0t_diag_data,
                              HYPRE_Int     *G0t_offd_i,
                              HYPRE_Complex *G0t_offd_data,
                              HYPRE_Real    *interior_nodes_data)
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int not1 = 0;

   if (lane == 0)
   {
      not1 = read_only_load(&interior_nodes_data[row_i]) != 1.0;
   }

   not1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, not1, 0);

   if (!not1)
   {
      return;
   }

   HYPRE_Int p1 = 0, q1, p2 = 0, q2 = 0;
   bool nonempty_offd = G0t_offd_data != NULL;

   if (lane < 2)
   {
      p1 = read_only_load(G0t_diag_i + row_i + lane);
      if (nonempty_offd)
      {
         p2 = read_only_load(G0t_offd_i + row_i + lane);
      }
   }

   q1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 0);
   if (nonempty_offd)
   {
      q2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (HYPRE_Int j = p1 + lane; j < q1; j += HYPRE_WARP_SIZE)
   {
      G0t_diag_data[j] = 0.0;
   }
   for (HYPRE_Int j = p2 + lane; j < q2; j += HYPRE_WARP_SIZE)
   {
      G0t_offd_data[j] = 0.0;
   }
}

__global__ void
hypreGPUKernel_AMSSetupScaleGGt( hypre_DeviceItem &item,
                                 HYPRE_Int   Gt_num_rows,
                                 HYPRE_Int  *Gt_diag_i,
                                 HYPRE_Int  *Gt_diag_j,
                                 HYPRE_Real *Gt_diag_data,
                                 HYPRE_Int  *Gt_offd_i,
                                 HYPRE_Real *Gt_offd_data,
                                 HYPRE_Real *Gx_data,
                                 HYPRE_Real *Gy_data,
                                 HYPRE_Real *Gz_data )
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= Gt_num_rows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Real h2 = 0.0;
   HYPRE_Int ne, p1 = 0, q1, p2 = 0, q2 = 0;

   if (lane < 2)
   {
      p1 = read_only_load(Gt_diag_i + row_i + lane);
   }
   q1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 1);
   p1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p1, 0);
   ne = q1 - p1;

   if (ne == 0)
   {
      return;
   }

   if (Gt_offd_data != NULL)
   {
      if (lane < 2)
      {
         p2 = read_only_load(Gt_offd_i + row_i + lane);
      }
      q2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 1);
      p2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p2, 0);
   }

   for (HYPRE_Int j = p1 + lane; j < q1; j += HYPRE_WARP_SIZE)
   {
      const HYPRE_Int k = read_only_load(&Gt_diag_j[j]);
      const HYPRE_Real Gx = read_only_load(&Gx_data[k]);
      const HYPRE_Real Gy = read_only_load(&Gy_data[k]);
      const HYPRE_Real Gz = read_only_load(&Gz_data[k]);

      h2 += Gx * Gx + Gy * Gy + Gz * Gz;
   }

   h2 = warp_allreduce_sum(item, h2) / ne;

   for (HYPRE_Int j = p1 + lane; j < q1; j += HYPRE_WARP_SIZE)
   {
      Gt_diag_data[j] *= h2;
   }

   for (HYPRE_Int j = p2 + lane; j < q2; j += HYPRE_WARP_SIZE)
   {
      Gt_offd_data[j] *= h2;
   }
}
#endif

HYPRE_Int
hypre_AMSSetup(void *solver,
               hypre_ParCSRMatrix *A,
               hypre_ParVector *b,
               hypre_ParVector *x)
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );
#endif

   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   HYPRE_Int input_info = 0;

   ams_data -> A = A;

   /* Modifications for problems with zero-conductivity regions */
   if (ams_data -> interior_nodes)
   {
      hypre_ParCSRMatrix *G0t, *Aorig = A;

      /* Make sure that multiple Setup()+Solve() give identical results */
      ams_data -> solve_counter = 0;

      /* Construct the discrete gradient matrix for the zero-conductivity region
         by eliminating the zero-conductivity nodes from G^t. The range of G0
         represents the kernel of A, i.e. the gradients of nodal basis functions
         supported in zero-conductivity regions. */
      hypre_ParCSRMatrixTranspose(ams_data -> G, &G0t, 1);

      {
         HYPRE_Int i, j;
         HYPRE_Int nv = hypre_ParCSRMatrixNumCols(ams_data -> G);
         hypre_CSRMatrix *G0td = hypre_ParCSRMatrixDiag(G0t);
         HYPRE_Int *G0tdI = hypre_CSRMatrixI(G0td);
         HYPRE_Real *G0tdA = hypre_CSRMatrixData(G0td);
         hypre_CSRMatrix *G0to = hypre_ParCSRMatrixOffd(G0t);
         HYPRE_Int *G0toI = hypre_CSRMatrixI(G0to);
         HYPRE_Real *G0toA = hypre_CSRMatrixData(G0to);
         HYPRE_Real *interior_nodes_data = hypre_VectorData(
                                              hypre_ParVectorLocalVector((hypre_ParVector*) ams_data -> interior_nodes));

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
            dim3 gDim = hypre_GetDefaultDeviceGridDimension(nv, "warp", bDim);
            HYPRE_GPU_LAUNCH( hypreGPUKernel_FixInterNodes, gDim, bDim,
                              nv, G0tdI, G0tdA, G0toI, G0toA, interior_nodes_data );
         }
         else
#endif
         {
            for (i = 0; i < nv; i++)
            {
               if (interior_nodes_data[i] != 1)
               {
                  for (j = G0tdI[i]; j < G0tdI[i + 1]; j++)
                  {
                     G0tdA[j] = 0.0;
                  }
                  if (G0toI)
                     for (j = G0toI[i]; j < G0toI[i + 1]; j++)
                     {
                        G0toA[j] = 0.0;
                     }
               }
            }
         }
      }
      hypre_ParCSRMatrixTranspose(G0t, & ams_data -> G0, 1);

      /* Construct the subspace matrix A_G0 = G0^T G0 */
#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         ams_data -> A_G0 = hypre_ParCSRMatMat(G0t, ams_data -> G0);
      }
      else
#endif
      {
         ams_data -> A_G0 = hypre_ParMatmul(G0t, ams_data -> G0);
      }
      hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G0);

      /* Create AMG solver for A_G0 */
      HYPRE_BoomerAMGCreate(&ams_data -> B_G0);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_G0, ams_data -> B_G_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_G0, ams_data -> B_G_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_G0, ams_data -> B_G_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_G0, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G0, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_G0, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_G0, 3); /* use just a few V-cycles */
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_G0, ams_data -> B_G_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_G0, ams_data -> B_G_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_G0, ams_data -> B_G_Pmax);
      HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_G0, 2); /* don't coarsen to 0 */
      /* Generally, don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_G0, ams_data -> B_G_coarse_relax_type, 3);
      HYPRE_BoomerAMGSetup(ams_data -> B_G0,
                           (HYPRE_ParCSRMatrix)ams_data -> A_G0,
                           0, 0);

      /* Construct the preconditioner for ams_data->A = A + G0 G0^T.
         NOTE: this can be optimized significantly by taking into account that
         the sparsity pattern of A is subset of the sparsity pattern of G0 G0^T */
      {
#if defined(HYPRE_USING_GPU)
         hypre_ParCSRMatrix *A;
         if (exec == HYPRE_EXEC_DEVICE)
         {
            A = hypre_ParCSRMatMat(ams_data -> G0, G0t);
         }
         else
#endif
         {
            A = hypre_ParMatmul(ams_data -> G0, G0t);
         }
         hypre_ParCSRMatrix *B = Aorig;
         hypre_ParCSRMatrix **C_ptr = &ams_data -> A;
         hypre_ParCSRMatrix *C;
         HYPRE_Real factor, lfactor;
         /* scale (penalize) G0 G0^T before adding it to the matrix */
         {
            HYPRE_Int i;
            HYPRE_Int B_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(B));
            HYPRE_Real *B_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(B));
            HYPRE_Real *B_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(B));
            HYPRE_Int *B_diag_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(B));
            HYPRE_Int *B_offd_i = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(B));
            lfactor = -1;
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               HYPRE_Int nnz_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
               HYPRE_Int nnz_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));
#if defined(HYPRE_DEBUG)
               HYPRE_Int nnz;
               hypre_TMemcpy(&nnz, &B_diag_i[B_num_rows], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
               hypre_assert(nnz == nnz_diag);
               hypre_TMemcpy(&nnz, &B_offd_i[B_num_rows], HYPRE_Int, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
               hypre_assert(nnz == nnz_offd);
#endif
#if defined(HYPRE_USING_SYCL)
               if (nnz_diag)
               {
                  lfactor = HYPRE_ONEDPL_CALL( std::reduce,
                                               oneapi::dpl::make_transform_iterator(B_diag_data,            absolute_value<HYPRE_Real>()),
                                               oneapi::dpl::make_transform_iterator(B_diag_data + nnz_diag, absolute_value<HYPRE_Real>()),
                                               -1.0,
                                               sycl::maximum<HYPRE_Real>() );
               }

               if (nnz_offd)
               {
                  lfactor = HYPRE_ONEDPL_CALL( std::reduce,
                                               oneapi::dpl::make_transform_iterator(B_offd_data,            absolute_value<HYPRE_Real>()),
                                               oneapi::dpl::make_transform_iterator(B_offd_data + nnz_offd, absolute_value<HYPRE_Real>()),
                                               lfactor,
                                               sycl::maximum<HYPRE_Real>() );

               }
#else
               if (nnz_diag)
               {
                  lfactor = HYPRE_THRUST_CALL( reduce,
                                               thrust::make_transform_iterator(B_diag_data,            absolute_value<HYPRE_Real>()),
                                               thrust::make_transform_iterator(B_diag_data + nnz_diag, absolute_value<HYPRE_Real>()),
                                               -1.0,
                                               thrust::maximum<HYPRE_Real>() );
               }

               if (nnz_offd)
               {
                  lfactor = HYPRE_THRUST_CALL( reduce,
                                               thrust::make_transform_iterator(B_offd_data,            absolute_value<HYPRE_Real>()),
                                               thrust::make_transform_iterator(B_offd_data + nnz_offd, absolute_value<HYPRE_Real>()),
                                               lfactor,
                                               thrust::maximum<HYPRE_Real>() );

               }
#endif
            }
            else
#endif
            {
               for (i = 0; i < B_diag_i[B_num_rows]; i++)
                  if (hypre_abs(B_diag_data[i]) > lfactor)
                  {
                     lfactor = hypre_abs(B_diag_data[i]);
                  }
               for (i = 0; i < B_offd_i[B_num_rows]; i++)
                  if (hypre_abs(B_offd_data[i]) > lfactor)
                  {
                     lfactor = hypre_abs(B_offd_data[i]);
                  }
            }

            lfactor *= 1e-10; /* scaling factor: max|A_ij|*1e-10 */
            hypre_MPI_Allreduce(&lfactor, &factor, 1, HYPRE_MPI_REAL, hypre_MPI_MAX, hypre_ParCSRMatrixComm(A));
         }

         hypre_ParCSRMatrixAdd(factor, A, 1.0, B, &C);

         /*hypre_CSRMatrix *A_local, *B_local, *C_local, *C_tmp;

         MPI_Comm comm = hypre_ParCSRMatrixComm(A);
         HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
         HYPRE_BigInt global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
         HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(A);
         HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(A);
         HYPRE_Int A_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
         HYPRE_Int A_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
         HYPRE_Int A_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));
         HYPRE_Int B_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(B));
         HYPRE_Int B_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
         HYPRE_Int B_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));

         A_local = hypre_MergeDiagAndOffd(A);
         B_local = hypre_MergeDiagAndOffd(B);*/
         /* scale (penalize) G0 G0^T before adding it to the matrix */
         /*{
            HYPRE_Int i, nnz = hypre_CSRMatrixNumNonzeros(A_local);
            HYPRE_Real *data = hypre_CSRMatrixData(A_local);
            HYPRE_Real *dataB = hypre_CSRMatrixData(B_local);
            HYPRE_Int nnzB = hypre_CSRMatrixNumNonzeros(B_local);
            HYPRE_Real factor, lfactor;
            lfactor = -1;
            for (i = 0; i < nnzB; i++)
               if (hypre_abs(dataB[i]) > lfactor)
                  lfactor = hypre_abs(dataB[i]);
            lfactor *= 1e-10;
            hypre_MPI_Allreduce(&lfactor, &factor, 1, HYPRE_MPI_REAL, hypre_MPI_MAX,
                                hypre_ParCSRMatrixComm(A));
            for (i = 0; i < nnz; i++)
               data[i] *= factor;
         }
         C_tmp = hypre_CSRMatrixBigAdd(A_local, B_local);
         C_local = hypre_CSRMatrixBigDeleteZeros(C_tmp,0.0);
         if (C_local)
            hypre_CSRMatrixDestroy(C_tmp);
         else
            C_local = C_tmp;

         C = hypre_ParCSRMatrixCreate (comm,
                                       global_num_rows,
                                       global_num_cols,
                                       row_starts,
                                       col_starts,
                                       A_num_cols_offd + B_num_cols_offd,
                                       A_num_nonzeros_diag + B_num_nonzeros_diag,
                                       A_num_nonzeros_offd + B_num_nonzeros_offd);
         GenerateDiagAndOffd(C_local, C,
                             hypre_ParCSRMatrixFirstColDiag(A),
                             hypre_ParCSRMatrixLastColDiag(A));

         hypre_CSRMatrixDestroy(A_local);
         hypre_CSRMatrixDestroy(B_local);
         hypre_CSRMatrixDestroy(C_local);
         */

         hypre_ParCSRMatrixDestroy(A);

         *C_ptr = C;
      }

      hypre_ParCSRMatrixDestroy(G0t);
   }

   /* Make sure that the first entry in each row is the diagonal one. */
   /* hypre_CSRMatrixReorder(hypre_ParCSRMatrixDiag(ams_data -> A)); */

   /* Compute the l1 norm of the rows of A */
   if (ams_data -> A_relax_type >= 1 && ams_data -> A_relax_type <= 4)
   {
      HYPRE_Real *l1_norm_data = NULL;

      hypre_ParCSRComputeL1Norms(ams_data -> A, ams_data -> A_relax_type, NULL, &l1_norm_data);

      ams_data -> A_l1_norms = hypre_SeqVectorCreate(hypre_ParCSRMatrixNumRows(ams_data -> A));
      hypre_VectorData(ams_data -> A_l1_norms) = l1_norm_data;
      hypre_SeqVectorInitialize_v2(ams_data -> A_l1_norms,
                                   hypre_ParCSRMatrixMemoryLocation(ams_data -> A));
   }

   /* Chebyshev? */
   if (ams_data -> A_relax_type == 16)
   {
      hypre_ParCSRMaxEigEstimateCG(ams_data->A, 1, 10,
                                   &ams_data->A_max_eig_est,
                                   &ams_data->A_min_eig_est);
   }

   /* If not given, compute Gx, Gy and Gz */
   {
      if (ams_data -> x != NULL &&
          (ams_data -> dim == 1 || ams_data -> y != NULL) &&
          (ams_data -> dim <= 2 || ams_data -> z != NULL))
      {
         input_info = 1;
      }

      if (ams_data -> Gx != NULL &&
          (ams_data -> dim == 1 || ams_data -> Gy != NULL) &&
          (ams_data -> dim <= 2 || ams_data -> Gz != NULL))
      {
         input_info = 2;
      }

      if (input_info == 1)
      {
         ams_data -> Gx = hypre_ParVectorInRangeOf(ams_data -> G);
         hypre_ParCSRMatrixMatvec (1.0, ams_data -> G, ams_data -> x, 0.0, ams_data -> Gx);
         if (ams_data -> dim >= 2)
         {
            ams_data -> Gy = hypre_ParVectorInRangeOf(ams_data -> G);
            hypre_ParCSRMatrixMatvec (1.0, ams_data -> G, ams_data -> y, 0.0, ams_data -> Gy);
         }
         if (ams_data -> dim == 3)
         {
            ams_data -> Gz = hypre_ParVectorInRangeOf(ams_data -> G);
            hypre_ParCSRMatrixMatvec (1.0, ams_data -> G, ams_data -> z, 0.0, ams_data -> Gz);
         }
      }
   }

   if (ams_data -> Pi == NULL && ams_data -> Pix == NULL)
   {
      if (ams_data -> cycle_type == 20)
         /* Construct the combined interpolation matrix [G,Pi] */
         hypre_AMSComputeGPi(ams_data -> A,
                             ams_data -> G,
                             ams_data -> Gx,
                             ams_data -> Gy,
                             ams_data -> Gz,
                             ams_data -> dim,
                             &ams_data -> Pi);
      else if (ams_data -> cycle_type > 10)
         /* Construct Pi{x,y,z} instead of Pi = [Pix,Piy,Piz] */
         hypre_AMSComputePixyz(ams_data -> A,
                               ams_data -> G,
                               ams_data -> Gx,
                               ams_data -> Gy,
                               ams_data -> Gz,
                               ams_data -> dim,
                               &ams_data -> Pix,
                               &ams_data -> Piy,
                               &ams_data -> Piz);
      else
         /* Construct the Pi interpolation matrix */
         hypre_AMSComputePi(ams_data -> A,
                            ams_data -> G,
                            ams_data -> Gx,
                            ams_data -> Gy,
                            ams_data -> Gz,
                            ams_data -> dim,
                            &ams_data -> Pi);
   }

   /* Keep Gx, Gy and Gz only if use the method with discrete divergence
      stabilization (where we use them to compute the local mesh size). */
   if (input_info == 1 && ams_data -> cycle_type != 9)
   {
      hypre_ParVectorDestroy(ams_data -> Gx);
      if (ams_data -> dim >= 2)
      {
         hypre_ParVectorDestroy(ams_data -> Gy);
      }
      if (ams_data -> dim == 3)
      {
         hypre_ParVectorDestroy(ams_data -> Gz);
      }
   }

   /* Create the AMG solver on the range of G^T */
   if (!ams_data -> beta_is_zero && ams_data -> cycle_type != 20)
   {
      HYPRE_BoomerAMGCreate(&ams_data -> B_G);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_G, ams_data -> B_G_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_G, ams_data -> B_G_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_G, ams_data -> B_G_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_G, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_G, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_G, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_G, ams_data -> B_G_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_G, ams_data -> B_G_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_G, ams_data -> B_G_Pmax);
      HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_G, 2); /* don't coarsen to 0 */

      /* Generally, don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_G, ams_data -> B_G_coarse_relax_type, 3);

      if (ams_data -> cycle_type == 0)
      {
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_G, 2);
      }

      /* If not given, construct the coarse space matrix by RAP */
      if (!ams_data -> A_G)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> G))
         {
            hypre_MatvecCommPkgCreate(ams_data -> G);
         }

         if (!hypre_ParCSRMatrixCommPkg(ams_data -> A))
         {
            hypre_MatvecCommPkgCreate(ams_data -> A);
         }

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            ams_data -> A_G = hypre_ParCSRMatrixRAPKT(ams_data -> G,
                                                      ams_data -> A,
                                                      ams_data -> G, 1);
         }
         else
#endif
         {
            hypre_BoomerAMGBuildCoarseOperator(ams_data -> G,
                                               ams_data -> A,
                                               ams_data -> G,
                                               &ams_data -> A_G);
         }

         /* Make sure that A_G has no zero rows (this can happen
            if beta is zero in part of the domain). */
         hypre_ParCSRMatrixFixZeroRows(ams_data -> A_G);
         ams_data -> owns_A_G = 1;
      }

      HYPRE_BoomerAMGSetup(ams_data -> B_G,
                           (HYPRE_ParCSRMatrix)ams_data -> A_G,
                           NULL, NULL);
   }

   if (ams_data -> cycle_type > 10 && ams_data -> cycle_type != 20)
      /* Create the AMG solvers on the range of Pi{x,y,z}^T */
   {
      HYPRE_BoomerAMGCreate(&ams_data -> B_Pix);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Pix, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Pix, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Pix, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Pix, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pix, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Pix, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Pix, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Pix, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Pix, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Pix, ams_data -> B_Pi_Pmax);
      HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Pix, 2);

      HYPRE_BoomerAMGCreate(&ams_data -> B_Piy);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Piy, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Piy, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Piy, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Piy, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piy, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Piy, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Piy, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Piy, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Piy, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Piy, ams_data -> B_Pi_Pmax);
      HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Piy, 2);

      HYPRE_BoomerAMGCreate(&ams_data -> B_Piz);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Piz, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Piz, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Piz, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Piz, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piz, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Piz, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Piz, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Piz, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Piz, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Piz, ams_data -> B_Pi_Pmax);
      HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Piz, 2);

      /* Generally, don't use exact solve on the coarsest level (matrices may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Pix, ams_data -> B_Pi_coarse_relax_type, 3);
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Piy, ams_data -> B_Pi_coarse_relax_type, 3);
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Piz, ams_data -> B_Pi_coarse_relax_type, 3);

      if (ams_data -> cycle_type == 0)
      {
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pix, 2);
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piy, 2);
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Piz, 2);
      }

      /* Construct the coarse space matrices by RAP */
      if (!hypre_ParCSRMatrixCommPkg(ams_data -> Pix))
      {
         hypre_MatvecCommPkgCreate(ams_data -> Pix);
      }

#if defined(HYPRE_USING_GPU)
      if (exec == HYPRE_EXEC_DEVICE)
      {
         ams_data -> A_Pix = hypre_ParCSRMatrixRAPKT(ams_data -> Pix, ams_data -> A, ams_data -> Pix, 1);
      }
      else
#endif
      {
         hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pix,
                                            ams_data -> A,
                                            ams_data -> Pix,
                                            &ams_data -> A_Pix);
      }

      /* Make sure that A_Pix has no zero rows (this can happen
         for some kinds of boundary conditions with contact). */
      hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Pix);

      HYPRE_BoomerAMGSetup(ams_data -> B_Pix,
                           (HYPRE_ParCSRMatrix)ams_data -> A_Pix,
                           NULL, NULL);

      if (ams_data -> Piy)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> Piy))
         {
            hypre_MatvecCommPkgCreate(ams_data -> Piy);
         }

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            ams_data -> A_Piy = hypre_ParCSRMatrixRAPKT(ams_data -> Piy,
                                                        ams_data -> A,
                                                        ams_data -> Piy, 1);
         }
         else
#endif
         {
            hypre_BoomerAMGBuildCoarseOperator(ams_data -> Piy,
                                               ams_data -> A,
                                               ams_data -> Piy,
                                               &ams_data -> A_Piy);
         }

         /* Make sure that A_Piy has no zero rows (this can happen
            for some kinds of boundary conditions with contact). */
         hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Piy);

         HYPRE_BoomerAMGSetup(ams_data -> B_Piy,
                              (HYPRE_ParCSRMatrix)ams_data -> A_Piy,
                              NULL, NULL);
      }

      if (ams_data -> Piz)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> Piz))
         {
            hypre_MatvecCommPkgCreate(ams_data -> Piz);
         }

#if defined(HYPRE_USING_GPU)
         if (exec == HYPRE_EXEC_DEVICE)
         {
            ams_data -> A_Piz = hypre_ParCSRMatrixRAPKT(ams_data -> Piz,
                                                        ams_data -> A,
                                                        ams_data -> Piz, 1);
         }
         else
#endif
         {
            hypre_BoomerAMGBuildCoarseOperator(ams_data -> Piz,
                                               ams_data -> A,
                                               ams_data -> Piz,
                                               &ams_data -> A_Piz);
         }

         /* Make sure that A_Piz has no zero rows (this can happen
            for some kinds of boundary conditions with contact). */
         hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Piz);

         HYPRE_BoomerAMGSetup(ams_data -> B_Piz,
                              (HYPRE_ParCSRMatrix)ams_data -> A_Piz,
                              NULL, NULL);
      }
   }
   else
      /* Create the AMG solver on the range of Pi^T */
   {
      HYPRE_BoomerAMGCreate(&ams_data -> B_Pi);
      HYPRE_BoomerAMGSetCoarsenType(ams_data -> B_Pi, ams_data -> B_Pi_coarsen_type);
      HYPRE_BoomerAMGSetAggNumLevels(ams_data -> B_Pi, ams_data -> B_Pi_agg_levels);
      HYPRE_BoomerAMGSetRelaxType(ams_data -> B_Pi, ams_data -> B_Pi_relax_type);
      HYPRE_BoomerAMGSetNumSweeps(ams_data -> B_Pi, 1);
      HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pi, 25);
      HYPRE_BoomerAMGSetTol(ams_data -> B_Pi, 0.0);
      HYPRE_BoomerAMGSetMaxIter(ams_data -> B_Pi, 1);
      HYPRE_BoomerAMGSetStrongThreshold(ams_data -> B_Pi, ams_data -> B_Pi_theta);
      HYPRE_BoomerAMGSetInterpType(ams_data -> B_Pi, ams_data -> B_Pi_interp_type);
      HYPRE_BoomerAMGSetPMaxElmts(ams_data -> B_Pi, ams_data -> B_Pi_Pmax);
      HYPRE_BoomerAMGSetMinCoarseSize(ams_data -> B_Pi, 2); /* don't coarsen to 0 */

      /* Generally, don't use exact solve on the coarsest level (matrix may be singular) */
      HYPRE_BoomerAMGSetCycleRelaxType(ams_data -> B_Pi, ams_data -> B_Pi_coarse_relax_type, 3);

      if (ams_data -> cycle_type == 0)
      {
         HYPRE_BoomerAMGSetMaxLevels(ams_data -> B_Pi, 2);
      }

      /* If not given, construct the coarse space matrix by RAP and
         notify BoomerAMG that this is a dim x dim block system. */
      if (!ams_data -> A_Pi)
      {
         if (!hypre_ParCSRMatrixCommPkg(ams_data -> Pi))
         {
            hypre_MatvecCommPkgCreate(ams_data -> Pi);
         }

         if (!hypre_ParCSRMatrixCommPkg(ams_data -> A))
         {
            hypre_MatvecCommPkgCreate(ams_data -> A);
         }

         if (ams_data -> cycle_type == 9)
         {
            /* Add a discrete divergence term to A before computing  Pi^t A Pi */
            {
               hypre_ParCSRMatrix *Gt, *GGt = NULL, *ApGGt;
               hypre_ParCSRMatrixTranspose(ams_data -> G, &Gt, 1);

               /* scale GGt by h^2 */
               {
                  HYPRE_Real h2;
                  HYPRE_Int i, j, k, ne;

                  hypre_CSRMatrix *Gt_diag = hypre_ParCSRMatrixDiag(Gt);
                  HYPRE_Int Gt_num_rows = hypre_CSRMatrixNumRows(Gt_diag);
                  HYPRE_Int *Gt_diag_I = hypre_CSRMatrixI(Gt_diag);
                  HYPRE_Int *Gt_diag_J = hypre_CSRMatrixJ(Gt_diag);
                  HYPRE_Real *Gt_diag_data = hypre_CSRMatrixData(Gt_diag);

                  hypre_CSRMatrix *Gt_offd = hypre_ParCSRMatrixOffd(Gt);
                  HYPRE_Int *Gt_offd_I = hypre_CSRMatrixI(Gt_offd);
                  HYPRE_Real *Gt_offd_data = hypre_CSRMatrixData(Gt_offd);

                  HYPRE_Real *Gx_data = hypre_VectorData(hypre_ParVectorLocalVector(ams_data -> Gx));
                  HYPRE_Real *Gy_data = hypre_VectorData(hypre_ParVectorLocalVector(ams_data -> Gy));
                  HYPRE_Real *Gz_data = hypre_VectorData(hypre_ParVectorLocalVector(ams_data -> Gz));

#if defined(HYPRE_USING_GPU)
                  if (exec == HYPRE_EXEC_DEVICE)
                  {
                     dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
                     dim3 gDim = hypre_GetDefaultDeviceGridDimension(Gt_num_rows, "warp", bDim);
                     HYPRE_GPU_LAUNCH( hypreGPUKernel_AMSSetupScaleGGt, gDim, bDim,
                                       Gt_num_rows, Gt_diag_I, Gt_diag_J, Gt_diag_data, Gt_offd_I, Gt_offd_data,
                                       Gx_data, Gy_data, Gz_data );
                  }
                  else
#endif
                  {
                     for (i = 0; i < Gt_num_rows; i++)
                     {
                        /* determine the characteristic mesh size for vertex i */
                        h2 = 0.0;
                        ne = 0;
                        for (j = Gt_diag_I[i]; j < Gt_diag_I[i + 1]; j++)
                        {
                           k = Gt_diag_J[j];
                           h2 += Gx_data[k] * Gx_data[k] + Gy_data[k] * Gy_data[k] + Gz_data[k] * Gz_data[k];
                           ne++;
                        }

                        if (ne != 0)
                        {
                           h2 /= ne;
                           for (j = Gt_diag_I[i]; j < Gt_diag_I[i + 1]; j++)
                           {
                              Gt_diag_data[j] *= h2;
                           }
                           for (j = Gt_offd_I[i]; j < Gt_offd_I[i + 1]; j++)
                           {
                              Gt_offd_data[j] *= h2;
                           }
                        }
                     }
                  }
               }

               /* we only needed Gx, Gy and Gz to compute the local mesh size */
               if (input_info == 1)
               {
                  hypre_ParVectorDestroy(ams_data -> Gx);
                  if (ams_data -> dim >= 2)
                  {
                     hypre_ParVectorDestroy(ams_data -> Gy);
                  }
                  if (ams_data -> dim == 3)
                  {
                     hypre_ParVectorDestroy(ams_data -> Gz);
                  }
               }

#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  GGt = hypre_ParCSRMatMat(ams_data -> G, Gt);
               }
               else
#endif
               {
                  GGt = hypre_ParMatmul(ams_data -> G, Gt);
               }
               hypre_ParCSRMatrixDestroy(Gt);

               /* hypre_ParCSRMatrixAdd(GGt, A, &ams_data -> A); */
               hypre_ParCSRMatrixAdd(1.0, GGt, 1.0, ams_data -> A, &ApGGt);
               /*{
                  hypre_ParCSRMatrix *A = GGt;
                  hypre_ParCSRMatrix *B = ams_data -> A;
                  hypre_ParCSRMatrix **C_ptr = &ApGGt;

                  hypre_ParCSRMatrix *C;
                  hypre_CSRMatrix *A_local, *B_local, *C_local;

                  MPI_Comm comm = hypre_ParCSRMatrixComm(A);
                  HYPRE_BigInt global_num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
                  HYPRE_BigInt global_num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
                  HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(A);
                  HYPRE_BigInt *col_starts = hypre_ParCSRMatrixColStarts(A);
                  HYPRE_Int A_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(A));
                  HYPRE_Int A_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(A));
                  HYPRE_Int A_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(A));
                  HYPRE_Int B_num_cols_offd = hypre_CSRMatrixNumCols(hypre_ParCSRMatrixOffd(B));
                  HYPRE_Int B_num_nonzeros_diag = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixDiag(B));
                  HYPRE_Int B_num_nonzeros_offd = hypre_CSRMatrixNumNonzeros(hypre_ParCSRMatrixOffd(B));

                  A_local = hypre_MergeDiagAndOffd(A);
                  B_local = hypre_MergeDiagAndOffd(B);
                  C_local = hypre_CSRMatrixBigAdd(A_local, B_local);
                  hypre_CSRMatrixBigJtoJ(C_local);

                  C = hypre_ParCSRMatrixCreate (comm,
                                                global_num_rows,
                                                global_num_cols,
                                                row_starts,
                                                col_starts,
                                                A_num_cols_offd + B_num_cols_offd,
                                                A_num_nonzeros_diag + B_num_nonzeros_diag,
                                                A_num_nonzeros_offd + B_num_nonzeros_offd);
                  GenerateDiagAndOffd(C_local, C,
                                      hypre_ParCSRMatrixFirstColDiag(A),
                                      hypre_ParCSRMatrixLastColDiag(A));

                  hypre_CSRMatrixDestroy(A_local);
                  hypre_CSRMatrixDestroy(B_local);
                  hypre_CSRMatrixDestroy(C_local);

                  *C_ptr = C;
               }*/

               hypre_ParCSRMatrixDestroy(GGt);

#if defined(HYPRE_USING_GPU)
               if (exec == HYPRE_EXEC_DEVICE)
               {
                  ams_data -> A_Pi = hypre_ParCSRMatrixRAPKT(ams_data -> Pi, ApGGt, ams_data -> Pi, 1);
               }
               else
#endif
               {
                  hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                                     ApGGt,
                                                     ams_data -> Pi,
                                                     &ams_data -> A_Pi);
               }
            }
         }
         else
         {
#if defined(HYPRE_USING_GPU)
            if (exec == HYPRE_EXEC_DEVICE)
            {
               ams_data -> A_Pi = hypre_ParCSRMatrixRAPKT(ams_data -> Pi, ams_data -> A, ams_data -> Pi, 1);
            }
            else
#endif
            {
               hypre_BoomerAMGBuildCoarseOperator(ams_data -> Pi,
                                                  ams_data -> A,
                                                  ams_data -> Pi,
                                                  &ams_data -> A_Pi);
            }
         }

         ams_data -> owns_A_Pi = 1;

         if (ams_data -> cycle_type != 20)
         {
            HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim);
         }
         else
         {
            HYPRE_BoomerAMGSetNumFunctions(ams_data -> B_Pi, ams_data -> dim + 1);
         }
         /* HYPRE_BoomerAMGSetNodal(ams_data -> B_Pi, 1); */
      }

      /* Make sure that A_Pi has no zero rows (this can happen for
         some kinds of boundary conditions with contact). */
      hypre_ParCSRMatrixFixZeroRows(ams_data -> A_Pi);

      HYPRE_BoomerAMGSetup(ams_data -> B_Pi,
                           (HYPRE_ParCSRMatrix)ams_data -> A_Pi,
                           0, 0);
   }

   /* Allocate temporary vectors */
   ams_data -> r0 = hypre_ParVectorInRangeOf(ams_data -> A);
   ams_data -> g0 = hypre_ParVectorInRangeOf(ams_data -> A);
   if (ams_data -> A_G)
   {
      ams_data -> r1 = hypre_ParVectorInRangeOf(ams_data -> A_G);
      ams_data -> g1 = hypre_ParVectorInRangeOf(ams_data -> A_G);
   }
   if (ams_data -> r1 == NULL && ams_data -> A_Pix)
   {
      ams_data -> r1 = hypre_ParVectorInRangeOf(ams_data -> A_Pix);
      ams_data -> g1 = hypre_ParVectorInRangeOf(ams_data -> A_Pix);
   }
   if (ams_data -> Pi)
   {
      ams_data -> r2 = hypre_ParVectorInDomainOf(ams_data -> Pi);
      ams_data -> g2 = hypre_ParVectorInDomainOf(ams_data -> Pi);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSSolve
 *
 * Solve the system A x = b.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSSolve(void *solver,
                         hypre_ParCSRMatrix *A,
                         hypre_ParVector *b,
                         hypre_ParVector *x)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   HYPRE_Int  i, my_id = -1;
   HYPRE_Real r0_norm = 1.0;
   HYPRE_Real r_norm  = 1.0;
   HYPRE_Real b_norm  = 1.0;
   HYPRE_Real relative_resid = 0, old_resid;

   char cycle[30];
   hypre_ParCSRMatrix *Ai[5], *Pi[5];
   HYPRE_Solver Bi[5];
   HYPRE_PtrToSolverFcn HBi[5];
   hypre_ParVector *ri[5], *gi[5];
   HYPRE_Int needZ = 0;

   hypre_ParVector *z = ams_data -> zz;

   Ai[0] = ams_data -> A_G;    Pi[0] = ams_data -> G;
   Ai[1] = ams_data -> A_Pi;   Pi[1] = ams_data -> Pi;
   Ai[2] = ams_data -> A_Pix;  Pi[2] = ams_data -> Pix;
   Ai[3] = ams_data -> A_Piy;  Pi[3] = ams_data -> Piy;
   Ai[4] = ams_data -> A_Piz;  Pi[4] = ams_data -> Piz;

   Bi[0] = ams_data -> B_G;    HBi[0] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;
   Bi[1] = ams_data -> B_Pi;   HBi[1] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGBlockSolve;
   Bi[2] = ams_data -> B_Pix;  HBi[2] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;
   Bi[3] = ams_data -> B_Piy;  HBi[3] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;
   Bi[4] = ams_data -> B_Piz;  HBi[4] = (HYPRE_PtrToSolverFcn) hypre_BoomerAMGSolve;

   ri[0] = ams_data -> r1;     gi[0] = ams_data -> g1;
   ri[1] = ams_data -> r2;     gi[1] = ams_data -> g2;
   ri[2] = ams_data -> r1;     gi[2] = ams_data -> g1;
   ri[3] = ams_data -> r1;     gi[3] = ams_data -> g1;
   ri[4] = ams_data -> r1;     gi[4] = ams_data -> g1;

   /* may need to create an additional temporary vector for relaxation */
#if defined(HYPRE_USING_GPU)
   HYPRE_ExecutionPolicy exec = hypre_GetExecPolicy1( hypre_ParCSRMatrixMemoryLocation(A) );

   if (exec == HYPRE_EXEC_DEVICE)
   {
      needZ = ams_data -> A_relax_type == 2 || ams_data -> A_relax_type == 4 ||
              ams_data -> A_relax_type == 16;
   }
   else
#endif
   {
      needZ = hypre_NumThreads() > 1 || ams_data -> A_relax_type == 16;
   }

   if (needZ && !z)
   {
      z = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                                hypre_ParCSRMatrixGlobalNumRows(A),
                                hypre_ParCSRMatrixRowStarts(A));
      hypre_ParVectorInitialize(z);
      ams_data -> zz = z;
   }

   if (ams_data -> print_level > 0)
   {
      hypre_MPI_Comm_rank(hypre_ParCSRMatrixComm(A), &my_id);
   }

   /* Compatible subspace projection for problems with zero-conductivity regions.
      Note that this modifies the input (r.h.s.) vector b! */
   if ( (ams_data -> B_G0) &&
        (++ams_data->solve_counter % ( ams_data -> projection_frequency ) == 0) )
   {
      /* hypre_printf("Projecting onto the compatible subspace...\n"); */
      hypre_AMSProjectOutGradients(ams_data, b);
   }

   if (ams_data -> beta_is_zero)
   {
      switch (ams_data -> cycle_type)
      {
         case 0:
            hypre_sprintf(cycle, "%s", "0");
            break;

         case 1:
         case 3:
         case 5:
         case 7:
         default:
            hypre_sprintf(cycle, "%s", "020");
            break;

         case 2:
         case 4:
         case 6:
         case 8:
            hypre_sprintf(cycle, "%s", "(0+2)");
            break;

         case 11:
         case 13:
            hypre_sprintf(cycle, "%s", "0345430");
            break;

         case 12:
            hypre_sprintf(cycle, "%s", "(0+3+4+5)");
            break;

         case 14:
            hypre_sprintf(cycle, "%s", "0(+3+4+5)0");
            break;
      }
   }
   else
   {
      switch (ams_data -> cycle_type)
      {
         case 0:
            hypre_sprintf(cycle, "%s", "010");
            break;
         case 1:
         default:
            hypre_sprintf(cycle, "%s", "01210");
            break;

         case 2:
            hypre_sprintf(cycle, "%s", "(0+1+2)");
            break;

         case 3:
            hypre_sprintf(cycle, "%s", "02120");
            break;

         case 4:
            hypre_sprintf(cycle, "%s", "(010+2)");
            break;

         case 5:
            hypre_sprintf(cycle, "%s", "0102010");
            break;

         case 6:
            hypre_sprintf(cycle, "%s", "(020+1)");
            break;

         case 7:
            hypre_sprintf(cycle, "%s", "0201020");
            break;

         case 8:
            hypre_sprintf(cycle, "%s", "0(+1+2)0");
            break;

         case 9:
            hypre_sprintf(cycle, "%s", "01210");
            break;

         case 11:
            hypre_sprintf(cycle, "%s", "013454310");
            break;

         case 12:
            hypre_sprintf(cycle, "%s", "(0+1+3+4+5)");
            break;

         case 13:
            hypre_sprintf(cycle, "%s", "034515430");
            break;

         case 14:
            hypre_sprintf(cycle, "%s", "01(+3+4+5)10");
            break;

         case 20:
            hypre_sprintf(cycle, "%s", "020");
            break;
      }
   }

   for (i = 0; i < ams_data -> maxit; i++)
   {
      /* Compute initial residual norms */
      if (ams_data -> maxit > 1 && i == 0)
      {
         hypre_ParVectorCopy(b, ams_data -> r0);
         hypre_ParCSRMatrixMatvec(-1.0, ams_data -> A, x, 1.0, ams_data -> r0);
         r_norm = hypre_sqrt(hypre_ParVectorInnerProd(ams_data -> r0, ams_data -> r0));
         r0_norm = r_norm;
         b_norm = hypre_sqrt(hypre_ParVectorInnerProd(b, b));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ams_data -> print_level > 0)
         {
            hypre_printf("                                            relative\n");
            hypre_printf("               residual        factor       residual\n");
            hypre_printf("               --------        ------       --------\n");
            hypre_printf("    Initial    %e                 %e\n",
                         r_norm, relative_resid);
         }
      }

      /* Apply the preconditioner */
      hypre_ParCSRSubspacePrec(ams_data -> A,
                               ams_data -> A_relax_type,
                               ams_data -> A_relax_times,
                               ams_data -> A_l1_norms ? hypre_VectorData(ams_data -> A_l1_norms) : NULL,
                               ams_data -> A_relax_weight,
                               ams_data -> A_omega,
                               ams_data -> A_max_eig_est,
                               ams_data -> A_min_eig_est,
                               ams_data -> A_cheby_order,
                               ams_data -> A_cheby_fraction,
                               Ai, Bi, HBi, Pi, ri, gi,
                               b, x,
                               ams_data -> r0,
                               ams_data -> g0,
                               cycle,
                               z);

      /* Compute new residual norms */
      if (ams_data -> maxit > 1)
      {
         old_resid = r_norm;
         hypre_ParVectorCopy(b, ams_data -> r0);
         hypre_ParCSRMatrixMatvec(-1.0, ams_data -> A, x, 1.0, ams_data -> r0);
         r_norm = hypre_sqrt(hypre_ParVectorInnerProd(ams_data -> r0, ams_data -> r0));
         if (b_norm)
         {
            relative_resid = r_norm / b_norm;
         }
         else
         {
            relative_resid = r_norm;
         }
         if (my_id == 0 && ams_data -> print_level > 0)
            hypre_printf("    Cycle %2d   %e    %f     %e \n",
                         i + 1, r_norm, r_norm / old_resid, relative_resid);
      }

      if (relative_resid < ams_data -> tol)
      {
         i++;
         break;
      }
   }

   if (my_id == 0 && ams_data -> print_level > 0 && ams_data -> maxit > 1)
   {
      hypre_printf("\n\n Average Convergence Factor = %f\n\n",
                   hypre_pow((r_norm / r0_norm), (1.0 / (HYPRE_Real) i)));
   }

   ams_data -> num_iterations = i;
   ams_data -> rel_resid_norm = relative_resid;

   if (ams_data -> num_iterations == ams_data -> maxit && ams_data -> tol > 0.0)
   {
      hypre_error(HYPRE_ERROR_CONV);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRSubspacePrec
 *
 * General subspace preconditioner for A0 y = x, based on ParCSR storage.
 *
 * P[i] and A[i] are the interpolation and coarse grid matrices for
 * the (i+1)'th subspace. B[i] is an AMG solver for A[i]. r[i] and g[i]
 * are temporary vectors. A0_* are the fine grid smoothing parameters.
 *
 * The default mode is multiplicative, '+' changes the next correction
 * to additive, based on residual computed at '('.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_ParCSRSubspacePrec(/* fine space matrix */
   hypre_ParCSRMatrix *A0,
   /* relaxation parameters */
   HYPRE_Int A0_relax_type,
   HYPRE_Int A0_relax_times,
   HYPRE_Real *A0_l1_norms,
   HYPRE_Real A0_relax_weight,
   HYPRE_Real A0_omega,
   HYPRE_Real A0_max_eig_est,
   HYPRE_Real A0_min_eig_est,
   HYPRE_Int A0_cheby_order,
   HYPRE_Real A0_cheby_fraction,
   /* subspace matrices */
   hypre_ParCSRMatrix **A,
   /* subspace preconditioners */
   HYPRE_Solver *B,
   /* hypre solver functions for B */
   HYPRE_PtrToSolverFcn *HB,
   /* subspace interpolations */
   hypre_ParCSRMatrix **P,
   /* temporary subspace vectors */
   hypre_ParVector **r,
   hypre_ParVector **g,
   /* right-hand side */
   hypre_ParVector *x,
   /* current approximation */
   hypre_ParVector *y,
   /* current residual */
   hypre_ParVector *r0,
   /* temporary vector */
   hypre_ParVector *g0,
   char *cycle,
   /* temporary vector */
   hypre_ParVector *z)
{
   char *op;
   HYPRE_Int use_saved_residual = 0;

   for (op = cycle; *op != '\0'; op++)
   {
      /* do nothing */
      if (*op == ')')
      {
         continue;
      }

      /* compute the residual: r = x - Ay */
      else if (*op == '(')
      {
         hypre_ParVectorCopy(x, r0);
         hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, r0);
      }

      /* switch to additive correction */
      else if (*op == '+')
      {
         use_saved_residual = 1;
         continue;
      }

      /* smooth: y += S (x - Ay) */
      else if (*op == '0')
      {
         hypre_ParCSRRelax(A0, x,
                           A0_relax_type,
                           A0_relax_times,
                           A0_l1_norms,
                           A0_relax_weight,
                           A0_omega,
                           A0_max_eig_est,
                           A0_min_eig_est,
                           A0_cheby_order,
                           A0_cheby_fraction,
                           y, g0, z);
      }

      /* subspace correction: y += P B^{-1} P^t r */
      else
      {
         HYPRE_Int i = *op - '1';
         if (i < 0)
         {
            hypre_error_in_arg(16);
         }

         /* skip empty subspaces */
         if (!A[i]) { continue; }

         /* compute the residual? */
         if (use_saved_residual)
         {
            use_saved_residual = 0;
            hypre_ParCSRMatrixMatvecT(1.0, P[i], r0, 0.0, r[i]);
         }
         else
         {
            hypre_ParVectorCopy(x, g0);
            hypre_ParCSRMatrixMatvec(-1.0, A0, y, 1.0, g0);
            hypre_ParCSRMatrixMatvecT(1.0, P[i], g0, 0.0, r[i]);
         }

         hypre_ParVectorSetConstantValues(g[i], 0.0);
         (*HB[i]) (B[i], (HYPRE_Matrix)A[i],
                   (HYPRE_Vector)r[i], (HYPRE_Vector)g[i]);
         hypre_ParCSRMatrixMatvec(1.0, P[i], g[i], 0.0, g0);
         hypre_ParVectorAxpy(1.0, g0, y);
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSGetNumIterations
 *
 * Get the number of AMS iterations.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSGetNumIterations(void *solver,
                                    HYPRE_Int *num_iterations)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   *num_iterations = ams_data -> num_iterations;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSGetFinalRelativeResidualNorm
 *
 * Get the final relative residual norm in AMS.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSGetFinalRelativeResidualNorm(void *solver,
                                                HYPRE_Real *rel_resid_norm)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;
   *rel_resid_norm = ams_data -> rel_resid_norm;
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSProjectOutGradients
 *
 * For problems with zero-conductivity regions, project the vector onto the
 * compatible subspace: x = (I - G0 (G0^t G0)^{-1} G0^T) x, where G0 is the
 * discrete gradient restricted to the interior nodes of the regions with
 * zero conductivity. This ensures that x is orthogonal to the gradients in
 * the range of G0.
 *
 * This function is typically called after the solution iteration is complete,
 * in order to facilitate the visualization of the computed field. Without it
 * the values in the zero-conductivity regions contain kernel components.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSProjectOutGradients(void *solver,
                                       hypre_ParVector *x)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   if (ams_data -> B_G0)
   {
      hypre_ParCSRMatrixMatvecT(1.0, ams_data -> G0, x, 0.0, ams_data -> r1);
      hypre_ParVectorSetConstantValues(ams_data -> g1, 0.0);
      hypre_BoomerAMGSolve(ams_data -> B_G0, ams_data -> A_G0, ams_data -> r1, ams_data -> g1);
      hypre_ParCSRMatrixMatvec(1.0, ams_data -> G0, ams_data -> g1, 0.0, ams_data -> g0);
      hypre_ParVectorAxpy(-1.0, ams_data -> g0, x);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSConstructDiscreteGradient
 *
 * Construct and return the lowest-order discrete gradient matrix G, based on:
 * - a matrix on the egdes (e.g. the stiffness matrix A)
 * - a vector on the vertices (e.g. the x coordinates)
 * - the array edge_vertex, which lists the global indexes of the
 *   vertices of the local edges.
 *
 * We assume that edge_vertex lists the edge vertices consecutively,
 * and that the orientation of all edges is consistent. More specificaly:
 * If edge_orientation = 1, the edges are already oriented.
 * If edge_orientation = 2, the orientation of edge i depends only on the
 *                          sign of edge_vertex[2*i+1] - edge_vertex[2*i].
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSConstructDiscreteGradient(hypre_ParCSRMatrix *A,
                                             hypre_ParVector *x_coord,
                                             HYPRE_BigInt *edge_vertex,
                                             HYPRE_Int edge_orientation,
                                             hypre_ParCSRMatrix **G_ptr)
{
   hypre_ParCSRMatrix *G;

   HYPRE_Int nedges;

   nedges = hypre_ParCSRMatrixNumRows(A);

   /* Construct the local part of G based on edge_vertex and the edge
      and vertex partitionings from A and x_coord */
   {
      HYPRE_Int i, *I = hypre_CTAlloc(HYPRE_Int,  nedges + 1, HYPRE_MEMORY_HOST);
      HYPRE_Real *data = hypre_CTAlloc(HYPRE_Real,  2 * nedges, HYPRE_MEMORY_HOST);
      hypre_CSRMatrix *local = hypre_CSRMatrixCreate (nedges,
                                                      hypre_ParVectorGlobalSize(x_coord),
                                                      2 * nedges);

      for (i = 0; i <= nedges; i++)
      {
         I[i] = 2 * i;
      }

      if (edge_orientation == 1)
      {
         /* Assume that the edges are already oriented */
         for (i = 0; i < 2 * nedges; i += 2)
         {
            data[i]   = -1.0;
            data[i + 1] =  1.0;
         }
      }
      else if (edge_orientation == 2)
      {
         /* Assume that the edge orientation is based on the vertex indexes */
         for (i = 0; i < 2 * nedges; i += 2)
         {
            if (edge_vertex[i] < edge_vertex[i + 1])
            {
               data[i]   = -1.0;
               data[i + 1] =  1.0;
            }
            else
            {
               data[i]   =  1.0;
               data[i + 1] = -1.0;
            }
         }
      }
      else
      {
         hypre_error_in_arg(4);
      }

      hypre_CSRMatrixI(local) = I;
      hypre_CSRMatrixBigJ(local) = edge_vertex;
      hypre_CSRMatrixData(local) = data;

      hypre_CSRMatrixRownnz(local) = NULL;
      hypre_CSRMatrixOwnsData(local) = 1;
      hypre_CSRMatrixNumRownnz(local) = nedges;

      /* Generate the discrete gradient matrix */
      G = hypre_ParCSRMatrixCreate(hypre_ParCSRMatrixComm(A),
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   hypre_ParVectorGlobalSize(x_coord),
                                   hypre_ParCSRMatrixRowStarts(A),
                                   hypre_ParVectorPartitioning(x_coord),
                                   0, 0, 0);
      hypre_CSRMatrixBigJtoJ(local);
      GenerateDiagAndOffd(local, G,
                          hypre_ParVectorFirstIndex(x_coord),
                          hypre_ParVectorLastIndex(x_coord));


      /* Account for empty rows in G. These may appear when A includes only
         the interior (non-Dirichlet b.c.) edges. */
      {
         hypre_CSRMatrix *G_diag = hypre_ParCSRMatrixDiag(G);
         G_diag->num_cols = hypre_VectorSize(hypre_ParVectorLocalVector(x_coord));
      }

      /* Free the local matrix */
      hypre_CSRMatrixDestroy(local);
   }

   *G_ptr = G;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSFEISetup
 *
 * Construct an AMS solver object based on the following data:
 *
 *    A              - the edge element stiffness matrix
 *    num_vert       - number of vertices (nodes) in the processor
 *    num_local_vert - number of vertices owned by the processor
 *    vert_number    - global indexes of the vertices in the processor
 *    vert_coord     - coordinates of the vertices in the processor
 *    num_edges      - number of edges owned by the processor
 *    edge_vertex    - the vertices of the edges owned by the processor.
 *                     Vertices are in local numbering (the same as in
 *                     vert_number), and edge orientation is always from
 *                     the first to the second vertex.
 *
 * Here we distinguish between vertices that belong to elements in the
 * current processor, and the subset of these vertices that is owned by
 * the processor.
 *
 * This function is written specifically for input from the FEI and should
 * be called before hypre_AMSSetup().
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_AMSFEISetup(void *solver,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector *b,
                  hypre_ParVector *x,
                  HYPRE_Int num_vert,
                  HYPRE_Int num_local_vert,
                  HYPRE_BigInt *vert_number,
                  HYPRE_Real *vert_coord,
                  HYPRE_Int num_edges,
                  HYPRE_BigInt *edge_vertex)
{
   HYPRE_UNUSED_VAR(b);
   HYPRE_UNUSED_VAR(x);

   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   HYPRE_Int i, j;

   hypre_ParCSRMatrix *G;
   hypre_ParVector *x_coord, *y_coord, *z_coord;
   HYPRE_Real *x_data, *y_data, *z_data;

   MPI_Comm comm = hypre_ParCSRMatrixComm(A);
   HYPRE_BigInt vert_part[2], num_global_vert;
   HYPRE_BigInt vert_start, vert_end;
   HYPRE_BigInt big_local_vert = (HYPRE_BigInt) num_local_vert;

   /* Find the processor partitioning of the vertices */
   hypre_MPI_Scan(&big_local_vert, &vert_part[1], 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);
   vert_part[0] = vert_part[1] - big_local_vert;
   hypre_MPI_Allreduce(&big_local_vert, &num_global_vert, 1, HYPRE_MPI_BIG_INT, hypre_MPI_SUM, comm);

   /* Construct hypre parallel vectors for the vertex coordinates */
   x_coord = hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   hypre_ParVectorInitialize(x_coord);
   hypre_ParVectorOwnsData(x_coord) = 1;
   x_data = hypre_VectorData(hypre_ParVectorLocalVector(x_coord));

   y_coord = hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   hypre_ParVectorInitialize(y_coord);
   hypre_ParVectorOwnsData(y_coord) = 1;
   y_data = hypre_VectorData(hypre_ParVectorLocalVector(y_coord));

   z_coord = hypre_ParVectorCreate(comm, num_global_vert, vert_part);
   hypre_ParVectorInitialize(z_coord);
   hypre_ParVectorOwnsData(z_coord) = 1;
   z_data = hypre_VectorData(hypre_ParVectorLocalVector(z_coord));

   vert_start = hypre_ParVectorFirstIndex(x_coord);
   vert_end   = hypre_ParVectorLastIndex(x_coord);

   /* Save coordinates of locally owned vertices */
   for (i = 0; i < num_vert; i++)
   {
      if (vert_number[i] >= vert_start && vert_number[i] <= vert_end)
      {
         j = (HYPRE_Int)(vert_number[i] - vert_start);
         x_data[j] = vert_coord[3 * i];
         y_data[j] = vert_coord[3 * i + 1];
         z_data[j] = vert_coord[3 * i + 2];
      }
   }

   /* Change vertex numbers from local to global */
   for (i = 0; i < 2 * num_edges; i++)
   {
      edge_vertex[i] = vert_number[edge_vertex[i]];
   }

   /* Construct the local part of G based on edge_vertex */
   {
      /* HYPRE_Int num_edges = hypre_ParCSRMatrixNumRows(A); */
      HYPRE_Int *I = hypre_CTAlloc(HYPRE_Int,  num_edges + 1, HYPRE_MEMORY_HOST);
      HYPRE_Real *data = hypre_CTAlloc(HYPRE_Real,  2 * num_edges, HYPRE_MEMORY_HOST);
      hypre_CSRMatrix *local = hypre_CSRMatrixCreate (num_edges,
                                                      num_global_vert,
                                                      2 * num_edges);

      for (i = 0; i <= num_edges; i++)
      {
         I[i] = 2 * i;
      }

      /* Assume that the edge orientation is based on the vertex indexes */
      for (i = 0; i < 2 * num_edges; i += 2)
      {
         data[i]   =  1.0;
         data[i + 1] = -1.0;
      }

      hypre_CSRMatrixI(local) = I;
      hypre_CSRMatrixBigJ(local) = edge_vertex;
      hypre_CSRMatrixData(local) = data;

      hypre_CSRMatrixRownnz(local) = NULL;
      hypre_CSRMatrixOwnsData(local) = 1;
      hypre_CSRMatrixNumRownnz(local) = num_edges;

      G = hypre_ParCSRMatrixCreate(comm,
                                   hypre_ParCSRMatrixGlobalNumRows(A),
                                   num_global_vert,
                                   hypre_ParCSRMatrixRowStarts(A),
                                   vert_part,
                                   0, 0, 0);
      hypre_CSRMatrixBigJtoJ(local);
      GenerateDiagAndOffd(local, G, vert_start, vert_end);

      //hypre_CSRMatrixJ(local) = NULL;
      hypre_CSRMatrixDestroy(local);
   }

   ams_data -> G = G;

   ams_data -> x = x_coord;
   ams_data -> y = y_coord;
   ams_data -> z = z_coord;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_AMSFEIDestroy
 *
 * Free the additional memory allocated in hypre_AMSFEISetup().
 *
 * This function is written specifically for input from the FEI and should
 * be called before hypre_AMSDestroy().
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_AMSFEIDestroy(void *solver)
{
   hypre_AMSData *ams_data = (hypre_AMSData *) solver;

   if (ams_data -> G)
   {
      hypre_ParCSRMatrixDestroy(ams_data -> G);
   }

   if (ams_data -> x)
   {
      hypre_ParVectorDestroy(ams_data -> x);
   }
   if (ams_data -> y)
   {
      hypre_ParVectorDestroy(ams_data -> y);
   }
   if (ams_data -> z)
   {
      hypre_ParVectorDestroy(ams_data -> z);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_ParCSRComputeL1Norms Threads
 *
 * Compute the l1 norms of the rows of a given matrix, depending on
 * the option parameter:
 *
 * option 1 = Compute the l1 norm of the rows
 * option 2 = Compute the l1 norm of the (processor) off-diagonal
 *            part of the rows plus the diagonal of A
 * option 3 = Compute the l2 norm^2 of the rows
 * option 4 = Truncated version of option 2 based on Remark 6.2 in "Multigrid
 *            Smoothers for Ultra-Parallel Computing"
 *
 * The above computations are done in a CF manner, whenever the provided
 * cf_marker is not NULL.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRComputeL1NormsThreads(hypre_ParCSRMatrix *A,
                                  HYPRE_Int           option,
                                  HYPRE_Int           num_threads,
                                  HYPRE_Int          *cf_marker,
                                  HYPRE_Real        **l1_norm_ptr)
{
   HYPRE_Int i, j, k;
   HYPRE_Int num_rows = hypre_ParCSRMatrixNumRows(A);

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Int *A_diag_I = hypre_CSRMatrixI(A_diag);
   HYPRE_Int *A_diag_J = hypre_CSRMatrixJ(A_diag);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);
   HYPRE_Int *A_offd_I = hypre_CSRMatrixI(A_offd);
   HYPRE_Int *A_offd_J = hypre_CSRMatrixJ(A_offd);
   HYPRE_Real *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

   HYPRE_Real diag = 1.0;
   HYPRE_Real *l1_norm = hypre_TAlloc(HYPRE_Real, num_rows, hypre_ParCSRMatrixMemoryLocation(A));
   HYPRE_Int ii, ns, ne, rest, size;

   HYPRE_Int *cf_marker_offd = NULL;
   HYPRE_Int cf_diag;

   /* collect the cf marker data from other procs */
   if (cf_marker != NULL)
   {
      HYPRE_Int index;
      HYPRE_Int num_sends;
      HYPRE_Int start;
      HYPRE_Int *int_buf_data = NULL;

      hypre_ParCSRCommPkg  *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
      hypre_ParCSRCommHandle *comm_handle;

      if (num_cols_offd)
      {
         cf_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
      }
      num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
      if (hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends))
         int_buf_data = hypre_CTAlloc(HYPRE_Int,
                                      hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends),
                                      HYPRE_MEMORY_HOST);
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i + 1); j++)
         {
            int_buf_data[index++] = cf_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg, j)];
         }
      }
      comm_handle = hypre_ParCSRCommHandleCreate(11, comm_pkg, int_buf_data,
                                                 cf_marker_offd);
      hypre_ParCSRCommHandleDestroy(comm_handle);
      hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
   }

#ifdef HYPRE_USING_OPENMP
   #pragma omp parallel for private(i,ii,j,k,ns,ne,rest,size,diag,cf_diag) HYPRE_SMP_SCHEDULE
#endif
   for (k = 0; k < num_threads; k++)
   {
      size = num_rows / num_threads;
      rest = num_rows - size * num_threads;
      if (k < rest)
      {
         ns = k * size + k;
         ne = (k + 1) * size + k + 1;
      }
      else
      {
         ns = k * size + rest;
         ne = (k + 1) * size + rest;
      }

      if (option == 1)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            if (cf_marker == NULL)
            {
               /* Add the l1 norm of the diag part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  l1_norm[i] += hypre_abs(A_diag_data[j]);
               }
               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += hypre_abs(A_offd_data[j]);
                  }
               }
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the CF l1 norm of the diag part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
                  if (cf_diag == cf_marker[A_diag_J[j]])
                  {
                     l1_norm[i] += hypre_abs(A_diag_data[j]);
                  }
               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += hypre_abs(A_offd_data[j]);
                     }
               }
            }
         }
      }
      else if (option == 2)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            if (cf_marker == NULL)
            {
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if (ii == i || ii < ns || ii >= ne)
                  {
                     l1_norm[i] += hypre_abs(A_diag_data[j]);
                  }
               }
               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += hypre_abs(A_offd_data[j]);
                  }
               }
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if ((ii == i || ii < ns || ii >= ne) &&
                      (cf_diag == cf_marker[A_diag_J[j]]))
                  {
                     l1_norm[i] += hypre_abs(A_diag_data[j]);
                  }
               }
               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += hypre_abs(A_offd_data[j]);
                     }
                  }
               }
            }
         }
      }
      else if (option == 3)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
            {
               l1_norm[i] += A_diag_data[j] * A_diag_data[j];
            }
            if (num_cols_offd)
            {
               for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
               {
                  l1_norm[i] += A_offd_data[j] * A_offd_data[j];
               }
            }
         }
      }
      else if (option == 4)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;
            if (cf_marker == NULL)
            {
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if (ii == i || ii < ns || ii >= ne)
                  {
                     if (ii == i)
                     {
                        diag = hypre_abs(A_diag_data[j]);
                        l1_norm[i] += hypre_abs(A_diag_data[j]);
                     }
                     else
                     {
                        l1_norm[i] += 0.5 * hypre_abs(A_diag_data[j]);
                     }
                  }
               }

               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += 0.5 * hypre_abs(A_offd_data[j]);
                  }
               }
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if ((ii == i || ii < ns || ii >= ne) &&
                      (cf_diag == cf_marker[A_diag_J[j]]))
                  {
                     if (ii == i)
                     {
                        diag = hypre_abs(A_diag_data[j]);
                        l1_norm[i] += hypre_abs(A_diag_data[j]);
                     }
                     else
                     {
                        l1_norm[i] += 0.5 * hypre_abs(A_diag_data[j]);
                     }
                  }
               }

               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += 0.5 * hypre_abs(A_offd_data[j]);
                     }
                  }
               }
            }

            /* Truncate according to Remark 6.2 */
            if (l1_norm[i] <= 4.0 / 3.0 * diag)
            {
               l1_norm[i] = diag;
            }
         }
      }
      else if (option == 5) /*stores diagonal of A for Jacobi using matvec, rlx 7 */
      {
         /* Set the diag element */
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] =  A_diag_data[A_diag_I[i]];
            if (l1_norm[i] == 0) { l1_norm[i] = 1.0; }
         }
      }
      else if (option == 6)
      {
         for (i = ns; i < ne; i++)
         {
            l1_norm[i] = 0.0;

            if (cf_marker == NULL)
            {
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if (ii == i || ii < ns || ii >= ne)
                  {
                     if (ii == i)
                     {
                        diag = hypre_abs(A_diag_data[j]);
                     }
                     else
                     {
                        l1_norm[i] += 0.5 * hypre_abs(A_diag_data[j]);
                     }
                  }
               }
               /* Add the l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     l1_norm[i] += 0.5 * hypre_abs(A_offd_data[j]);
                  }
               }

               l1_norm[i] = (diag + l1_norm[i] + hypre_sqrt(diag * diag + l1_norm[i] * l1_norm[i])) * 0.5;
            }
            else
            {
               cf_diag = cf_marker[i];
               /* Add the diagonal and the local off-thread part of the ith row */
               for (j = A_diag_I[i]; j < A_diag_I[i + 1]; j++)
               {
                  ii = A_diag_J[j];
                  if ((ii == i || ii < ns || ii >= ne) &&
                      (cf_diag == cf_marker[A_diag_J[j]]))
                  {
                     if (ii == i)
                     {
                        diag = hypre_abs(A_diag_data[j]);
                     }
                     else
                     {
                        l1_norm[i] += 0.5 * hypre_abs(A_diag_data[j]);
                     }
                  }
               }
               /* Add the CF l1 norm of the offd part of the ith row */
               if (num_cols_offd)
               {
                  for (j = A_offd_I[i]; j < A_offd_I[i + 1]; j++)
                  {
                     if (cf_diag == cf_marker_offd[A_offd_J[j]])
                     {
                        l1_norm[i] += 0.5 * hypre_abs(A_offd_data[j]);
                     }
                  }
               }

               l1_norm[i] = (diag + l1_norm[i] + hypre_sqrt(diag * diag + l1_norm[i] * l1_norm[i])) * 0.5;
            }
         }
      }

      if (option < 5)
      {
         /* Handle negative definite matrices */
         for (i = ns; i < ne; i++)
            if (A_diag_data[A_diag_I[i]] < 0)
            {
               l1_norm[i] = -l1_norm[i];
            }

         for (i = ns; i < ne; i++)
            /* if (hypre_abs(l1_norm[i]) < DBL_EPSILON) */
            if (hypre_abs(l1_norm[i]) == 0.0)
            {
               hypre_error_in_arg(1);
               break;
            }
      }

   }

   hypre_TFree(cf_marker_offd, HYPRE_MEMORY_HOST);

   *l1_norm_ptr = l1_norm;

   return hypre_error_flag;
}
