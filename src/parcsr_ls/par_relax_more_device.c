/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * a few more relaxation schemes: Chebychev, FCF-Jacobi, CG  -
 * these do not go through the CF interface (hypre_BoomerAMGRelaxIF)
 *
 *****************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_parcsr_ls.h"

#if defined(HYPRE_USING_GPU)
#include "_hypre_utilities.hpp"

/**
 * @brief Calculates row sums and other metrics of a matrix on the device
 * to be used for the MaxEigEstimate
 */
__global__ void
hypreGPUKernel_CSRMaxEigEstimate(hypre_DeviceItem    &item,
                                 HYPRE_Int      nrows,
                                 HYPRE_Int     *diag_ia,
                                 HYPRE_Int     *diag_ja,
                                 HYPRE_Complex *diag_aa,
                                 HYPRE_Int     *offd_ia,
                                 HYPRE_Int     *offd_ja,
                                 HYPRE_Complex *offd_aa,
                                 HYPRE_Complex *row_sum_lower,
                                 HYPRE_Complex *row_sum_upper,
                                 HYPRE_Int      scale)
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q;

   HYPRE_Complex diag_value = 0.0;
   HYPRE_Complex row_sum_i  = 0.0;
   HYPRE_Complex lower, upper;

   if (lane < 2)
   {
      p = read_only_load(diag_ia + row_i + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Complex aij = read_only_load(&diag_aa[j]);
      if ( read_only_load(&diag_ja[j]) == row_i )
      {
         diag_value = aij;
      }
      else
      {
         row_sum_i += hypre_abs(aij);
      }
   }

   if (lane < 2)
   {
      p = read_only_load(offd_ia + row_i + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      HYPRE_Complex aij = read_only_load(&offd_aa[j]);
      row_sum_i += hypre_abs(aij);
   }

   // Get the row_sum and diagonal value on lane 0
   row_sum_i = warp_reduce_sum(item, row_sum_i);

   diag_value = warp_reduce_sum(item, diag_value);

   if (lane == 0)
   {
      lower = diag_value - row_sum_i;
      upper = diag_value + row_sum_i;

      if (scale)
      {
         lower /= hypre_abs(diag_value);
         upper /= hypre_abs(diag_value);
      }

      row_sum_upper[row_i] = upper;
      row_sum_lower[row_i] = lower;
   }
}

/**
 * @brief Estimates the max eigenvalue using infinity norm on the device
 *
 * @param[in] A Matrix to relax with
 * @param[in] to scale by diagonal
 * @param[out] Maximum eigenvalue
 */
HYPRE_Int
hypre_ParCSRMaxEigEstimateDevice( hypre_ParCSRMatrix *A,
                                  HYPRE_Int           scale,
                                  HYPRE_Real         *max_eig,
                                  HYPRE_Real         *min_eig )
{
   HYPRE_Real e_max;
   HYPRE_Real e_min;
   HYPRE_Int  A_num_rows;


   HYPRE_Real *A_diag_data;
   HYPRE_Real *A_offd_data;
   HYPRE_Int  *A_diag_i;
   HYPRE_Int  *A_offd_i;
   HYPRE_Int  *A_diag_j;
   HYPRE_Int  *A_offd_j;


   A_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   HYPRE_Real *rowsums_lower = hypre_TAlloc(HYPRE_Real, A_num_rows,
                                            hypre_ParCSRMatrixMemoryLocation(A));
   HYPRE_Real *rowsums_upper = hypre_TAlloc(HYPRE_Real, A_num_rows,
                                            hypre_ParCSRMatrixMemoryLocation(A));

   A_diag_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   A_diag_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A));
   A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   A_offd_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
   A_offd_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A));
   A_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(A_num_rows, "warp", bDim);
   HYPRE_GPU_LAUNCH(hypreGPUKernel_CSRMaxEigEstimate,
                    gDim,
                    bDim,
                    A_num_rows,
                    A_diag_i,
                    A_diag_j,
                    A_diag_data,
                    A_offd_i,
                    A_offd_j,
                    A_offd_data,
                    rowsums_lower,
                    rowsums_upper,
                    scale);

   hypre_SyncComputeStream(hypre_handle());

#if defined(HYPRE_USING_SYCL)
   e_min = HYPRE_ONEDPL_CALL(std::reduce, rowsums_lower, rowsums_lower + A_num_rows, (HYPRE_Real)0,
                             oneapi::dpl::minimum<HYPRE_Real>());
   e_max = HYPRE_ONEDPL_CALL(std::reduce, rowsums_upper, rowsums_upper + A_num_rows, (HYPRE_Real)0,
                             oneapi::dpl::maximum<HYPRE_Real>());
#else
   e_min = HYPRE_THRUST_CALL(reduce, rowsums_lower, rowsums_lower + A_num_rows, (HYPRE_Real)0,
                             thrust::minimum<HYPRE_Real>());
   e_max = HYPRE_THRUST_CALL(reduce, rowsums_upper, rowsums_upper + A_num_rows, (HYPRE_Real)0,
                             thrust::maximum<HYPRE_Real>());
#endif

   /* Same as hypre_ParCSRMaxEigEstimateHost */

   HYPRE_Real send_buf[2];
   HYPRE_Real recv_buf[2];

   send_buf[0] = -e_min;
   send_buf[1] = e_max;

   hypre_MPI_Allreduce(send_buf, recv_buf, 2, HYPRE_MPI_REAL, hypre_MPI_MAX,
                       hypre_ParCSRMatrixComm(A));

   /* return */
   if ( hypre_abs(e_min) > hypre_abs(e_max) )
   {
      *min_eig = e_min;
      *max_eig = hypre_min(0.0, e_max);
   }
   else
   {
      *min_eig = hypre_max(e_min, 0.0);
      *max_eig = e_max;
   }

   hypre_TFree(rowsums_lower, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_TFree(rowsums_upper, hypre_ParCSRMatrixMemoryLocation(A));

   return hypre_error_flag;
}

/**
 *  @brief Uses CG to get the eigenvalue estimate on the device
 *
 *  @param[in] A Matrix to relax with
 *  @param[in] scale Gets the eigenvalue est of D^{-1/2} A D^{-1/2}
 *  @param[in] max_iter Maximum number of CG iterations
 *  @param[out] max_eig Estimated max eigenvalue
 *  @param[out] min_eig Estimated min eigenvalue
 */
HYPRE_Int
hypre_ParCSRMaxEigEstimateCGDevice(hypre_ParCSRMatrix *A,     /* matrix to relax with */
                                   HYPRE_Int           scale, /* scale by diagonal?*/
                                   HYPRE_Int           max_iter,
                                   HYPRE_Real         *max_eig,
                                   HYPRE_Real         *min_eig)
{
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup");
   HYPRE_Int        i, err;
   hypre_ParVector *p;
   hypre_ParVector *s;
   hypre_ParVector *r;
   hypre_ParVector *ds;
   hypre_ParVector *u;

   HYPRE_Real *tridiag = NULL;
   HYPRE_Real *trioffd = NULL;

   HYPRE_Real  lambda_max;
   HYPRE_Real  beta, gamma = 0.0, alpha, sdotp, gamma_old, alphainv;
   HYPRE_Real  lambda_min;
   HYPRE_Real *s_data, *p_data, *ds_data, *u_data, *r_data;
   HYPRE_Int   local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   /* check the size of A - don't iterate more than the size */
   HYPRE_BigInt size = hypre_ParCSRMatrixGlobalNumRows(A);

   if (size < (HYPRE_BigInt)max_iter)
   {
      max_iter = (HYPRE_Int)size;
   }

   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_DataAlloc");
   /* create some temp vectors: p, s, r , ds, u*/
   r = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(r, hypre_ParCSRMatrixMemoryLocation(A));

   p = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(p, hypre_ParCSRMatrixMemoryLocation(A));

   s = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(s, hypre_ParCSRMatrixMemoryLocation(A));

   /* DS Starts on host to be populated, then transferred to device */
   ds = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                              hypre_ParCSRMatrixGlobalNumRows(A),
                              hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(ds, hypre_ParCSRMatrixMemoryLocation(A));
   ds_data = hypre_VectorData(hypre_ParVectorLocalVector(ds));

   u = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(u, hypre_ParCSRMatrixMemoryLocation(A));

   /* point to local data */
   s_data = hypre_VectorData(hypre_ParVectorLocalVector(s));
   p_data = hypre_VectorData(hypre_ParVectorLocalVector(p));
   u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));
   r_data = hypre_VectorData(hypre_ParVectorLocalVector(r));

   hypre_GpuProfilingPopRange(); /*Setup Data Alloc*/

   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup");

   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup_Alloc");

   /* make room for tri-diag matrix */
   tridiag = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   trioffd = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   hypre_GpuProfilingPopRange(); /*SETUP_Alloc*/

   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Zeroing");
   for (i = 0; i < max_iter + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }
   hypre_GpuProfilingPopRange(); /*Zeroing */

   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Random");

   /* set residual to random */
   hypre_CurandUniform(local_size, r_data, 0, 0, 0, 0);

   hypre_SyncComputeStream(hypre_handle());

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::transform,
                     r_data, r_data + local_size, r_data,
   [] (auto x) { return 2.0 * x - 1.0; } );
#else
   HYPRE_THRUST_CALL(transform,
                     r_data, r_data + local_size, r_data,
                     2.0 * _1 - 1.0);
#endif

   hypre_GpuProfilingPopRange(); /*CPUAlloc_Random*/

   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Diag");

   if (scale)
   {
      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A), ds_data, 4);
   }
   else
   {
      /* set ds to 1 */
      hypre_ParVectorSetConstantValues(ds, 1.0);
   }

   hypre_GpuProfilingPopRange(); /*Setup_CPUAlloc__Diag */
   hypre_GpuProfilingPopRange(); /*CPUAlloc_Setup */
   hypre_GpuProfilingPopRange(); /* Setup */
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Iter");

   /* gamma = <r,Cr> */
   gamma = hypre_ParVectorInnerProd(r, p);

   /* for the initial filling of the tridiag matrix */
   beta = 1.0;

   i = 0;
   while (i < max_iter)
   {
      /* s = C*r */
      /* TO DO:  C = diag scale */
      hypre_ParVectorCopy(r, s);

      /*gamma = <r,Cr> */
      gamma_old = gamma;
      gamma     = hypre_ParVectorInnerProd(r, s);

      if (gamma < HYPRE_REAL_EPSILON)
      {
         break;
      }

      if (i == 0)
      {
         beta = 1.0;
         /* p_0 = C*r */
         hypre_ParVectorCopy(s, p);
      }
      else
      {
         /* beta = gamma / gamma_old */
         beta = gamma / gamma_old;

         /* p = s + beta p */
         hypreDevice_ComplexAxpyn(p_data, local_size, s_data, p_data, beta);
      }

      if (scale)
      {
         /* s = D^{-1/2}A*D^{-1/2}*p */

         /* u = ds .* p */
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::transform, ds_data, ds_data + local_size, p_data, u_data,
         [] (auto x, auto y) { return x * y; } );
#else
         HYPRE_THRUST_CALL( transform, ds_data, ds_data + local_size, p_data, u_data, _1 * _2 );
#endif

         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, s);

         /* s = ds .* s */
#if defined(HYPRE_USING_SYCL)
         HYPRE_ONEDPL_CALL( std::transform, ds_data, ds_data + local_size, s_data, s_data,
         [] (auto x, auto y) { return x * y; } );
#else
         HYPRE_THRUST_CALL( transform, ds_data, ds_data + local_size, s_data, s_data, _1 * _2 );
#endif
      }
      else
      {
         /* s = A*p */
         hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, s);
      }

      /* <s,p> */
      sdotp = hypre_ParVectorInnerProd(s, p);

      /* alpha = gamma / <s,p> */
      alpha = gamma / sdotp;

      /* get tridiagonal matrix */
      alphainv = 1.0 / alpha;

      tridiag[i + 1] = alphainv;
      tridiag[i] *= beta;
      tridiag[i] += alphainv;

      trioffd[i + 1] = alphainv;
      trioffd[i] *= hypre_sqrt(beta);

      /* x = x + alpha*p */
      /* don't need */

      /* r = r - alpha*s */
      hypre_ParVectorAxpy(-alpha, s, r);

      i++;
   }

   /* GPU NOTE:
    * There is a CUDA whitepaper on calculating the eigenvalues of a symmetric
    * tridiagonal matrix via bisection
    * https://docs.nvidia.com/cuda/samples/6_Advanced/eigenvalues/doc/eigenvalues.pdf
    * As well as code in their sample code
    * https://docs.nvidia.com/cuda/cuda-samples/index.html#eigenvalues
    * They claim that all code is available under a permissive license
    * https://developer.nvidia.com/cuda-code-samples
    * I believe the applicable license is available at
    * https://docs.nvidia.com/cuda/eula/index.html#license-driver
    * but I am not certain, nor do I have the legal knowledge to know if the
    * license is compatible with that which HYPRE is released under.
    */
   hypre_GpuProfilingPopRange();
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_TriDiagEigenSolve");

   /* eispack routine - eigenvalues return in tridiag and ordered*/
   hypre_LINPACKcgtql1(&i, tridiag, trioffd, &err);

   lambda_max = tridiag[i - 1];
   lambda_min = tridiag[0];
   hypre_GpuProfilingPopRange();
   /* hypre_printf("linpack max eig est = %g\n", lambda_max);*/
   /* hypre_printf("linpack min eig est = %g\n", lambda_min);*/

   hypre_TFree(tridiag, HYPRE_MEMORY_HOST);
   hypre_TFree(trioffd, HYPRE_MEMORY_HOST);

   hypre_ParVectorDestroy(r);
   hypre_ParVectorDestroy(s);
   hypre_ParVectorDestroy(p);
   hypre_ParVectorDestroy(ds);
   hypre_ParVectorDestroy(u);

   /* return */
   *max_eig = lambda_max;
   *min_eig = lambda_min;

   return hypre_error_flag;
}

#endif
