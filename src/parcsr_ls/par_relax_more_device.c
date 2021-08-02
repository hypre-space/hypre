/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

#include "_hypre_parcsr_ls.h"
#include "float.h"

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#include "_hypre_utilities.hpp"
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

HYPRE_Int  hypre_LINPACKcgtql1(HYPRE_Int *, HYPRE_Real *, HYPRE_Real *, HYPRE_Int *);
HYPRE_Real hypre_LINPACKcgpthy(HYPRE_Real *, HYPRE_Real *);

/**
 * @brief Calculates row sums and other metrics of a matrix on the device
 * to be used for the MaxEigEstimate
 */
__global__ void
hypreCUDAKernel_CSRMaxEigEstimate(HYPRE_Int      nrows,
                                  HYPRE_Int     *diag_ia,
                                  HYPRE_Int     *diag_ja,
                                  HYPRE_Complex *diag_aa,
                                  HYPRE_Int     *offd_ia,
                                  HYPRE_Int     *offd_ja,
                                  HYPRE_Complex *offd_aa,
                                  HYPRE_Complex *row_sum,
                                  HYPRE_Int      scale,
                                  HYPRE_Int     *pos_diag,
                                  HYPRE_Int     *neg_diag)
{
   HYPRE_Int row_i = hypre_cuda_get_grid_warp_id<1, 1>();

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;

   HYPRE_Real    diag_value = 0.0;
   HYPRE_Complex row_sum_i  = 0.0;

   if (lane < 2)
   {
      p = read_only_load(diag_ia + row_i + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j >= q)
      {
         continue;
      }

      hypre_int find_diag = j < q && diag_ja[j] == row_i;

      HYPRE_Complex aij = diag_aa[j];
      if (find_diag)
      {
         diag_value = aij;
      }

      row_sum_i += fabs(aij);
   }

   if (lane < 2)
   {
      p = read_only_load(offd_ia + row_i + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if (j >= q)
      {
         continue;
      }
      row_sum_i += fabs(offd_aa[j]);
   }

   row_sum_i = warp_reduce_sum(row_sum_i);

   // Get the diagonal value on lane 0
   diag_value = warp_reduce_sum(diag_value);

   if (lane == 0)
   {
      if (diag_value > 0)
      {
         atomicAdd(pos_diag, 1);
      }
      else if (diag_value < 0)
      {
         atomicAdd(neg_diag, 1);
      }

      row_sum[row_i] = (scale && diag_value != 0.0) ? row_sum_i / diag_value : row_sum_i;
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
hypre_ParCSRMaxEigEstimateDevice(hypre_ParCSRMatrix *A, HYPRE_Int scale, HYPRE_Real *max_eig)
{
   HYPRE_Real e_max;
   HYPRE_Real max_norm;
   HYPRE_Int  A_num_rows;

   HYPRE_Real temp;

   HYPRE_Int pos_diag;
   HYPRE_Int neg_diag;

   HYPRE_Real *A_diag_data;
   HYPRE_Real *A_offd_data;
   HYPRE_Int  *A_diag_i;
   HYPRE_Int  *A_offd_i;
   HYPRE_Int  *A_diag_j;
   HYPRE_Int  *A_offd_j;

   HYPRE_Int *pos_diag_d = hypre_CTAlloc(HYPRE_Int, 1, hypre_ParCSRMatrixMemoryLocation(A));
   HYPRE_Int *neg_diag_d = hypre_CTAlloc(HYPRE_Int, 1, hypre_ParCSRMatrixMemoryLocation(A));

   A_num_rows = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   HYPRE_Real *rowsums = hypre_CTAlloc(HYPRE_Real, A_num_rows, hypre_ParCSRMatrixMemoryLocation(A));

   A_diag_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   A_diag_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixDiag(A));
   A_diag_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   A_offd_i    = hypre_CSRMatrixI(hypre_ParCSRMatrixOffd(A));
   A_offd_j    = hypre_CSRMatrixJ(hypre_ParCSRMatrixOffd(A));
   A_offd_data = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(A));

   dim3 bDim, gDim;

   bDim = hypre_GetDefaultCUDABlockDimension();
   gDim = hypre_GetDefaultCUDAGridDimension(A_num_rows, "warp", bDim);
   HYPRE_CUDA_LAUNCH(hypreCUDAKernel_CSRMaxEigEstimate,
                     gDim,
                     bDim,
                     A_num_rows,
                     A_diag_i,
                     A_diag_j,
                     A_diag_data,
                     A_offd_i,
                     A_offd_j,
                     A_offd_data,
                     rowsums,
                     scale,
                     pos_diag_d,
                     neg_diag_d);

   hypre_SyncCudaComputeStream(hypre_handle());
   hypre_TMemcpy(&pos_diag, pos_diag_d, HYPRE_Int, 1, HYPRE_MEMORY_HOST, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_TMemcpy(&neg_diag, neg_diag_d, HYPRE_Int, 1, HYPRE_MEMORY_HOST, hypre_ParCSRMatrixMemoryLocation(A));

   hypre_SyncCudaComputeStream(hypre_handle());
   // Max over rows
   // Probably can do this without the extra memory/pass
   max_norm = HYPRE_THRUST_CALL(reduce, rowsums, rowsums + A_num_rows, (HYPRE_Real)0, thrust::maximum<HYPRE_Real>());

   /* get max across procs */
   hypre_MPI_Allreduce(&max_norm, &temp, 1, HYPRE_MPI_REAL, hypre_MPI_MAX, hypre_ParCSRMatrixComm(A));
   max_norm = temp;

   /* from Charles */
   if (pos_diag == 0 && neg_diag > 0) max_norm = -max_norm;

   /* eig estimates */
   e_max = max_norm;

   /* return */
   *max_eig = e_max;

   hypre_TFree(rowsums, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_TFree(pos_diag_d, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_TFree(neg_diag_d, hypre_ParCSRMatrixMemoryLocation(A));

   return hypre_error_flag;
}
struct print_functor
{
   __host__ __device__
      void operator()(HYPRE_Real val)
      {
         printf("%f\n", val);
      }
};

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
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup");
#endif
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

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_DataAlloc");
#endif
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

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange(); /*Setup Data Alloc*/
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup");
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup_Alloc");
#endif

   /* make room for tri-diag matrix */
   tridiag = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
   trioffd = hypre_CTAlloc(HYPRE_Real, max_iter + 1, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange(); /*SETUP_Alloc*/
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Zeroing");
#endif
   for (i = 0; i < max_iter + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange(); /*Zeroing */
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Random");
#endif

   /* set residual to random */
  hypre_CurandUniform(local_size, r_data, 0, 0, 0, 0);

  hypre_SyncCudaComputeStream(hypre_handle());

  HYPRE_THRUST_CALL(transform,
                    r_data, r_data + local_size, r_data,
                    2.0 * _1 - 1.0);

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange(); /*CPUAlloc_Random*/
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Diag");
#endif

   if (scale)
   {
      hypre_CSRMatrixExtractDiagonal(hypre_ParCSRMatrixDiag(A), ds_data, 3);
   }
   else
   {
      /* set ds to 1 */
      hypre_ParVectorSetConstantValues(ds, 1.0);
   }

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange(); /*Setup_CPUAlloc__Diag */
#endif
#if defined(HYPRE_USING_CUDA) /*CPUAlloc_Setup */
   hypre_GpuProfilingPopRange();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange(); /* Setup */
#endif
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Iter");
#endif

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
         HYPRE_THRUST_CALL(transform,
                           s_data, s_data + local_size, p_data, p_data,
                           _1 + beta * _2);
      }

      if (scale)
      {
         /* s = D^{-1/2}A*D^{-1/2}*p */

         /* u = ds .* p */
         HYPRE_THRUST_CALL( transform, ds_data, ds_data + local_size, p_data, u_data, _1 * _2 );

         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, s);

         /* s = ds .* s */
         HYPRE_THRUST_CALL( transform, ds_data, ds_data + local_size, s_data, s_data, _1 * _2 );
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
      trioffd[i] *= sqrt(beta);

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
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_TriDiagEigenSolve");
#endif

   /* eispack routine - eigenvalues return in tridiag and ordered*/
   hypre_LINPACKcgtql1(&i, tridiag, trioffd, &err);

   lambda_max = tridiag[i - 1];
   lambda_min = tridiag[0];
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif
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
