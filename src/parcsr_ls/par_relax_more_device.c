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

#if defined(HYPRE_USING_CUDA)
#include "_hypre_utilities.hpp"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/for_each.h>

HYPRE_Int hypre_LINPACKcgtql1(HYPRE_Int*,HYPRE_Real *,HYPRE_Real *,HYPRE_Int *);
HYPRE_Real hypre_LINPACKcgpthy(HYPRE_Real*, HYPRE_Real*);

/**
 * @brief xpay
 *
 * Performs
 * y = x + a*y
 * For vectors x,y, scalar a
 */
template<typename T>
struct xpay
{
   typedef thrust::tuple<T&, T> Tuple;

   const T scale;
   xpay(T _scale): scale(_scale) {}

   __host__ __device__ void operator()(Tuple t) {
	    thrust::get<0>(t) = thrust::get<1>(t) + scale * thrust::get<0>(t);
   }
};

/**
 * @brief xy
 *
 * Performs
 * y = x .* y
 * For vectors x,y
 */
template<typename T>
struct xy
{
   typedef thrust::tuple<T&, T> Tuple;

   __host__ __device__ void operator()(Tuple t) {
	    thrust::get<0>(t) = thrust::get<1>(t) * thrust::get<0>(t);
   }
};

/******************************************************************************
   use CG to get the eigenvalue estimate
  scale means get eig est of  (D^{-1/2} A D^{-1/2}
******************************************************************************/
HYPRE_Int
hypre_ParCSRMaxEigEstimateCGDevice( hypre_ParCSRMatrix *A,     /* matrix to relax with */
                              HYPRE_Int           scale, /* scale by diagonal?*/
                              HYPRE_Int           max_iter,
                              HYPRE_Real         *max_eig,
                              HYPRE_Real         *min_eig )
{
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup");
#endif
   HYPRE_Int i, j, err;
   hypre_ParVector *p;
   hypre_ParVector *s;
   hypre_ParVector *r;
   hypre_ParVector *ds;
   hypre_ParVector *u;

   HYPRE_Real *tridiag = NULL;
   HYPRE_Real *trioffd = NULL;

   HYPRE_Real lambda_max ;
   HYPRE_Real beta, gamma = 0.0, alpha, sdotp, gamma_old, alphainv;
   HYPRE_Real diag;
   HYPRE_Real lambda_min;
   HYPRE_Real *s_data, *p_data, *ds_data, *u_data;
   HYPRE_Int local_size = hypre_CSRMatrixNumRows(hypre_ParCSRMatrixDiag(A));

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);
   HYPRE_Real *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int *A_diag_i = hypre_CSRMatrixI(A_diag);

   /* check the size of A - don't iterate more than the size */
   HYPRE_BigInt size = hypre_ParCSRMatrixGlobalNumRows(A);

   if (size < (HYPRE_BigInt) max_iter)
   {
      max_iter = (HYPRE_Int) size;
   }

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_DataAlloc");
#endif
   /* create some temp vectors: p, s, r , ds, u*/
   r = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(r, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_ParVectorSetPartitioningOwner(r,0);

   p = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(p, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_ParVectorSetPartitioningOwner(p,0);

   s = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(s, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_ParVectorSetPartitioningOwner(s,0);

   /* DS Starts on host to be populated, then transferred to device */
   ds = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                              hypre_ParCSRMatrixGlobalNumRows(A),
                              hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(ds, hypre_ParCSRMatrixMemoryLocation(A));
   ds_data = hypre_VectorData(hypre_ParVectorLocalVector(ds));
   hypre_MemPrefetch(ds_data, sizeof(HYPRE_Real) * local_size, HYPRE_MEMORY_HOST);
   hypre_ParVectorSetPartitioningOwner(ds,0);

   u = hypre_ParVectorCreate(hypre_ParCSRMatrixComm(A),
                             hypre_ParCSRMatrixGlobalNumRows(A),
                             hypre_ParCSRMatrixRowStarts(A));
   hypre_ParVectorInitialize_v2(u, hypre_ParCSRMatrixMemoryLocation(A));
   hypre_ParVectorSetPartitioningOwner(u,0);

   /* point to local data */
   s_data = hypre_VectorData(hypre_ParVectorLocalVector(s));
   p_data = hypre_VectorData(hypre_ParVectorLocalVector(p));
   u_data = hypre_VectorData(hypre_ParVectorLocalVector(u));

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup");
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Setup_Alloc");
#endif

   /* make room for tri-diag matrix */
   tridiag = hypre_CTAlloc(HYPRE_Real, max_iter+1, HYPRE_MEMORY_HOST);
   trioffd = hypre_CTAlloc(HYPRE_Real, max_iter+1, HYPRE_MEMORY_HOST);
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Zeroing");
#endif
   for (i=0; i < max_iter + 1; i++)
   {
      tridiag[i] = 0;
      trioffd[i] = 0;
   }
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Random");
#endif

   /* set residual to random */
   hypre_ParVectorSetRandomValues(r,1);

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Setup_CPUAlloc_Diag");
#endif

   if (scale)
   {
      for (i = 0; i < local_size; i++)
      {
         diag = A_diag_data[A_diag_i[i]];
         ds_data[i] = 1/sqrt(diag);
      }

   }
   else
   {
      /* set ds to 1 */
      hypre_ParVectorSetConstantValues(ds,1.0);
   }

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif

#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPopRange();
#endif
#if defined(HYPRE_USING_CUDA)
   hypre_GpuProfilingPushRange("ParCSRMaxEigEstimate_Iter");
#endif
   hypre_MemPrefetch(ds_data, sizeof(HYPRE_Real) * local_size, hypre_ParCSRMatrixMemoryLocation(A));

   /* gamma = <r,Cr> */
   gamma = hypre_ParVectorInnerProd(r,p);

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
      gamma = hypre_ParVectorInnerProd(r,s);

      if (i==0)
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
         HYPRE_THRUST_CALL(for_each,
         thrust::make_zip_iterator(thrust::make_tuple(p_data, s_data)),
         thrust::make_zip_iterator(thrust::make_tuple(p_data, s_data))+local_size,
         xpay<HYPRE_Real>(beta));

      }

      if (scale)
      {
         /* s = D^{-1/2}A*D^{-1/2}*p */
         HYPRE_THRUST_CALL(for_each,
         thrust::make_zip_iterator(thrust::make_tuple(u_data, ds_data, p_data)),
         thrust::make_zip_iterator(thrust::make_tuple(u_data , ds_data, p_data ))+local_size,
         oop_xy<HYPRE_Real>());

         hypre_ParCSRMatrixMatvec(1.0, A, u, 0.0, s);

         HYPRE_THRUST_CALL(for_each,
         thrust::make_zip_iterator(thrust::make_tuple(s_data, ds_data)),
         thrust::make_zip_iterator(thrust::make_tuple(s_data , ds_data))+local_size,
         xy<HYPRE_Real>());
      }
      else
      {
         /* s = A*p */
         hypre_ParCSRMatrixMatvec(1.0, A, p, 0.0, s);
      }

      /* <s,p> */
      sdotp =  hypre_ParVectorInnerProd(s,p);

      /* alpha = gamma / <s,p> */
      alpha = gamma/sdotp;

      /* get tridiagonal matrix */
      alphainv = 1.0/alpha;

      tridiag[i+1] = alphainv;
      tridiag[i] *= beta;
      tridiag[i] += alphainv;

      trioffd[i+1] = alphainv;
      trioffd[i] *= sqrt(beta);

      /* x = x + alpha*p */
      /* don't need */

      /* r = r - alpha*s */
      hypre_ParVectorAxpy( -alpha, s, r);

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
   hypre_LINPACKcgtql1(&i,tridiag,trioffd,&err);

   lambda_max = tridiag[i-1];
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
