#include "HYPRE_config.h"
#ifndef HYPRE_SEQUENTIAL
#define HYPRE_SEQUENTIAL
#endif
#include "_hypre_utilities.h"
#include "_hypre_blas.h"

#if defined(HYPRE_USING_CUDA)

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

#endif

