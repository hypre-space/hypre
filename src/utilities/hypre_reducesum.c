#include "_hypre_utilities.h"

#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && defined(HYPRE_USING_CUDA)

void *cuda_reduce_buffer = NULL;

extern "C++" {

template<typename T>
__global__ void OneBlockReduceKernel(T *arr, HYPRE_Int N)
{
   T sum;
   sum = 0.0;
   if (threadIdx.x < N)
   {
      sum = arr[threadIdx.x];
   }
   sum = blockReduceSum(sum);
   if (threadIdx.x == 0)
   {
      arr[0] = sum;
   }
}

template<typename T>
void OneBlockReduce(T *d_arr, HYPRE_Int N, T *h_out)
{
   hypre_assert(N <= 1024);
   //printf("OneBlockReduce N = %d\n", N);
   //cudaDeviceSynchronize();
#if 1
   OneBlockReduceKernel<<<1, 1024>>>(d_arr, N);
   hypre_TMemcpy(h_out, d_arr, T, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
#else
   T tmp[1024];
   hypre_TMemcpy(tmp, d_arr, T, N, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
   *h_out = 0.0;
   for (HYPRE_Int i=0; i<N; i++)
   {
      *h_out += tmp[i];
   }
#endif
}

template void OneBlockReduce<HYPRE_Real>   (HYPRE_Real    *d_arr, HYPRE_Int N, HYPRE_Real    *h_out);
template void OneBlockReduce<HYPRE_double4>(HYPRE_double4 *d_arr, HYPRE_Int N, HYPRE_double4 *h_out);
template void OneBlockReduce<HYPRE_double6>(HYPRE_double6 *d_arr, HYPRE_Int N, HYPRE_double6 *h_out);

} /* extern "C++" { */

#endif

