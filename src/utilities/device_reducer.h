/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* CUDA reducer class */

#ifndef HYPRE_CUDA_REDUCER_H
#define HYPRE_CUDA_REDUCER_H

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS)

template<typename T> void OneBlockReduce(T *d_arr, HYPRE_Int N, T *h_out);

struct HYPRE_double4
{
   HYPRE_Real x, y, z, w;

   __host__ __device__
   HYPRE_double4() {}

   __host__ __device__
   HYPRE_double4(HYPRE_Real x1, HYPRE_Real x2, HYPRE_Real x3, HYPRE_Real x4)
   {
      x = x1;
      y = x2;
      z = x3;
      w = x4;
   }

   __host__ __device__
   void operator=(HYPRE_Real val)
   {
      x = y = z = w = val;
   }

   __host__ __device__
   void operator+=(HYPRE_double4 rhs)
   {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      w += rhs.w;
   }

};

struct HYPRE_double6
{
   HYPRE_Real x, y, z, w, u, v;

   __host__ __device__
   HYPRE_double6() {}

   __host__ __device__
   HYPRE_double6(HYPRE_Real x1, HYPRE_Real x2, HYPRE_Real x3, HYPRE_Real x4,
                 HYPRE_Real x5, HYPRE_Real x6)
   {
      x = x1;
      y = x2;
      z = x3;
      w = x4;
      u = x5;
      v = x6;
   }

   __host__ __device__
   void operator=(HYPRE_Real val)
   {
      x = y = z = w = u = v = val;
   }

   __host__ __device__
   void operator+=(HYPRE_double6 rhs)
   {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      w += rhs.w;
      u += rhs.u;
      v += rhs.v;
   }

};

/* reduction within a warp */
__inline__ __host__ __device__
HYPRE_Real warpReduceSum(HYPRE_Real val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   for (HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
   {
      val += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val, offset);
   }
#endif
   return val;
}

__inline__ __host__ __device__
HYPRE_double4 warpReduceSum(HYPRE_double4 val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   for (HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
   {
      val.x += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.x, offset);
      val.y += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.y, offset);
      val.z += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.z, offset);
      val.w += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.w, offset);
   }
#endif
   return val;
}

__inline__ __host__ __device__
HYPRE_double6 warpReduceSum(HYPRE_double6 val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   for (HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
   {
      val.x += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.x, offset);
      val.y += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.y, offset);
      val.z += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.z, offset);
      val.w += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.w, offset);
      val.u += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.u, offset);
      val.v += __shfl_down_sync(HYPRE_WARP_FULL_MASK, val.v, offset);
   }
#endif
   return val;
}

/* reduction within a block */
template <typename T>
__inline__ __host__ __device__
T blockReduceSum(T val)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   // Shared mem for HYPRE_WARP_SIZE partial sums
   __shared__ T shared[HYPRE_WARP_SIZE];

   HYPRE_Int lane = threadIdx.x & (warpSize - 1);
   HYPRE_Int wid  = threadIdx.x >> HYPRE_WARP_BITSHIFT;

   // Each warp performs partial reduction
   val = warpReduceSum(val);

   // Write reduced value to shared memory
   if (lane == 0)
   {
      shared[wid] = val;
   }

   // Wait for all partial reductions
   __syncthreads();

   // read from shared memory only if that warp existed
   if (threadIdx.x < (blockDim.x >> HYPRE_WARP_BITSHIFT))
   {
      val = shared[lane];
   }
   else
   {
      val = 0.0;
   }

   // Final reduce within first warp
   if (wid == 0)
   {
      val = warpReduceSum(val);
   }

#endif
   return val;
}

template<typename T>
__global__ void
OneBlockReduceKernel(hypre_DeviceItem &item,
                     T                *arr,
                     HYPRE_Int         N)
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

/* Reducer class */
template <typename T>
struct ReduceSum
{
   using value_type = T;

   T init;                    /* initial value passed in */
   mutable T __thread_sum;    /* place to hold local sum of a thread,
                                 and partial sum of a block */
   T *d_buf;                  /* place to store partial sum within blocks
                                 in the 1st round, used in the 2nd round */
   HYPRE_Int nblocks;         /* number of blocks used in the 1st round */

   /* constructor
    * val is the initial value (added to the reduced sum) */
   __host__
   ReduceSum(T val)
   {
      init = val;
      __thread_sum = 0.0;
      nblocks = -1;
   }

   /* copy constructor */
   __host__ __device__
   ReduceSum(const ReduceSum<T>& other)
   {
      *this = other;
   }

   __host__ void
   Allocate2ndPhaseBuffer()
   {
      if (hypre_HandleReduceBuffer(hypre_handle()) == NULL)
      {
         /* allocate for the max size for reducing double6 type */
         hypre_HandleReduceBuffer(hypre_handle()) =
            hypre_TAlloc(HYPRE_double6, HYPRE_MAX_NTHREADS_BLOCK, HYPRE_MEMORY_DEVICE);
      }

      d_buf = (T*) hypre_HandleReduceBuffer(hypre_handle());
   }

   /* reduction within blocks */
   __host__ __device__
   void BlockReduce() const
   {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      __thread_sum = blockReduceSum(__thread_sum);
      if (threadIdx.x == 0)
      {
         d_buf[blockIdx.x] = __thread_sum;
      }
#endif
   }

   __host__ __device__
   void operator+=(T val) const
   {
      __thread_sum += val;
   }

   /* invoke the 2nd reduction at the time want the sum from the reducer */
   __host__
   operator T()
   {
      T val;

      const HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());
      const HYPRE_ExecutionPolicy exec_policy = hypre_GetExecPolicy1(memory_location);

      if (exec_policy == HYPRE_EXEC_HOST)
      {
         val = __thread_sum;
         val += init;
      }
      else
      {
         /* 2nd reduction with only *one* block */
         hypre_assert(nblocks >= 0 && nblocks <= HYPRE_MAX_NTHREADS_BLOCK);
         const dim3 gDim(1), bDim(HYPRE_MAX_NTHREADS_BLOCK);
         HYPRE_GPU_LAUNCH( OneBlockReduceKernel, gDim, bDim, d_buf, nblocks );
         hypre_TMemcpy(&val, d_buf, T, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
         val += init;
      }

      return val;
   }

   /* destructor */
   __host__ __device__
   ~ReduceSum<T>()
   {
   }
};

#endif /* #if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) */
#endif /* #if defined(HYPRE_USING_CUDA)  || defined(HYPRE_USING_HIP) */
#endif /* #ifndef HYPRE_CUDA_REDUCER_H */
