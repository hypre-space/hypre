#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && defined(HYPRE_USING_CUDA)

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 9000)
#define WARP_SHFL_DOWN(mask, var, delta)  __shfl_down_sync(mask, var, delta)
#elif (CUDART_VERSION <= 8000)
#define WARP_SHFL_DOWN(mask, var, delta)  __shfl_down(var, delta);
#endif

extern "C++" {

extern void *cuda_reduce_buffer;

template<typename T> void OneBlockReduce(T *d_arr, HYPRE_Int N, T *h_out);

struct HYPRE_double4
{
   HYPRE_Real x,y,z,w;

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
   HYPRE_Real x,y,z,w,u,v;
 
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

__inline__ __host__ __device__
HYPRE_Real warpReduceSum(HYPRE_Real val)
{
#ifdef __CUDA_ARCH__
  for (HYPRE_Int offset = warpSize/2; offset > 0; offset /= 2)
  {
    val += WARP_SHFL_DOWN(0xFFFFFFFF, val, offset);
  }
#endif
  return val;
}

__inline__ __host__ __device__
HYPRE_double4 warpReduceSum(HYPRE_double4 val) {
#ifdef __CUDA_ARCH__
  for (HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
  {
    val.x += WARP_SHFL_DOWN(0xFFFFFFFF, val.x, offset);
    val.y += WARP_SHFL_DOWN(0xFFFFFFFF, val.y, offset);
    val.z += WARP_SHFL_DOWN(0xFFFFFFFF, val.z, offset);
    val.w += WARP_SHFL_DOWN(0xFFFFFFFF, val.w, offset);
  }
#endif
  return val;
}

__inline__ __host__ __device__
HYPRE_double6 warpReduceSum(HYPRE_double6 val) {
#ifdef __CUDA_ARCH__
  for (HYPRE_Int offset = warpSize / 2; offset > 0; offset /= 2)
  {
    val.x += WARP_SHFL_DOWN(0xFFFFFFFF, val.x, offset);
    val.y += WARP_SHFL_DOWN(0xFFFFFFFF, val.y, offset);
    val.z += WARP_SHFL_DOWN(0xFFFFFFFF, val.z, offset);
    val.w += WARP_SHFL_DOWN(0xFFFFFFFF, val.w, offset);
    val.u += WARP_SHFL_DOWN(0xFFFFFFFF, val.u, offset);
    val.v += WARP_SHFL_DOWN(0xFFFFFFFF, val.v, offset);
  }
#endif
  return val;
}

template <typename T>
__inline__ __host__ __device__
T blockReduceSum(T val) 
{
#ifdef __CUDA_ARCH__
   //static __shared__ T shared[32]; // Shared mem for 32 partial sums
   __shared__ T shared[32];        // Shared mem for 32 partial sums
   HYPRE_Int lane = threadIdx.x % warpSize;
   HYPRE_Int wid  = threadIdx.x / warpSize;

   val = warpReduceSum(val);       // Each warp performs partial reduction

   if (lane == 0)
   {
      shared[wid] = val;          // Write reduced value to shared memory
   }

   __syncthreads();               // Wait for all partial reductions

   //read from shared memory only if that warp existed
   if (threadIdx.x < blockDim.x / warpSize)
   {
      val = shared[lane];
   }
   else
   {
      val = 0.0;
   }

   if (wid == 0)
   {
      val = warpReduceSum(val); //Final reduce within first warp
   }

#endif
   return val;
}

/* Reducer class */
template <typename T>
struct ReduceSum
{
   T init;                    /* initial value passed in */
   mutable T __thread_sum;    /* place to hold local sum of a thread,
                                 and partial sum of a block */
   T *d_buf;                  /* place to store partial sum of a block */
   HYPRE_Int nblocks;         /* number of blocks used in the first round */

   __host__
   ReduceSum(T val)
   {
      init = val;
      __thread_sum = 0.0;
      d_buf = (T*) cuda_reduce_buffer;
      //d_buf = hypre_CTAlloc(T, 1024, HYPRE_MEMORY_DEVICE);
   }

   __host__ __device__ 
   ReduceSum(const ReduceSum<T>& other)
   {
      *this = other;
   }

   __host__ __device__
   void BlockReduce() const
   {
#ifdef __CUDA_ARCH__
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

   /* we invoke the 2nd reduction at the time we want the sum from the reducer
    * class */
   __host__
   operator T()
   {
      T val;
      /* 2nd reduction with only *one* block */
      OneBlockReduce(d_buf, nblocks, &val);
      val += init;
      //hypre_TFree(d_buf, HYPRE_MEMORY_DEVICE);
      return val;
   }

   __host__ __device__ 
   ~ReduceSum<T>()
   {
   }
};

} // extern "C++"

#endif

