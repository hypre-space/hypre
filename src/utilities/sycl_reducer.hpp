/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/* SYCL reducer class */

#ifndef HYPRE_SYCL_REDUCER_H
#define HYPRE_SYCL_REDUCER_H

#if defined(HYPRE_USING_SYCL)
#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS)

template<typename T> void OneWorkgroupReduce(T *d_arr, HYPRE_Int N, T *h_out);

struct HYPRE_double4
{
   HYPRE_Real x,y,z,w;

   HYPRE_double4() {}

   HYPRE_double4(HYPRE_Real x1, HYPRE_Real x2, HYPRE_Real x3, HYPRE_Real x4)
   {
      x = x1;
      y = x2;
      z = x3;
      w = x4;
   }

   void operator=(HYPRE_Real val)
   {
      x = y = z = w = val;
   }

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

   HYPRE_double6() {}

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

   void operator=(HYPRE_Real val)
   {
      x = y = z = w = u = v = val;
   }

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
__inline__ __attribute__((always_inline))
HYPRE_Real warpReduceSum(HYPRE_Real val, sycl::nd_item<1>& item)
{
  sycl::ONEAPI::sub_group SG = item.get_sub_group();
  
  for (HYPRE_Int offset = SG.get_local_range().get(0) / 2;
       offset > 0; offset /= 2)
  {
    val += SG.shuffle_down(val, offset);
  }
  return val;
}

__inline__ __attribute__((always_inline))
HYPRE_double4 warpReduceSum(HYPRE_double4 val, sycl::nd_item<1>& item) {
  sycl::ONEAPI::sub_group SG = item.get_sub_group();

  for (HYPRE_Int offset = SG.get_local_range().get(0) / 2;
       offset > 0; offset /= 2)
  {
    val.x += SG.shuffle_down(val.x, offset);
    val.y += SG.shuffle_down(val.y, offset);
    val.z += SG.shuffle_down(val.z, offset);
    val.w += SG.shuffle_down(val.w, offset);
  }
  return val;
}

__inline__ __attribute__((always_inline))
HYPRE_double6 warpReduceSum(HYPRE_double6 val, sycl::nd_item<1>& item) {
  sycl::ONEAPI::sub_group SG = item.get_sub_group();

  for (HYPRE_Int offset = SG.get_local_range().get(0) / 2;
       offset > 0; offset /= 2)
  {
    val.x += SG.shuffle_down(val.x, offset);
    val.y += SG.shuffle_down(val.y, offset);
    val.z += SG.shuffle_down(val.z, offset);
    val.w += SG.shuffle_down(val.w, offset);
    val.u += SG.shuffle_down(val.u, offset);
    val.v += SG.shuffle_down(val.v, offset);
  }
  return val;
}

/* reduction within a block */
template <typename T>
__inline__ __attribute__((always_inline))
T blockReduceSum(T val, sycl::nd_item<1>& item, T* shared)
{
   //HYPRE_Int lane = threadIdx.x % item.get_sub_group().get_local_range().get(0);
   //HYPRE_Int wid  = threadIdx.x / item.get_sub_group().get_local_range().get(0);
   size_t threadIdx_x = item.get_local_id(0);
   size_t warpSize = item.get_sub_group().get_local_range().get(0);

   HYPRE_Int lane = threadIdx_x & (warpSize - 1);
   HYPRE_Int wid = threadIdx_x >> HYPRE_SUBGROUP_BITSHIFT;

   val = warpReduceSum(val, item);       // Each warp performs partial reduction

   if (lane == 0)
   {
      shared[wid] = val;          // Write reduced value to shared memory
   }

   item.barrier(sycl::access::fence_space::local_space); // Wait for all partial reductions

   //read from shared memory only if that warp existed
   if (threadIdx_x < item.get_local_range(0) / warpSize)
   {
     val = shared[lane];
   }
   else
   {
      val = 0.0;
   }

   if (wid == 0)
   {
     val = warpReduceSum(val, item); //Final reduce within first warp
   }

   return val;
}

template<typename T>
void OneWorkgroupReduceKernel(T *arr, HYPRE_Int N, sycl::nd_item<1>& item, T *shared)
{
   size_t threadIdx_x = item.get_local_id(0);

   T sum;
   sum = 0.0;
   if (threadIdx_x < N)
   {
      sum = arr[threadIdx_x];
   }
   sum = blockReduceSum(sum, item, shared);
   if (threadIdx_x == 0)
   {
      arr[0] = sum;
   }
}

/* Reducer class */
template <typename T>
struct ReduceSum
{
   T init;                    /* initial value passed in */
   mutable T __thread_sum;    /* place to hold local sum of a work-item,
                                 and partial sum of a work-group */
   T *d_buf;                  /* place to store partial sum within work-groups
                                 in the 1st round, used in the 2nd round */
   HYPRE_Int num_workgroups;  /* number of workgroups used in the 1st round */

   /* constructor
    * val is the initial value (added to the reduced sum) */
   ReduceSum(T val)
   {
      init = val;
      __thread_sum = 0.0;
      num_workgroups = -1;

      if (hypre_HandleSyclReduceBuffer(hypre_handle()) == nullptr)
      {
         /* allocate for the max size for reducing double6 type */
         hypre_HandleSyclReduceBuffer(hypre_handle()) = hypre_TAlloc(HYPRE_double6, 1024, HYPRE_MEMORY_DEVICE);
      }

      d_buf = (T*) hypre_HandleSyclReduceBuffer(hypre_handle());
   }

   /* copy constructor */
   ReduceSum(const ReduceSum<T>& other)
   {
      *this = other;
   }

  //  /* reduction within blocks */
  // // abb: need to fix this
  //  void BlockReduce() const
  //  {
  //     __thread_sum = blockReduceSum(__thread_sum);
  //     if (threadIdx.x == 0)
  //     {
  //        d_buf[blockIdx.x] = __thread_sum;
  //     }
  //  }


   void operator+=(T val) const
   {
      __thread_sum += val;
   }

   /* invoke the 2nd reduction at the time want the sum from the reducer */
   operator T()
   {
      T val;
      /* 2nd reduction with only *one* block */
      hypre_assert(num_workgroups >= 0 && num_workgroups <= 1024);

      //abb todo: fix this
      const sycl::range<1> gDim(1), bDim(1024);
      hypre_HandleSyclComputeQueue(hypre_handle())->submit([&] (sycl::handler& cgh) {

	  sycl::accessor<T, 1, sycl::access_mode::read_write,
			 sycl::target::local> shared_acc(HYPRE_SUBGROUP_SIZE, cgh);

	  cgh.parallel_for(sycl::nd_range<1>(gDim*bDim, bDim),
			   [=] (sycl::nd_item<1> item) {
			     OneWorkgroupReduceKernel(d_buf, num_workgroups, item, shared_acc.get_pointer());
			   });
	});

      //HYPRE_SYCL_1D_LAUNCH( OneWorkgroupReduceKernel, gDim, bDim, d_buf, num_workgroups );

      hypre_TMemcpy(&val, d_buf, T, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);
      val += init;

      return val;
   }

   /* destructor */
   ~ReduceSum<T>()
   {
   }
};

#endif /* #if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) */
#endif /* #if defined(HYPRE_USING_SYCL) */
#endif /* #ifndef HYPRE_SYCL_REDUCER_H */
