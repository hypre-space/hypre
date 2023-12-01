/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"
#include "_hypre_onedpl.hpp"

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * hypre_IntArraySetConstantValuesDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySetConstantValuesDevice( hypre_IntArray *v,
                                       HYPRE_Int       value )
{
   HYPRE_Int *array_data = hypre_IntArrayData(v);
   HYPRE_Int  size       = hypre_IntArraySize(v);

#if defined(HYPRE_USING_GPU)
   hypreDevice_IntFilln( array_data, size, value );

   hypre_SyncComputeStream(hypre_handle());

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(array_data)
   for (i = 0; i < size; i++)
   {
      array_data[i] = value;
   }
#endif

   return hypre_error_flag;
}

#if defined(HYPRE_USING_GPU)
/*--------------------------------------------------------------------------
 * hypreGPUKernel_IntArrayInverseMapping
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_IntArrayInverseMapping( hypre_DeviceItem  &item,
                                       HYPRE_Int          size,
                                       HYPRE_Int         *v_data,
                                       HYPRE_Int         *w_data )
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

   if (i < size)
   {
      w_data[v_data[i]] = i;
   }
}
#endif

/*--------------------------------------------------------------------------
 * hypre_IntArrayInverseMappingDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayInverseMappingDevice( hypre_IntArray  *v,
                                    hypre_IntArray  *w )
{
   HYPRE_Int   size    = hypre_IntArraySize(v);
   HYPRE_Int  *v_data  = hypre_IntArrayData(v);
   HYPRE_Int  *w_data  = hypre_IntArrayData(w);

#if defined(HYPRE_USING_GPU)
   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(size, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_IntArrayInverseMapping, gDim, bDim, size, v_data, w_data );

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   #pragma omp target teams distribute parallel for private(i) is_device_ptr(v_data, w_data)
   for (i = 0; i < size; i++)
   {
      w_data[v_data[i]] = i;
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayCountDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayCountDevice( hypre_IntArray *v,
                           HYPRE_Int       value,
                           HYPRE_Int      *num_values_ptr )
{
   HYPRE_Int  *array_data  = hypre_IntArrayData(v);
   HYPRE_Int   size        = hypre_IntArraySize(v);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   *num_values_ptr = HYPRE_THRUST_CALL( count,
                                        array_data,
                                        array_data + size,
                                        value );

#elif defined(HYPRE_USING_SYCL)
   *num_values_ptr = HYPRE_ONEDPL_CALL( std::count,
                                        array_data,
                                        array_data + size,
                                        value );

#elif defined (HYPRE_USING_DEVICE_OPENMP)
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Function not implemented for Device OpenMP");
   *num_values_ptr = 0;
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArrayNegateDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArrayNegateDevice( hypre_IntArray *v )
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_THRUST_CALL( transform,
                      hypre_IntArrayData(v),
                      hypre_IntArrayData(v) + hypre_IntArraySize(v),
                      hypre_IntArrayData(v),
                      thrust::negate<HYPRE_Int>() );
#elif defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      hypre_IntArrayData(v),
                      hypre_IntArrayData(v) + hypre_IntArraySize(v),
                      hypre_IntArrayData(v),
                      std::negate<HYPRE_Int>() );
#else
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!");
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_IntArraySetInterleavedValuesDevice
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
struct hypre_cycle_functor
{
   HYPRE_Int cycle_length;

   hypre_cycle_functor(HYPRE_Int _cycle_length) : cycle_length(_cycle_length) {}

   __host__ __device__ HYPRE_Int operator()(HYPRE_Int x) const
   {
      return x % cycle_length;
   }
};
#endif

HYPRE_Int
hypre_IntArraySetInterleavedValuesDevice( hypre_IntArray *v,
                                          HYPRE_Int       cycle )
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   HYPRE_THRUST_CALL( sequence,
                      hypre_IntArrayData(v),
                      hypre_IntArrayData(v) + hypre_IntArraySize(v));

   HYPRE_THRUST_CALL( transform,
                      hypre_IntArrayData(v),
                      hypre_IntArrayData(v) + hypre_IntArraySize(v),
                      hypre_IntArrayData(v),
                      hypre_cycle_functor(cycle) );

#else
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not implemented yet!");
#endif

   return hypre_error_flag;
}

#endif
