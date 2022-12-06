/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_onedpl.hpp"
#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_IntArraySetConstantValuesDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_IntArraySetConstantValuesDevice( hypre_IntArray *v,
                                       HYPRE_Int       value )
{
   HYPRE_Int *array_data = hypre_IntArrayData(v);
   HYPRE_Int  size       = hypre_IntArraySize(v);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP) || defined(HYPRE_USING_SYCL)
   hypreDevice_IntFilln( array_data, size, value );

#elif defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;
   #pragma omp target teams distribute parallel for private(i) is_device_ptr(array_data)
   for (i = 0; i < size; i++)
   {
      array_data[i] = value;
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

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

#endif
