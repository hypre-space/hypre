/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_handle utility functions
 *
 *****************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

/* GPU SpTrans */
HYPRE_Int
hypre_SetSpTransUseVendor( HYPRE_Int use_vendor )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpTransUseVendor(hypre_handle()) = use_vendor;
#else
   HYPRE_UNUSED_VAR(use_vendor);
#endif

   return hypre_error_flag;
}

/* GPU SpMV */
HYPRE_Int
hypre_SetSpMVUseVendor( HYPRE_Int use_vendor )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpMVUseVendor(hypre_handle()) = use_vendor;
#else
   HYPRE_UNUSED_VAR(use_vendor);
#endif

   return hypre_error_flag;
}

/* GPU SpGemm */
HYPRE_Int
hypre_SetSpGemmUseVendor( HYPRE_Int use_vendor )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpgemmUseVendor(hypre_handle()) = use_vendor;
#else
   HYPRE_UNUSED_VAR(use_vendor);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmAlgorithm( HYPRE_Int value )
{
#if defined(HYPRE_USING_GPU)
   if (value >= 1 && value <= 3)
   {
      hypre_HandleSpgemmAlgorithm(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#else
   HYPRE_UNUSED_VAR(value);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmBinned( HYPRE_Int value )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpgemmBinned(hypre_handle()) = value;
#else
   HYPRE_UNUSED_VAR(value);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmRownnzEstimateMethod( HYPRE_Int value )
{
#if defined(HYPRE_USING_GPU)
   if (value >= 1 && value <= 3)
   {
      hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#else
   HYPRE_UNUSED_VAR(value);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmRownnzEstimateNSamples( HYPRE_Int value )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle()) = value;
#else
   HYPRE_UNUSED_VAR(value);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmRownnzEstimateMultFactor( HYPRE_Real value )
{
#if defined(HYPRE_USING_GPU)
   if (value > 0.0)
   {
      hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#else
   HYPRE_UNUSED_VAR(value);
#endif

   return hypre_error_flag;
}

/* GPU Rand */
HYPRE_Int
hypre_SetUseGpuRand( HYPRE_Int use_gpurand )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleUseGpuRand(hypre_handle()) = use_gpurand;
#else
   HYPRE_UNUSED_VAR(use_gpurand);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetGaussSeidelMethod( HYPRE_Int gs_method )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleDeviceGSMethod(hypre_handle()) = gs_method;
#else
   HYPRE_UNUSED_VAR(gs_method);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetUserDeviceMalloc(GPUMallocFunc func)
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleUserDeviceMalloc(hypre_handle()) = func;
#else
   HYPRE_UNUSED_VAR(func);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetUserDeviceMfree(GPUMfreeFunc func)
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleUserDeviceMfree(hypre_handle()) = func;
#else
   HYPRE_UNUSED_VAR(func);
#endif

   return hypre_error_flag;
}

HYPRE_Int
hypre_SetGpuAwareMPI( HYPRE_Int use_gpu_aware_mpi )
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_HandleUseGpuAwareMPI(hypre_handle()) = use_gpu_aware_mpi;
#else
   HYPRE_UNUSED_VAR(use_gpu_aware_mpi);
#endif
   return hypre_error_flag;
}

HYPRE_Int
hypre_GetGpuAwareMPI(void)
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   return hypre_HandleUseGpuAwareMPI(hypre_handle());
#else
   return 0;
#endif
}
