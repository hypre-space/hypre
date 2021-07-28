/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

/* GPU SpGemm */
HYPRE_Int
hypre_SetSpGemmUseCusparse( HYPRE_Int use_cusparse )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpgemmUseCusparse(hypre_handle()) = use_cusparse;
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
#endif
   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmRownnzEstimateNSamples( HYPRE_Int value )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle()) = value;
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
#endif
   return hypre_error_flag;
}

HYPRE_Int
hypre_SetSpGemmHashType( char value )
{
#if defined(HYPRE_USING_GPU)
   if (value == 'L' || value == 'Q' || value == 'D')
   {
      hypre_HandleSpgemmHashType(hypre_handle()) = value;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#endif
   return hypre_error_flag;
}

/* GPU Rand */
HYPRE_Int
hypre_SetUseGpuRand( HYPRE_Int use_gpurand )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleUseGpuRand(hypre_handle()) = use_gpurand;
#endif
   return hypre_error_flag;
}

HYPRE_Int
hypre_SetGaussSeidelMethod( HYPRE_Int gs_method )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleDeviceGSMethod(hypre_handle()) = gs_method;
#endif
   return hypre_error_flag;
}

HYPRE_Int
hypre_SetUserDeviceMalloc(GPUMallocFunc func)
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleUserDeviceMalloc(hypre_handle()) = func;
#endif
   return hypre_error_flag;
}

HYPRE_Int
hypre_SetUserDeviceMfree(GPUMfreeFunc func)
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleUserDeviceMfree(hypre_handle()) = func;
#endif
   return hypre_error_flag;
}
