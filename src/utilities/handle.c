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

/*--------------------------------------------------------------------------
 * Set log level and update temporary log level with previous state
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetLogLevel(HYPRE_Int log_level)
{
   hypre_HandleLogLevelSaved(hypre_handle()) = hypre_HandleLogLevel(hypre_handle());
   hypre_HandleLogLevel(hypre_handle()) = log_level;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Set temporary variable for the log level
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetLogLevelSaved(HYPRE_Int log_level_saved)
{
   hypre_HandleLogLevelSaved(hypre_handle()) = log_level_saved;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Restore log level value from the saved variable
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_RestoreLogLevel(void)
{
   hypre_HandleLogLevel(hypre_handle()) = hypre_HandleLogLevelSaved(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SetSpTransUseVendor
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpMVUseVendor
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpMVAlgorithm
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetSpMVAlgorithm( HYPRE_Int algorithm )
{
#if defined(HYPRE_USING_GPU)
#if defined(HYPRE_USING_ROCSPARSE)
   HYPRE_Int valid = algorithm == (HYPRE_Int) rocsparse_spmv_alg_default ||
                     algorithm == (HYPRE_Int) rocsparse_spmv_alg_csr_adaptive;

#if (ROCSPARSE_VERSION >= 400100)
   valid |= algorithm == (HYPRE_Int) rocsparse_spmv_alg_csr_rowsplit;
#else
   valid |= algorithm == (HYPRE_Int) rocsparse_spmv_alg_csr_stream;
#endif
#if (ROCSPARSE_VERSION >= 300100)
   valid |= algorithm == (HYPRE_Int) rocsparse_spmv_alg_csr_lrb;
#endif
#if (ROCSPARSE_VERSION >= 400100)
   valid |= algorithm == (HYPRE_Int) rocsparse_spmv_alg_csr_nnzsplit;
#endif

#elif defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   HYPRE_Int valid = algorithm == (HYPRE_Int) HYPRE_CUSPARSE_SPMV_ALG_DEFAULT ||
                     algorithm == (HYPRE_Int) HYPRE_CUSPARSE_SPMV_CSR_ALG1 ||
                     algorithm == (HYPRE_Int) HYPRE_CUSPARSE_SPMV_CSR_ALG2;
#else
   HYPRE_Int valid = algorithm == 0;
#endif

   if (valid)
   {
      hypre_HandleSpMVAlgorithm(hypre_handle()) = algorithm;
   }
   else
   {
      hypre_error_in_arg(1);
   }
#else
   HYPRE_UNUSED_VAR(algorithm);
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SetSpGemmUseVendor
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpGemmAlgorithm
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpGemmBinned
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpGemmRownnzEstimateMethod
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpGemmRownnzEstimateNSamples
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetSpGemmRownnzEstimateMultFactor
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetUseGpuRand
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_SetGaussSeidelMethod
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GetDeviceGSMethod(void)
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   return hypre_HandleDeviceGSMethod(hypre_handle());
#else
   return 0;
#endif
}

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetUserDeviceMalloc(GPUMallocFunc func)
{
   hypre_HandleUserDeviceMalloc(hypre_handle()) = func;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SetUserDeviceMfree(GPUMfreeFunc func)
{
   hypre_HandleUserDeviceMfree(hypre_handle()) = func;

   return hypre_error_flag;
}

#endif /* if defined(HYPRE_USING_GPU) */

/*--------------------------------------------------------------------------
 * hypre_SetGpuAwareMPI
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_GetGpuAwareMPI
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GetGpuAwareMPI(void)
{
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   return hypre_HandleUseGpuAwareMPI(hypre_handle());
#else
   return 0;
#endif
}
