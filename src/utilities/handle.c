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

HYPRE_Int
hypre_SetSpGemmUseCusparse( HYPRE_Int use_cusparse )
{
#if defined(HYPRE_USING_GPU)
   hypre_HandleSpgemmUseCusparse(hypre_handle()) = use_cusparse;
#endif
   return hypre_error_flag;
}

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

