/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_MULTIPRECISION_HEADER
#define hypre_MULTIPRECISION_HEADER

#ifdef __cplusplus
extern "C" {
#endif

#ifdef HYPRE_MIXED_PRECISION

/*--------------------------------------------------------------------------
 * Global variable
 *--------------------------------------------------------------------------*/

extern HYPRE_Precision hypre__global_precision;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

HYPRE_Precision
hypre_GlobalPrecision();

#endif

#ifdef __cplusplus
}
#endif

#endif

