/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header file for HYPRE_parcsr_mv library
 *
 *****************************************************************************/

#ifndef HYPRE_STRUCT_MV_MP_HEADER
#define HYPRE_STRUCT_MV_MP_HEADER

#include "_hypre_struct_mv.h"

#ifdef HYPRE_MIXED_PRECISION

#ifdef __cplusplus
extern "C" {
#endif

HYPRE_Int HYPRE_StructVectorCopy_mp( HYPRE_StructVector x, HYPRE_StructVector y );

//HYPRE_Int HYPRE_StructVectorConvert_mp( HYPRE_StructVector v, HYPRE_Precision new_precision );

#ifdef __cplusplus
}
#endif

#endif

#endif
