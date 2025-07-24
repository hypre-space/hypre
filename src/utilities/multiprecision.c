/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#ifdef HYPRE_MIXED_PRECISION

/* Global variable for default runtime precision */
/* No guard is needed since this file is only compiled once */
HYPRE_Precision hypre__global_precision = HYPRE_REAL_DOUBLE;

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Precision
hypre_GlobalPrecision()
{
   return hypre__global_precision;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetGlobalPrecision(HYPRE_Precision precision)
{
   hypre__global_precision = precision;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GetGlobalPrecision(HYPRE_Precision *precision)
{
   *precision = hypre_GlobalPrecision();

   return hypre_error_flag;
}

#else

/*--------------------------------------------------------------------------
 * non-multiprecision case
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SetGlobalPrecision(HYPRE_Precision precision)
{
   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GetGlobalPrecision(HYPRE_Precision *precision)
{
   *precision = HYPRE_OBJECT_PRECISION;

   return hypre_error_flag;
}

#endif

