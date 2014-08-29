/*BHEADER**********************************************************************
 * Copyright (c) 2014,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Structured matrix-matrix multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/* this currently cannot be greater than 7 */
#ifdef MAX_DEPTH
#undef MAX_DEPTH
#endif
#define MAX_DEPTH 7

/*--------------------------------------------------------------------------
 * Multiply nmatrices > 1 matrices, each possibly transposed.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatmult( HYPRE_Int            nmatrices,
                     hypre_StructMatrix **matrices,
                     HYPRE_Int           *transposes,
                     hypre_StructMatrix **C_ptr )
{
   hypre_StructMatrix *C;

   /* Use the StMatrix routines to determine if the operation is allowable and
    * to compute the stencil and stencil formulas for C */

   C = NULL;

   /* Copy A and B into AA and BB (matrices with additional ghost layers) and
    * update their ghost values */

   /* Loop through AA and BB to compute C */

   /* Free AA and BB */

   *C_ptr = C;

   return hypre_error_flag;
}
