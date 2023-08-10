/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre seq_mv mixed-precision interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"

#ifdef HYPRE_MIXED_PRECISION

/******************************************************************************
 *
 * Member functions for hypre_ParVector class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Mixed-precision hypre_ParVectorCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorCopy_mp( hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   return hypre_SeqVectorCopy_mp(x_local, y_local);
}

/*--------------------------------------------------------------------------
 * Mixed-Precision hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParVectorAxpy_mp( HYPRE_Complex    alpha,
                     hypre_ParVector *x,
                     hypre_ParVector *y )
{
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
           
   return hypre_SeqVectorAxpy_mp( alpha, x_local, y_local);
}

#endif