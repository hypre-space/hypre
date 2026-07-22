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

#include "_hypre_struct_mv.h"

#if defined(HYPRE_MIXED_PRECISION)

/******************************************************************************
 *
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Mixed precision hypre_StructVectorCopy -- -- TODO: Needs GPU support - DOK
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructVectorCopy_mp( hypre_StructVector *x,
                           hypre_StructVector *y )
{
   /* Generic pointer type */
   void               *xp, *yp;

   HYPRE_Int          size;

   /* Call standard vector copy if precisions match. */
   if (hypre_StructVectorPrecision (y) == hypre_StructVectorPrecision (x))
   {
      return HYPRE_StructVectorCopy_pre(hypre_StructVectorPrecision (y), (HYPRE_StructVector)x,
                                        (HYPRE_StructVector)y);
   }

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   size = hypre_StructVectorDataSize(x);

   /* Implicit conversion to generic data type (void pointer) */
   xp = hypre_StructVectorData(x);
   yp = hypre_StructVectorData(y);
   /* copy data */
   hypre_RealArrayCopy_mp(hypre_StructVectorPrecision (x), xp, hypre_StructVectorMemoryLocation(y),
                          hypre_StructVectorPrecision (y), yp, hypre_StructVectorMemoryLocation(y), size);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mixed-precision clone of struct vector.
 * New vector resides in the same memory location
 *--------------------------------------------------------------------------*/
/*
hypre_StructVector *
hypre_StructVectorClone_mp( hypre_StructVector *x, HYPRE_Precision new_precision )
{
   hypre_StructVector   *y;

   if (hypre_StructVectorPrecision(A) == new_precision)
   {
      return hypre_ParCSRMatrixClone_pre( hypre_ParCSRMatrixPrecision(A), A, 1 );
   }
}
*/
#endif

