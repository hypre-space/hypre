/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Utilities for removing duplicates of arrays
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_EntriesEqualIntArrayND
 *--------------------------------------------------------------------------*/
static inline HYPRE_Int
hypre_EntriesEqualIntArrayND( HYPRE_Int   ndim,
                              HYPRE_Int   posA,
                              HYPRE_Int   posB,
                              HYPRE_Int **array )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      if (array[d][posA] != array[d][posB])
      {
         return 0;
      }
   }

   return 1;
}

/*--------------------------------------------------------------------------
 * hypre_CopyEntriesIntArrayND
 *--------------------------------------------------------------------------*/
static inline void
hypre_CopyEntriesIntArrayND( HYPRE_Int   ndim,
                             HYPRE_Int   posA,
                             HYPRE_Int   posB,
                             HYPRE_Int **array )
{
   HYPRE_Int d;

   for (d = 0; d < ndim; d++)
   {
      array[d][posA] = array[d][posB];
   }
}

/*--------------------------------------------------------------------------
 * hypre_UniqueIntArrayND
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UniqueIntArrayND( HYPRE_Int    ndim,
                        HYPRE_Int   *size,
                        HYPRE_Int  **array )
{
   HYPRE_Int i, ii;

   /* Trivial case */
   if (!size || *size < 1)
   {
      return hypre_error_flag;
   }

   /* Sort n-dimensional array */
   hypre_qsortND(array, ndim, 0, *size - 1);

   /* Eliminate duplicates */
   i = 0; ii = 1;
   while (ii < *size)
   {
      if (hypre_EntriesEqualIntArrayND(ndim, i, ii, array))
      {
         ii++;
      }
      else
      {
         i++;
         hypre_CopyEntriesIntArrayND(ndim, i, ii, array);
         ii++;
      }
   }

   *size = i + 1;

   return hypre_error_flag;
}
