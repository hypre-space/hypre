/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
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
 * Utilities for removing duplicates of arrays
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_CompareEntriesIntArrayND
 *--------------------------------------------------------------------------*/
static HYPRE_Int
hypre_CompareEntriesIntArrayND( HYPRE_Int   ndim,
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
 * hypre_UniqueNDimIntArray
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_UniqueIntArrayND( HYPRE_Int    ndim,
                        HYPRE_Int    size,
                        HYPRE_Int  **array )
{
   HYPRE_Int d, i, ii;

   /* Sort n-dimensional array */
   hypre_qsortND(array, ndim, 0, size - 1);

   /* Eliminate duplicates */
   i = 0; ii = 1;
   while (ii < size)
   {
      if (hypre_CompareEntriesIntArrayND(ndim, i, ii, array))
      {
         ii++;
      }
      else
      {
         i++;
         for (d = 0; d < ndim; d++)
         {
            array[d][i] = array[d][ii];
         }
         ii++;
      }
   }

   return hypre_error_flag;
}
