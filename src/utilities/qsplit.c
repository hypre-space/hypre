/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include <math.h>

/*--------------------------------------------------------------------------
 * hypre_DoubleQuickSplit
 * C version of the routine "qsplit" from SPARSKIT
 * Uses a quicksort-type algorithm to split data into
 * highest "NumberCut" values without completely sorting them.
 * Data is HYPRE_Real precision data.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_DoubleQuickSplit(HYPRE_Real *values, HYPRE_Int *indices,
                                 HYPRE_Int list_length, HYPRE_Int NumberKept )
{
   HYPRE_Int ierr = 0;
   HYPRE_Real interchange_value;
   HYPRE_Real abskey;
   HYPRE_Int interchange_index;
   HYPRE_Int first, last;
   HYPRE_Int mid, j;
   HYPRE_Int done;

   first = 0;
   last = list_length - 1;

   if ( (NumberKept < first + 1) || (NumberKept > last + 1) )
   {
      return ( ierr );
   }

   /* Loop until the "midpoint" is NumberKept */
   done = 0;

   for ( ; !done; )
   {
      mid = first;
      abskey = hypre_abs( values[ mid ]);

      for ( j = first + 1; j <= last; j ++)
      {
         if ( hypre_abs( values[ j ]) > abskey )
         {
            mid ++;
            /* interchange values */
            interchange_value = values[ mid];
            interchange_index = indices[ mid];
            values[ mid] = values[ j];
            indices[ mid] = indices[ j];
            values[ j] = interchange_value;
            indices[ j] = interchange_index;
         }
      }

      /*  interchange the first and mid value */
      interchange_value = values[ mid];
      interchange_index = indices[ mid];
      values[ mid] = values[ first];
      indices[ mid] = indices[ first];
      values[ first] = interchange_value;
      indices[ first] = interchange_index;

      if ( mid + 1 == NumberKept )
      {
         done = 1;
         break;
      }
      if ( mid + 1 > NumberKept )
      {
         last = mid - 1;
      }
      else
      {
         first = mid + 1;
      }
   }

   return ( ierr );
}

