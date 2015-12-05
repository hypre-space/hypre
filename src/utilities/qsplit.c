/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/


 
#include "_hypre_utilities.h"
#include <math.h>
 
/*--------------------------------------------------------------------------
 * hypre_DoubleQuickSplit
 * C version of the routine "qsplit" from SPARSKIT
 * Uses a quicksort-type algorithm to split data into 
 * highest "NumberCut" values without completely sorting them.
 * Data is double precision data.
 *--------------------------------------------------------------------------*/
 
HYPRE_Int hypre_DoubleQuickSplit(double *values, HYPRE_Int *indices, 
                           HYPRE_Int list_length, HYPRE_Int NumberKept )
{
   HYPRE_Int ierr = 0;
   double interchange_value;
   double abskey;
   HYPRE_Int interchange_index;
   HYPRE_Int first, last;
   HYPRE_Int mid, j;
   HYPRE_Int done;

   first = 0;
   last = list_length-1;

   if ( (NumberKept < first+1) || (NumberKept > last+1) )
      return( ierr );

   /* Loop until the "midpoint" is NumberKept */
   done = 0;

   for ( ; !done; )
   {
      mid = first;
      abskey = fabs( values[ mid ]);

      for( j = first+1; j <= last; j ++)
      {
         if( fabs( values[ j ]) > abskey )
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

      if ( mid+1 == NumberKept )
      {
         done = 1;
         break;
      }
      if ( mid+1 > NumberKept )
         last = mid - 1;
      else
         first = mid + 1;
   }

   return ( ierr );
}

