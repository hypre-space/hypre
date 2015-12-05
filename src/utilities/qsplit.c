/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.2 $
 ***********************************************************************EHEADER*/

 
#include "utilities.h"
#include <math.h>
 
/*--------------------------------------------------------------------------
 * hypre_DoubleQuickSplit
 * C version of the routine "qsplit" from SPARSKIT
 * Uses a quicksort-type algorithm to split data into 
 * highest "NumberCut" values without completely sorting them.
 * Data is double precision data.
 *--------------------------------------------------------------------------*/
 
int hypre_DoubleQuickSplit(double *values, int *indices, 
                           int list_length, int NumberKept )
{
   int ierr = 0;
   double interchange_value;
   double abskey;
   int interchange_index;
   int first, last;
   int mid, j;
   int done;

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

