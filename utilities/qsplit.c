/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
*********************************************************************EHEADER*/
 
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

