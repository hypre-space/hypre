/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/


 
#include "_hypre_utilities.h"
 
/*--------------------------------------------------------------------------
 * hypre_BinarySearch
 * performs a binary search for value on array list where list needs
 * to contain ordered nonnegative numbers
 * the routine returns the location of the value or -1
 *--------------------------------------------------------------------------*/
 
int hypre_BinarySearch(int *list, int value, int list_length)
{
   int low, high, m;
   int not_found = 1;

   low = 0;
   high = list_length-1; 
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (value < list[m])
      {
         high = m - 1;
      }
      else if (value > list[m])
      {
        low = m + 1;
      }
      else
      {
        not_found = 0;
        return m;
      }
   }
   return -1;
}

/*--------------------------------------------------------------------------
 * hypre_BinarySearch2
 * this one is a bit more robust:
 *   avoids overflow of m as can happen above when (low+high) overflows
 *   lets user specifiy high and low bounds for array (so a subset 
     of array can be used)
 *  if not found, then spot returns where is should be inserted

 *--------------------------------------------------------------------------*/
 
int hypre_BinarySearch2(int *list, int value, int low, int high, int *spot) 
{
   
   int m;
   
   while (low <= high) 
   {
      m = low + (high - low)/2;
 
      if (value < list[m])
         high = m - 1;
      else if (value > list[m])
         low = m + 1;
      else
      {
         *spot = m;
         return m;
      }
   }
   /* not found (high = low-1) - so insert at low */
      *spot = low;

   return -1;
}
