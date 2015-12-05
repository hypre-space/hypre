/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
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
