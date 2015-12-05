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

/*--------------------------------------------------------------------------
 * hypre_LowerBinarySearch
 * integers such that
 *      list[m-1] < value <= list[m].
 * The routine returns location m or -1.
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_LowerBinarySearch(HYPRE_Int *list, HYPRE_Int value, HYPRE_Int list_length)
{
   HYPRE_Int low, high, m;
   HYPRE_Int not_found = 1;

   /* special case, list is size zero. */
   if (list_length < 1)
   {
      return -1;
   }

   /* special case, list[0] >= value */
   if (list[0] >= value)
   {
      return 0;
   }

   low = 0;
   high= list_length-1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (m < 1)
      {
         m= 1;
      }

      if (list[m-1] < value && list[m] < value)
      {
         low= m + 1;
      }
      else if (value <= list[m-1] && value <= list[m])
      {
         high= m - 1;
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
 * hypre_UpperBinarySearch
 * integers such that
 *      list[m] <= value < list[m+1].
 * The routine returns location m or -1.
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_UpperBinarySearch(HYPRE_Int *list, HYPRE_Int value, HYPRE_Int list_length)
{
   HYPRE_Int low, high, m;
   HYPRE_Int not_found = 1;

   /* special case, list is size zero. */
   if (list_length < 1)
   {
      return -1;
   }

   /* special case, list[list_length-1] >= value */
   if (list[list_length-1] <= value)
   {
      return (list_length-1);
   }

   low = 0;
   high= list_length-1;
   while (not_found && low <= high)
   {
      m = (low + high) / 2;
      if (list[m] <= value && list[m+1] <= value)
      {
         low= m + 1;
      }
      else if (value < list[m] && value < list[m+1])
      {
         high= m - 1;
      }
      else
      {
        not_found = 0;
        return m;
      }
   }

   return -1;
}
