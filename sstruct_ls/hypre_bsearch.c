/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
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



/*--------------------------------------------------------------------------
 * hypre_LowerBinarySearch
 * integers such that
 *      list[m-1] < value <= list[m].
 * The routine returns location m or -1.
 *--------------------------------------------------------------------------*/
int hypre_LowerBinarySearch(int *list, int value, int list_length)
{
   int low, high, m;
   int not_found = 1;

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
int hypre_UpperBinarySearch(int *list, int value, int list_length)
{
   int low, high, m;
   int not_found = 1;

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
