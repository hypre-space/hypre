/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/


#include <math.h>
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void swap( HYPRE_Int *v,
           HYPRE_Int  i,
           HYPRE_Int  j )
{
   HYPRE_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void swap2(HYPRE_Int     *v,
           double  *w,
           HYPRE_Int      i,
           HYPRE_Int      j )
{
   HYPRE_Int temp;
   double temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap2i(HYPRE_Int  *v,
                  HYPRE_Int  *w,
                  HYPRE_Int  i,
                  HYPRE_Int  j )
{
   HYPRE_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


/* AB 11/04 */

void hypre_swap3i(HYPRE_Int  *v,
                  HYPRE_Int  *w,
                  HYPRE_Int  *z,
                  HYPRE_Int  i,
                  HYPRE_Int  j )
{
   HYPRE_Int temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap3_d(double  *v,
                  HYPRE_Int  *w,
                  HYPRE_Int  *z,
                  HYPRE_Int  i,
                  HYPRE_Int  j )
{
   HYPRE_Int temp;
   double temp_d;
   

   temp_d = v[i];
   v[i] = v[j];
   v[j] = temp_d;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap4_d(double  *v,
                  HYPRE_Int  *w,
                  HYPRE_Int  *z,
                  HYPRE_Int *y, 
                  HYPRE_Int  i,
                  HYPRE_Int  j )
{
   HYPRE_Int temp;
   double temp_d;
   

   temp_d = v[i];
   v[i] = v[j];
   v[j] = temp_d;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
   temp = y[i];
   y[i] = y[j];
   y[j] = temp;

}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap_d( double *v,
                   HYPRE_Int  i,
                   HYPRE_Int  j )
{
   double temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void qsort0( HYPRE_Int *v,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;
   swap( v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         swap(v, ++last, i);
      }
   swap(v, left, last);
   qsort0(v, left, last-1);
   qsort0(v, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void qsort1( HYPRE_Int *v,
	     double *w,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;
   swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         swap2(v, w, ++last, i);
      }
   swap2(v, w, left, last);
   qsort1(v, w, left, last-1);
   qsort1(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_qsort2i( HYPRE_Int *v,
                    HYPRE_Int *w,
                    HYPRE_Int  left,
                    HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap2i( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap2i(v, w, ++last, i);
      }
   }
   hypre_swap2i(v, w, left, last);
   hypre_qsort2i(v, w, left, last-1);
   hypre_qsort2i(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*   sort on w (double), move v (AB 11/04) */


void hypre_qsort2( HYPRE_Int *v,
	     double *w,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;
   swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (w[i] < w[left])
      {
         swap2(v, w, ++last, i);
      }
   swap2(v, w, left, last);
   hypre_qsort2(v, w, left, last-1);
   hypre_qsort2(v, w, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort on v, move w and z (AB 11/04) */

void hypre_qsort3i( HYPRE_Int *v,
                    HYPRE_Int *w,
                    HYPRE_Int *z,
                    HYPRE_Int  left,
                    HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap3i( v, w, z, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap3i(v, w, z, ++last, i);
      }
   }
   hypre_swap3i(v, w, z, left, last);
   hypre_qsort3i(v, w, z, left, last-1);
   hypre_qsort3i(v, w, z, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on absolute value */

void hypre_qsort3_abs(double *v,
                      HYPRE_Int *w,
                      HYPRE_Int *z,
                      HYPRE_Int  left,
                      HYPRE_Int  right )
{
   HYPRE_Int i, last;
   if (left >= right)
      return;
   hypre_swap3_d( v, w, z, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (fabs(v[i]) < fabs(v[left]))
      {
         hypre_swap3_d(v,w, z, ++last, i);
      }
   hypre_swap3_d(v, w, z, left, last);
   hypre_qsort3_abs(v, w, z, left, last-1);
   hypre_qsort3_abs(v, w, z, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on absolute value */

void hypre_qsort4_abs(double *v,
                      HYPRE_Int *w,
                      HYPRE_Int *z,
                      HYPRE_Int *y,
                      HYPRE_Int  left,
                      HYPRE_Int  right )
{
   HYPRE_Int i, last;
   if (left >= right)
      return;
   hypre_swap4_d( v, w, z, y, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (fabs(v[i]) < fabs(v[left]))
      {
         hypre_swap4_d(v,w, z, y, ++last, i);
      }
   hypre_swap4_d(v, w, z, y, left, last);
   hypre_qsort4_abs(v, w, z, y, left, last-1);
   hypre_qsort4_abs(v, w, z, y, last+1, right);
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* sort min to max based on absolute value */

void hypre_qsort_abs(double *w,
                     HYPRE_Int  left,
                     HYPRE_Int  right )
{
   HYPRE_Int i, last;
   if (left >= right)
      return;
   hypre_swap_d( w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (fabs(w[i]) < fabs(w[left]))
      {
         hypre_swap_d(w, ++last, i);
      }
   hypre_swap_d(w, left, last);
   hypre_qsort_abs(w, left, last-1);
   hypre_qsort_abs(w, last+1, right);
}




