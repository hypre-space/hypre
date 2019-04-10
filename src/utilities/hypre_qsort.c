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


#include <math.h>
#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap( HYPRE_Int *v,
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

void hypre_swap2(HYPRE_Int     *v,
           HYPRE_Real  *w,
           HYPRE_Int      i,
           HYPRE_Int      j )
{
   HYPRE_Int temp;
   HYPRE_Real temp2;

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

void hypre_swap3_d(HYPRE_Real  *v,
                  HYPRE_Int  *w,
                  HYPRE_Int  *z,
                  HYPRE_Int  i,
                  HYPRE_Int  j )
{
   HYPRE_Int temp;
   HYPRE_Real temp_d;


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

void hypre_swap4_d(HYPRE_Real  *v,
                  HYPRE_Int  *w,
                  HYPRE_Int  *z,
                  HYPRE_Int *y,
                  HYPRE_Int  i,
                  HYPRE_Int  j )
{
   HYPRE_Int temp;
   HYPRE_Real temp_d;


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

void hypre_swap_d( HYPRE_Real *v,
                   HYPRE_Int  i,
                   HYPRE_Int  j )
{
   HYPRE_Real temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_qsort0( HYPRE_Int *v,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;
   hypre_swap( v, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         hypre_swap(v, ++last, i);
      }
   hypre_swap(v, left, last);
   hypre_qsort0(v, left, last-1);
   hypre_qsort0(v, last+1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_qsort1( HYPRE_Int *v,
	     HYPRE_Real *w,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;
   hypre_swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (v[i] < v[left])
      {
         hypre_swap2(v, w, ++last, i);
      }
   hypre_swap2(v, w, left, last);
   hypre_qsort1(v, w, left, last-1);
   hypre_qsort1(v, w, last+1, right);
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

/*   sort on w (HYPRE_Real), move v (AB 11/04) */


void hypre_qsort2( HYPRE_Int *v,
	     HYPRE_Real *w,
             HYPRE_Int  left,
             HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
      return;
   hypre_swap2( v, w, left, (left+right)/2);
   last = left;
   for (i = left+1; i <= right; i++)
      if (w[i] < w[left])
      {
         hypre_swap2(v, w, ++last, i);
      }
   hypre_swap2(v, w, left, last);
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

void hypre_qsort3_abs(HYPRE_Real *v,
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

void hypre_qsort4_abs(HYPRE_Real *v,
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

void hypre_qsort_abs(HYPRE_Real *w,
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


// Recursive DFS search.
void hypre_search_row(HYPRE_Int row,
                      HYPRE_Int *row_ptr,
                      HYPRE_Int *col_inds,
                      HYPRE_Real *data,
                      HYPRE_Int *visited,
                      HYPRE_Int *ordering,
                      HYPRE_Int *order_ind)
{
   HYPRE_Int j;
   // If this row has not been visited, call recursive DFS on nonzero
   // column entries
   if (!visited[row]) {
      visited[row] = 1;
      for (j=row_ptr[row]; j<row_ptr[row+1]; j++) {
         HYPRE_Int col = col_inds[j];
         hypre_search_row(col, row_ptr, col_inds, data,
                         visited, ordering, order_ind);
      }
      // Add node to ordering *after* it has been searched
      ordering[*order_ind] = row;
      *order_ind += 1;
   }
}


// Recursive DFS search
void hypre_search_row_submat(HYPRE_Int row,
                             HYPRE_Int *row_ptr,
                             HYPRE_Int *col_inds,
                             HYPRE_Real *data,
                             HYPRE_Int *visited,
                             HYPRE_Int *ordering,
                             HYPRE_Int *order_ind,
                             HYPRE_Int *cf_marker,
                             HYPRE_Int CF)
{
   HYPRE_Int j;
   // Check that row is appropriate marker (C or F) and has not been visited.
   // Then call recursive DFS on nonzero column entries.
   if ( (cf_marker[row] == CF) && (!visited[row]) ) {
      visited[row] = 1;
      for (j=row_ptr[row]; j<row_ptr[row+1]; j++) {
         HYPRE_Int col = col_inds[j];
         hypre_search_row_submat(col, row_ptr, col_inds, data, visited,
                                ordering, order_ind, cf_marker, CF);
      }
      // Add node to ordering *after* it has been searched
      ordering[*order_ind] = row;
      *order_ind += 1;
   }
}


// Find topological ordering on acyclic CSR matrix. That is, find ordering
// of matrix to be triangular.
//
// INPUT
// -----
//    - rowptr[], colinds[], data[] form a CSR structure for nxn matrix
//    - ordering[] should be empty array of length n
//    - row is the row to start the search from
void hypre_topo_sort(HYPRE_Int *row_ptr,
                     HYPRE_Int *col_inds,
                     HYPRE_Real *data,
                     HYPRE_Int *ordering,
                     HYPRE_Int n)
{
   HYPRE_Int *visited = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int order_ind = 0;
   HYPRE_Int temp_row = 0;
   while (order_ind < n) {
      hypre_search_row(temp_row, row_ptr, col_inds, data,
                       visited, ordering, &order_ind);
      temp_row += 1;
   }
   hypre_TFree(visited, HYPRE_MEMORY_HOST);
}


// Find topological ordering on acyclic CSR submatrix. That is, find ordering
// of matrix to be triangular, where submatrix indices, i, are denoted by
// cf_marker[i] == CF.
//
// INPUT
// -----
//    - rowptr[], colinds[], data[] form a CSR structure for nxn matrix
//    - ordering[] should be empty array of length n
//    - row is the row to start the search from
void hypre_topo_sort_submat(HYPRE_Int *row_ptr,
                            HYPRE_Int *col_inds,
                            HYPRE_Real *data,
                            HYPRE_Int *ordering,
                            HYPRE_Int n,
                            HYPRE_Int *cf_marker,
                            HYPRE_Int CF)
{
   HYPRE_Int i;
   HYPRE_Int *visited = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int order_ind = 0;
   HYPRE_Int temp_row = 0;
   while (order_ind < n) {
      hypre_search_row_submat(temp_row, row_ptr, col_inds, data,
                              visited, ordering, &order_ind, cf_marker, CF);
      temp_row += 1;
   }
   for (i=order_ind; i<n; i++) {
      ordering[i] = -1;
   }
   hypre_TFree(visited, HYPRE_MEMORY_HOST);
}

