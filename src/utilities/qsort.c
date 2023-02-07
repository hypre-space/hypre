/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

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

void hypre_swap_c( HYPRE_Complex *v,
                   HYPRE_Int      i,
                   HYPRE_Int      j )
{
   HYPRE_Complex temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap2( HYPRE_Int  *v,
                  HYPRE_Real *w,
                  HYPRE_Int   i,
                  HYPRE_Int   j )
{
   HYPRE_Int  temp;
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

void hypre_BigSwap2( HYPRE_BigInt *v,
                     HYPRE_Real   *w,
                     HYPRE_Int     i,
                     HYPRE_Int     j )
{
   HYPRE_BigInt temp;
   HYPRE_Real   temp2;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp2 = w[i];
   w[i] = w[j];
   w[j] = temp2;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_swap2i( HYPRE_Int  *v,
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

void hypre_BigSwap2i( HYPRE_BigInt *v,
                      HYPRE_Int    *w,
                      HYPRE_Int     i,
                      HYPRE_Int     j )
{
   HYPRE_BigInt big_temp;
   HYPRE_Int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


/* AB 11/04 */

void hypre_swap3i( HYPRE_Int  *v,
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

void hypre_swap3_d( HYPRE_Real *v,
                    HYPRE_Int  *w,
                    HYPRE_Int  *z,
                    HYPRE_Int   i,
                    HYPRE_Int   j )
{
   HYPRE_Int  temp;
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

/* swap (v[i], v[j]), (w[i], w[j]), and (z[v[i]], z[v[j]]) - DOK */
void hypre_swap3_d_perm( HYPRE_Int  *v,
                         HYPRE_Real *w,
                         HYPRE_Int  *z,
                         HYPRE_Int  i,
                         HYPRE_Int  j )
{
   HYPRE_Int temp;
   HYPRE_Real temp_d;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
   temp_d = w[i];
   w[i] = w[j];
   w[j] = temp_d;
   temp = z[v[i]];
   z[v[i]] = z[v[j]];
   z[v[j]] = temp;
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigSwap4_d( HYPRE_Real   *v,
                       HYPRE_BigInt *w,
                       HYPRE_Int    *z,
                       HYPRE_Int    *y,
                       HYPRE_Int     i,
                       HYPRE_Int     j )
{
   HYPRE_Int temp;
   HYPRE_BigInt big_temp;
   HYPRE_Real temp_d;

   temp_d = v[i];
   v[i] = v[j];
   v[j] = temp_d;
   big_temp = w[i];
   w[i] = w[j];
   w[j] = big_temp;
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
   {
      return;
   }
   hypre_swap(v, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap(v, ++last, i);
      }
   }
   hypre_swap(v, left, last);
   hypre_qsort0(v, left, last - 1);
   hypre_qsort0(v, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_qsort1( HYPRE_Int  *v,
                   HYPRE_Real *w,
                   HYPRE_Int   left,
                   HYPRE_Int   right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap2( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap2(v, w, ++last, i);
      }
   }
   hypre_swap2(v, w, left, last);
   hypre_qsort1(v, w, left, last - 1);
   hypre_qsort1(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsort1( HYPRE_BigInt *v,
                      HYPRE_Real   *w,
                      HYPRE_Int     left,
                      HYPRE_Int     right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwap2(v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwap2(v, w, ++last, i);
      }
   }
   hypre_BigSwap2(v, w, left, last);
   hypre_BigQsort1(v, w, left, last - 1);
   hypre_BigQsort1(v, w, last + 1, right);
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
   hypre_swap2i( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap2i(v, w, ++last, i);
      }
   }
   hypre_swap2i(v, w, left, last);
   hypre_qsort2i(v, w, left, last - 1);
   hypre_qsort2i(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsort2i( HYPRE_BigInt *v,
                       HYPRE_Int *w,
                       HYPRE_Int  left,
                       HYPRE_Int  right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwap2i( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwap2i(v, w, ++last, i);
      }
   }
   hypre_BigSwap2i(v, w, left, last);
   hypre_BigQsort2i(v, w, left, last - 1);
   hypre_BigQsort2i(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/*   sort on w (HYPRE_Real), move v (AB 11/04) */

void hypre_qsort2( HYPRE_Int  *v,
                   HYPRE_Real *w,
                   HYPRE_Int   left,
                   HYPRE_Int   right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap2( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (w[i] < w[left])
      {
         hypre_swap2(v, w, ++last, i);
      }
   }
   hypre_swap2(v, w, left, last);
   hypre_qsort2(v, w, left, last - 1);
   hypre_qsort2(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* qsort2 based on absolute value of entries in w. */
void hypre_qsort2_abs( HYPRE_Int  *v,
                       HYPRE_Real *w,
                       HYPRE_Int   left,
                       HYPRE_Int   right )
{
   HYPRE_Int i, last;
   if (left >= right)
   {
      return;
   }
   hypre_swap2( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (hypre_abs(w[i]) > hypre_abs(w[left]))
      {
         hypre_swap2(v, w, ++last, i);
      }
   }
   hypre_swap2(v, w, left, last);
   hypre_qsort2_abs(v, w, left, last - 1);
   hypre_qsort2_abs(v, w, last + 1, right);
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
   hypre_swap3i( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap3i(v, w, z, ++last, i);
      }
   }
   hypre_swap3i(v, w, z, left, last);
   hypre_qsort3i(v, w, z, left, last - 1);
   hypre_qsort3i(v, w, z, last + 1, right);
}

/* sort on v, move w and z DOK */
void hypre_qsort3ir( HYPRE_Int  *v,
                     HYPRE_Real *w,
                     HYPRE_Int  *z,
                     HYPRE_Int   left,
                     HYPRE_Int   right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap3_d_perm( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap3_d_perm(v, w, z, ++last, i);
      }
   }
   hypre_swap3_d_perm(v, w, z, left, last);
   hypre_qsort3ir(v, w, z, left, last - 1);
   hypre_qsort3ir(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on real array v */
void hypre_qsort3( HYPRE_Real *v,
                   HYPRE_Int  *w,
                   HYPRE_Int  *z,
                   HYPRE_Int   left,
                   HYPRE_Int   right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap3_d( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_swap3_d(v, w, z, ++last, i);
      }
   }
   hypre_swap3_d(v, w, z, left, last);
   hypre_qsort3(v, w, z, left, last - 1);
   hypre_qsort3(v, w, z, last + 1, right);
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
   {
      return;
   }
   hypre_swap3_d( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (hypre_abs(v[i]) < hypre_abs(v[left]))
      {
         hypre_swap3_d(v, w, z, ++last, i);
      }
   }
   hypre_swap3_d(v, w, z, left, last);
   hypre_qsort3_abs(v, w, z, left, last - 1);
   hypre_qsort3_abs(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

/* sort min to max based on absolute value */

void hypre_BigQsort4_abs( HYPRE_Real   *v,
                          HYPRE_BigInt *w,
                          HYPRE_Int    *z,
                          HYPRE_Int    *y,
                          HYPRE_Int     left,
                          HYPRE_Int     right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwap4_d( v, w, z, y, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (hypre_abs(v[i]) < hypre_abs(v[left]))
      {
         hypre_BigSwap4_d(v, w, z, y, ++last, i);
      }
   }
   hypre_BigSwap4_d(v, w, z, y, left, last);
   hypre_BigQsort4_abs(v, w, z, y, left, last - 1);
   hypre_BigQsort4_abs(v, w, z, y, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/
/* sort min to max based on absolute value */

void hypre_qsort_abs( HYPRE_Real *w,
                      HYPRE_Int   left,
                      HYPRE_Int   right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_swap_d( w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (hypre_abs(w[i]) < hypre_abs(w[left]))
      {
         hypre_swap_d(w, ++last, i);
      }
   }
   hypre_swap_d(w, left, last);
   hypre_qsort_abs(w, left, last - 1);
   hypre_qsort_abs(w, last + 1, right);
}


/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigSwapbi( HYPRE_BigInt *v,
                      HYPRE_Int    *w,
                      HYPRE_Int     i,
                      HYPRE_Int     j )
{
   HYPRE_BigInt big_temp;
   HYPRE_Int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsortbi( HYPRE_BigInt *v,
                       HYPRE_Int    *w,
                       HYPRE_Int     left,
                       HYPRE_Int     right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwapbi( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwapbi(v, w, ++last, i);
      }
   }
   hypre_BigSwapbi(v, w, left, last);
   hypre_BigQsortbi(v, w, left, last - 1);
   hypre_BigQsortbi(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigSwapLoc( HYPRE_BigInt *v,
                       HYPRE_Int    *w,
                       HYPRE_Int     i,
                       HYPRE_Int     j )
{
   HYPRE_BigInt big_temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   w[i] = j;
   w[j] = i;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsortbLoc( HYPRE_BigInt *v,
                         HYPRE_Int    *w,
                         HYPRE_Int     left,
                         HYPRE_Int     right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwapLoc( v, w, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwapLoc(v, w, ++last, i);
      }
   }
   hypre_BigSwapLoc(v, w, left, last);
   hypre_BigQsortbLoc(v, w, left, last - 1);
   hypre_BigQsortbLoc(v, w, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/


void hypre_BigSwapb2i( HYPRE_BigInt *v,
                       HYPRE_Int    *w,
                       HYPRE_Int    *z,
                       HYPRE_Int     i,
                       HYPRE_Int     j )
{
   HYPRE_BigInt big_temp;
   HYPRE_Int temp;

   big_temp = v[i];
   v[i] = v[j];
   v[j] = big_temp;
   temp = w[i];
   w[i] = w[j];
   w[j] = temp;
   temp = z[i];
   z[i] = z[j];
   z[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsortb2i( HYPRE_BigInt *v,
                        HYPRE_Int    *w,
                        HYPRE_Int    *z,
                        HYPRE_Int     left,
                        HYPRE_Int     right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwapb2i( v, w, z, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwapb2i(v, w, z, ++last, i);
      }
   }
   hypre_BigSwapb2i(v, w, z, left, last);
   hypre_BigQsortb2i(v, w, z, left, last - 1);
   hypre_BigQsortb2i(v, w, z, last + 1, right);
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigSwap( HYPRE_BigInt *v,
                    HYPRE_Int     i,
                    HYPRE_Int     j )
{
   HYPRE_BigInt temp;

   temp = v[i];
   v[i] = v[j];
   v[j] = temp;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void hypre_BigQsort0( HYPRE_BigInt *v,
                      HYPRE_Int     left,
                      HYPRE_Int     right )
{
   HYPRE_Int i, last;

   if (left >= right)
   {
      return;
   }
   hypre_BigSwap( v, left, (left + right) / 2);
   last = left;
   for (i = left + 1; i <= right; i++)
   {
      if (v[i] < v[left])
      {
         hypre_BigSwap(v, ++last, i);
      }
   }
   hypre_BigSwap(v, left, last);
   hypre_BigQsort0(v, left, last - 1);
   hypre_BigQsort0(v, last + 1, right);
}

// Recursive DFS search.
static void hypre_search_row(HYPRE_Int            row,
                             const HYPRE_Int     *row_ptr,
                             const HYPRE_Int     *col_inds,
                             const HYPRE_Complex *data,
                             HYPRE_Int           *visited,
                             HYPRE_Int           *ordering,
                             HYPRE_Int           *order_ind)
{
   // If this row has not been visited, call recursive DFS on nonzero
   // column entries
   if (!visited[row])
   {
      HYPRE_Int j;
      visited[row] = 1;
      for (j = row_ptr[row]; j < row_ptr[row + 1]; j++)
      {
         HYPRE_Int col = col_inds[j];
         hypre_search_row(col, row_ptr, col_inds, data,
                          visited, ordering, order_ind);
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
void hypre_topo_sort( const HYPRE_Int     *row_ptr,
                      const HYPRE_Int     *col_inds,
                      const HYPRE_Complex *data,
                      HYPRE_Int           *ordering,
                      HYPRE_Int            n)
{
   HYPRE_Int *visited = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int order_ind = 0;
   HYPRE_Int temp_row = 0;
   while (order_ind < n)
   {
      hypre_search_row(temp_row, row_ptr, col_inds, data,
                       visited, ordering, &order_ind);
      temp_row += 1;
      if (temp_row == n)
      {
         temp_row = 0;
      }
   }
   hypre_TFree(visited, HYPRE_MEMORY_HOST);
}


// Recursive DFS search.
static void hypre_dense_search_row(HYPRE_Int            row,
                                   const HYPRE_Complex *L,
                                   HYPRE_Int           *visited,
                                   HYPRE_Int           *ordering,
                                   HYPRE_Int           *order_ind,
                                   HYPRE_Int            n,
                                   HYPRE_Int            is_col_major)
{
   // If this row has not been visited, call recursive DFS on nonzero
   // column entries
   if (!visited[row])
   {
      HYPRE_Int col;
      visited[row] = 1;
      for (col = 0; col < n; col++)
      {
         HYPRE_Complex val;
         if (is_col_major)
         {
            val = L[col * n + row];
         }
         else
         {
            val = L[row * n + col];
         }
         if (hypre_cabs(val) > 1e-14)
         {
            hypre_dense_search_row(col, L, visited, ordering, order_ind, n, is_col_major);
         }
      }
      // Add node to ordering *after* it has been searched
      ordering[*order_ind] = row;
      *order_ind += 1;
   }
}


// Find topological ordering of acyclic dense matrix in column major
// format. That is, find ordering of matrix to be triangular.
//
// INPUT
// -----
//    - L[] : dense nxn matrix in column major format
//    - ordering[] should be empty array of length n
//    - row is the row to start the search from
void hypre_dense_topo_sort(const HYPRE_Complex *L,
                           HYPRE_Int           *ordering,
                           HYPRE_Int            n,
                           HYPRE_Int            is_col_major)
{
   HYPRE_Int *visited = hypre_CTAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);
   HYPRE_Int order_ind = 0;
   HYPRE_Int temp_row = 0;
   while (order_ind < n)
   {
      hypre_dense_search_row(temp_row, L, visited, ordering, &order_ind, n, is_col_major);
      temp_row += 1;
      if (temp_row == n)
      {
         temp_row = 0;
      }
   }
   hypre_TFree(visited, HYPRE_MEMORY_HOST);
}
