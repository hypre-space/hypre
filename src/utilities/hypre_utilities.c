/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * hypre_multmod
 *--------------------------------------------------------------------------*/

/* This function computes (a*b) % mod, which can avoid overflow in large value of (a*b) */
HYPRE_Int
hypre_multmod(HYPRE_Int a,
              HYPRE_Int b,
              HYPRE_Int mod)
{
    HYPRE_Int res = 0; // Initialize result
    a %= mod;
    while (b)
    {
        // If b is odd, add a with result
        if (b & 1)
        {
            res = (res + a) % mod;
        }
        // Here we assume that doing 2*a
        // doesn't cause overflow
        a = (2 * a) % mod;
        b >>= 1;  // b = b / 2
    }
    return res;
}

/*--------------------------------------------------------------------------
 * hypre_MergeOrderedArrays: merge two ordered arrays
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_MergeOrderedArrays( HYPRE_Int  size1,     HYPRE_Int  *array1,
                          HYPRE_Int  size2,     HYPRE_Int  *array2,
                          HYPRE_Int *size3_ptr, HYPRE_Int **array3_ptr )
{
   HYPRE_Int  *array3;
   HYPRE_Int   i, j, k;

   array3 = hypre_CTAlloc(HYPRE_Int, (size1 + size2), HYPRE_MEMORY_HOST);

   i = j = k = 0;
   while (i < size1 && j < size2)
   {
      if (array1[i] > array2[j])
      {
         array3[k++] = array2[j++];
      }
      else if (array1[i] < array2[j])
      {
         array3[k++] = array1[i++];
      }
      else
      {
         array3[k++] = array1[i++];
         j++;
      }
   }

   while (i < size1)
   {
      array3[k++] = array1[i++];
   }

   while (j < size2)
   {
      array3[k++] = array2[j++];
   }

   /* Set pointers */
   *size3_ptr  = k;
   *array3_ptr = hypre_TReAlloc(array3, HYPRE_Int, k, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_partition1D
 *--------------------------------------------------------------------------*/
void
hypre_partition1D(HYPRE_Int  n, /* total number of elements */
                  HYPRE_Int  p, /* number of partitions */
                  HYPRE_Int  j, /* index of this partition */
                  HYPRE_Int *s, /* first element in this partition */
                  HYPRE_Int *e  /* past-the-end element */ )

{
   if (1 == p)
   {
      *s = 0;
      *e = n;
      return;
   }

   HYPRE_Int size = n / p;
   HYPRE_Int rest = n - size * p;
   if (j < rest)
   {
      *s = j * (size + 1);
      *e = (j + 1) * (size + 1);
   }
   else
   {
      *s = j * size + rest;
      *e = (j + 1) * size + rest;
   }
}
