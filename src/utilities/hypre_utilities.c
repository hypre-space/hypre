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

/*--------------------------------------------------------------------------
 * hypre_strcpy
 *
 * Note: strcpy that allows overlapping in memory
 *--------------------------------------------------------------------------*/

char *
hypre_strcpy(char *destination, const char *source)
{
   size_t len = strlen(source);

   /* no overlapping */
   if (source > destination + len || destination > source + len)
   {
      return strcpy(destination, source);
   }
   else
   {
      /* +1: including the terminating null character */
      return ((char *) memmove(destination, source, len + 1));
   }
}
