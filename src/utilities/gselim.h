/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_GSELIM_H
#define HYPRE_GSELIM_H

#define hypre_gselim(A,x,n)                                            \
{                                                                      \
   HYPRE_Int  __j, __k, __m;                                           \
   HYPRE_Real factor;                                                  \
   HYPRE_Real divA;                                                    \
   if (n == 1)  /* A is 1x1 */                                         \
   {                                                                   \
      if (A[0] != 0.0)                                                 \
      {                                                                \
         x[0] = x[0]/A[0];                                             \
      }                                                                \
   }                                                                   \
   else/* A is nxn. Forward elimination */                             \
   {                                                                   \
      for (__k = 0; __k < n - 1; __k++)                                \
      {                                                                \
         if (A[__k * n + __k] != 0.0)                                  \
         {                                                             \
            divA = 1.0/A[__k * n + __k];                               \
            for (__j = __k + 1; __j < n; __j++)                        \
            {                                                          \
               if (A[__j * n + __k] != 0.0)                            \
               {                                                       \
                  factor = A[__j * n + __k] * divA;                    \
                  for (__m = __k + 1; __m < n; __m++)                  \
                  {                                                    \
                     A[__j * n + __m]  -= factor * A[__k * n + __m];   \
                  }                                                    \
                  x[__j] -= factor * x[__k];                           \
               }                                                       \
            }                                                          \
         }                                                             \
      }                                                                \
      /* Back Substitution  */                                         \
      for (__k = n - 1; __k > 0; --__k)                                \
      {                                                                \
         if (A[__k * n + __k] != 0.0)                                  \
         {                                                             \
            x[__k] /= A[__k * n + __k];                                \
            for (__j = 0; __j < __k; __j++)                            \
            {                                                          \
               if (A[__j * n + __k] != 0.0)                            \
               {                                                       \
                  x[__j] -= x[__k] * A[__j * n + __k];                 \
               }                                                       \
            }                                                          \
         }                                                             \
      }                                                                \
      if (A[0] != 0.0) x[0] /= A[0];                                   \
   }                                                                   \
}

#endif /* #ifndef HYPRE_GSELIM_H */
