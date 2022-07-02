/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef HYPRE_GSELIM_H
#define HYPRE_GSELIM_H

#define hypre_gselim(A,x,n,error)                      \
{                                                      \
   HYPRE_Int    j,k,m;                                 \
   HYPRE_Real factor;                                  \
   HYPRE_Real divA;                                    \
   error = 0;                                          \
   if (n == 1)  /* A is 1x1 */                         \
   {                                                   \
      if (A[0] != 0.0)                                 \
      {                                                \
         x[0] = x[0]/A[0];                             \
      }                                                \
      else                                             \
      {                                                \
         error++;                                      \
      }                                                \
   }                                                   \
   else/* A is nxn. Forward elimination */             \
   {                                                   \
      for (k = 0; k < n-1; k++)                        \
      {                                                \
         if (A[k*n+k] != 0.0)                          \
         {                                             \
            divA = 1.0/A[k*n+k];                       \
            for (j = k+1; j < n; j++)                  \
            {                                          \
               if (A[j*n+k] != 0.0)                    \
               {                                       \
                  factor = A[j*n+k]*divA;              \
                  for (m = k+1; m < n; m++)            \
                  {                                    \
                     A[j*n+m]  -= factor * A[k*n+m];   \
                  }                                    \
                  x[j] -= factor * x[k];               \
               }                                       \
            }                                          \
         }                                             \
      }                                                \
      /* Back Substitution  */                         \
      for (k = n-1; k > 0; --k)                        \
      {                                                \
         if (A[k*n+k] != 0.0)                          \
         {                                             \
            x[k] /= A[k*n+k];                          \
            for (j = 0; j < k; j++)                    \
            {                                          \
               if (A[j*n+k] != 0.0)                    \
               {                                       \
                  x[j] -= x[k] * A[j*n+k];             \
               }                                       \
            }                                          \
         }                                             \
      }                                                \
      if (A[0] != 0.0) x[0] /= A[0];                   \
   }                                                   \
}

#endif /* #ifndef HYPRE_GSELIM_H */

