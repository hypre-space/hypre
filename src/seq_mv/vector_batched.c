/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "seq_mv.h"

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpy8
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorMassAxpy8( HYPRE_Complex *alpha,
                          hypre_Vector **x,
                          hypre_Vector  *y, HYPRE_Int k)
{
   HYPRE_Complex  *x_data = hypre_VectorData(x[0]);
   HYPRE_Complex  *y_data = hypre_VectorData(y);
   HYPRE_Int       size   = hypre_VectorSize(x[0]);

   HYPRE_Int      i, j, jstart, restk;


   restk = (k - (k / 8 * 8));

   if (k > 7)
   {
      for (j = 0; j < k - 7; j += 8)
      {
         jstart = j * size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            y_data[i] += alpha[j] * x_data[jstart + i] + alpha[j + 1] * x_data[jstart + i + size]
                         + alpha[j + 2] * x_data[(j + 2) * size + i] + alpha[j + 3] * x_data[(j + 3) * size + i]
                         + alpha[j + 4] * x_data[(j + 4) * size + i] + alpha[j + 5] * x_data[(j + 5) * size + i]
                         + alpha[j + 6] * x_data[(j + 6) * size + i] + alpha[j + 7] * x_data[(j + 7) * size + i];
         }
      }
   }
   if (restk == 1)
   {
      jstart = (k - 1) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 1] * x_data[jstart + i];
      }
   }
   else if (restk == 2)
   {
      jstart = (k - 2) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 2] * x_data[jstart + i] + alpha[k - 1] * x_data[jstart + size + i];
      }
   }
   else if (restk == 3)
   {
      jstart = (k - 3) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 3] * x_data[jstart + i] + alpha[k - 2] * x_data[jstart + size + i] + alpha[k
                                                                                                           - 1] *
                      x_data[(k - 1) * size + i];
      }
   }
   else if (restk == 4)
   {
      jstart = (k - 4) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 4] * x_data[(k - 4) * size + i] + alpha[k - 3] * x_data[(k - 3) * size + i]
                      + alpha[k - 2] * x_data[(k - 2) * size + i] + alpha[k - 1] * x_data[(k - 1) * size + i];
      }
   }
   else if (restk == 5)
   {
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += + alpha[k - 5] * x_data[(k - 5) * size + i] + alpha[k - 4] * x_data[(k - 4) * size + i]
                      + alpha[k - 3] * x_data[(k - 3) * size + i] + alpha[k - 2] * x_data[(k - 2) * size + i]
                      + alpha[k - 1] * x_data[(k - 1) * size + i];
      }
   }
   else if (restk == 6)
   {
      jstart = (k - 6) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 6] * x_data[jstart + i] + alpha[k - 5] * x_data[jstart + i + size]
                      + alpha[k - 4] * x_data[(k - 4) * size + i] + alpha[k - 3] * x_data[(k - 3) * size + i]
                      + alpha[k - 2] * x_data[(k - 2) * size + i] + alpha[k - 1] * x_data[(k - 1) * size + i];
      }
   }
   else if (restk == 7)
   {
      jstart = (k - 7) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 7] * x_data[jstart + i] + alpha[k - 6] * x_data[jstart + i + size]
                      + alpha[k - 5] * x_data[(k - 5) * size + i] + alpha[k - 4] * x_data[(k - 4) * size + i]
                      + alpha[k - 3] * x_data[(k - 3) * size + i] + alpha[k - 2] * x_data[(k - 2) * size + i]
                      + alpha[k - 1] * x_data[(k - 1) * size + i];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpy4
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorMassAxpy4( HYPRE_Complex *alpha,
                          hypre_Vector **x,
                          hypre_Vector  *y, HYPRE_Int k)
{
   HYPRE_Complex  *x_data = hypre_VectorData(x[0]);
   HYPRE_Complex  *y_data = hypre_VectorData(y);
   HYPRE_Int       size   = hypre_VectorSize(x[0]);

   HYPRE_Int      i, j, jstart, restk;


   restk = (k - (k / 4 * 4));

   if (k > 3)
   {
      for (j = 0; j < k - 3; j += 4)
      {
         jstart = j * size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            y_data[i] += alpha[j] * x_data[jstart + i] + alpha[j + 1] * x_data[jstart + i + size]
                         + alpha[j + 2] * x_data[(j + 2) * size + i] + alpha[j + 3] * x_data[(j + 3) * size + i];
         }
      }
   }
   if (restk == 1)
   {
      jstart = (k - 1) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 1] * x_data[jstart + i];
      }
   }
   else if (restk == 2)
   {
      jstart = (k - 2) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 2] * x_data[jstart + i] + alpha[k - 1] * x_data[jstart + size + i];
      }
   }
   else if (restk == 3)
   {
      jstart = (k - 3) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         y_data[i] += alpha[k - 3] * x_data[jstart + i] + alpha[k - 2] * x_data[jstart + size + i] + alpha[k
                                                                                                           - 1] *
                      x_data[(k - 1) * size + i];
      }
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorMassAxpy( HYPRE_Complex *alpha,
                         hypre_Vector **x,
                         hypre_Vector  *y, HYPRE_Int k, HYPRE_Int unroll)
{
   HYPRE_Complex  *x_data = hypre_VectorData(x[0]);
   HYPRE_Complex  *y_data = hypre_VectorData(y);
   HYPRE_Int       size   = hypre_VectorSize(x[0])*hypre_VectorNumVectors(x[0]);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

   HYPRE_Complex * d_alpha = hypre_CTAlloc(HYPRE_Complex,k,hypre_VectorMemoryLocation(y));
   hypre_TMemcpy(d_alpha, alpha, HYPRE_Complex, k, hypre_VectorMemoryLocation(y), HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_CUBLAS)

   // This works but it's not as fast as the Maxpy kernel below.
   double a = 1.0, b = 1.0;
   HYPRE_CUBLAS_CALL( cublasDgemv(hypre_HandleCublasHandle(hypre_handle()), CUBLAS_OP_N, size, k, &a,
				  x_data, size, d_alpha, 1, &b, y_data, 1));

#else

   hypreDevice_MassAxpy(k, size, d_alpha, x_data, y_data);

#endif

   hypre_TFree(d_alpha,hypre_VectorMemoryLocation(y));


#else

   HYPRE_Int      i, j, jstart;

   if (unroll == 8)
   {
      hypre_SeqVectorMassAxpy8(alpha, x, y, k);
      return hypre_error_flag;
   }
   else if (unroll == 4)
   {
      hypre_SeqVectorMassAxpy4(alpha, x, y, k);
      return hypre_error_flag;
   }
   else
   {
      for (j = 0; j < k; j++)
      {
         jstart = j * size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            y_data[i] += alpha[j] * x_data[jstart + i];
         }
      }
   }

#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProd8
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassInnerProd8( hypre_Vector *x,
                                         hypre_Vector **y, HYPRE_Int k, HYPRE_Real *result)
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y[0]);
   HYPRE_Int      size   = hypre_VectorSize(x)*hypre_VectorNumVectors(x);

   HYPRE_Int      i, j, restk;
   HYPRE_Real res1;
   HYPRE_Real res2;
   HYPRE_Real res3;
   HYPRE_Real res4;
   HYPRE_Real res5;
   HYPRE_Real res6;
   HYPRE_Real res7;
   HYPRE_Real res8;
   HYPRE_Int jstart;
   HYPRE_Int jstart1;
   HYPRE_Int jstart2;
   HYPRE_Int jstart3;
   HYPRE_Int jstart4;
   HYPRE_Int jstart5;
   HYPRE_Int jstart6;
   HYPRE_Int jstart7;

   restk = (k - (k / 8 * 8));

   if (k > 7)
   {
      for (j = 0; j < k - 7; j += 8)
      {
         res1 = 0;
         res2 = 0;
         res3 = 0;
         res4 = 0;
         res5 = 0;
         res6 = 0;
         res7 = 0;
         res8 = 0;
         jstart = j * size;
         jstart1 = jstart + size;
         jstart2 = jstart1 + size;
         jstart3 = jstart2 + size;
         jstart4 = jstart3 + size;
         jstart5 = jstart4 + size;
         jstart6 = jstart5 + size;
         jstart7 = jstart6 + size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5,res6,res7,res8) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
            res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
            res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
            res4 += hypre_conj(y_data[jstart3 + i]) * x_data[i];
            res5 += hypre_conj(y_data[jstart4 + i]) * x_data[i];
            res6 += hypre_conj(y_data[jstart5 + i]) * x_data[i];
            res7 += hypre_conj(y_data[jstart6 + i]) * x_data[i];
            res8 += hypre_conj(y_data[jstart7 + i]) * x_data[i];
         }
         result[j] = res1;
         result[j + 1] = res2;
         result[j + 2] = res3;
         result[j + 3] = res4;
         result[j + 4] = res5;
         result[j + 5] = res6;
         result[j + 6] = res7;
         result[j + 7] = res8;
      }
   }
   if (restk == 1)
   {
      res1 = 0;
      jstart = (k - 1) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
      }
      result[k - 1] = res1;
   }
   else if (restk == 2)
   {
      res1 = 0;
      res2 = 0;
      jstart = (k - 2) * size;
      jstart1 = jstart + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
      }
      result[k - 2] = res1;
      result[k - 1] = res2;
   }
   else if (restk == 3)
   {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      jstart = (k - 3) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2,res3) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
         res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
      }
      result[k - 3] = res1;
      result[k - 2] = res2;
      result[k - 1] = res3;
   }
   else if (restk == 4)
   {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      jstart = (k - 4) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
         res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
         res4 += hypre_conj(y_data[jstart3 + i]) * x_data[i];
      }
      result[k - 4] = res1;
      result[k - 3] = res2;
      result[k - 2] = res3;
      result[k - 1] = res4;
   }
   else if (restk == 5)
   {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      res5 = 0;
      jstart = (k - 5) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
      jstart4 = jstart3 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
         res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
         res4 += hypre_conj(y_data[jstart3 + i]) * x_data[i];
         res5 += hypre_conj(y_data[jstart4 + i]) * x_data[i];
      }
      result[k - 5] = res1;
      result[k - 4] = res2;
      result[k - 3] = res3;
      result[k - 2] = res4;
      result[k - 1] = res5;
   }
   else if (restk == 6)
   {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      res5 = 0;
      res6 = 0;
      jstart = (k - 6) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
      jstart4 = jstart3 + size;
      jstart5 = jstart4 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5,res6) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
         res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
         res4 += hypre_conj(y_data[jstart3 + i]) * x_data[i];
         res5 += hypre_conj(y_data[jstart4 + i]) * x_data[i];
         res6 += hypre_conj(y_data[jstart5 + i]) * x_data[i];
      }
      result[k - 6] = res1;
      result[k - 5] = res2;
      result[k - 4] = res3;
      result[k - 3] = res4;
      result[k - 2] = res5;
      result[k - 1] = res6;
   }
   else if (restk == 7)
   {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      res5 = 0;
      res6 = 0;
      res7 = 0;
      jstart = (k - 7) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
      jstart4 = jstart3 + size;
      jstart5 = jstart4 + size;
      jstart6 = jstart5 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5,res6,res7) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
         res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
         res4 += hypre_conj(y_data[jstart3 + i]) * x_data[i];
         res5 += hypre_conj(y_data[jstart4 + i]) * x_data[i];
         res6 += hypre_conj(y_data[jstart5 + i]) * x_data[i];
         res7 += hypre_conj(y_data[jstart6 + i]) * x_data[i];
      }
      result[k - 7] = res1;
      result[k - 6] = res2;
      result[k - 5] = res3;
      result[k - 4] = res4;
      result[k - 3] = res5;
      result[k - 2] = res6;
      result[k - 1] = res7;
   }


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProd4
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassInnerProd4( hypre_Vector *x,
                                         hypre_Vector **y, HYPRE_Int k, HYPRE_Real *result)
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y[0]);
   HYPRE_Int      size   = hypre_VectorSize(x);

   HYPRE_Int      i, j, restk;
   HYPRE_Real res1;
   HYPRE_Real res2;
   HYPRE_Real res3;
   HYPRE_Real res4;
   HYPRE_Int jstart;
   HYPRE_Int jstart1;
   HYPRE_Int jstart2;
   HYPRE_Int jstart3;

   restk = (k - (k / 4 * 4));

   if (k > 3)
   {
      for (j = 0; j < k - 3; j += 4)
      {
         res1 = 0;
         res2 = 0;
         res3 = 0;
         res4 = 0;
         jstart = j * size;
         jstart1 = jstart + size;
         jstart2 = jstart1 + size;
         jstart3 = jstart2 + size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
            res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
            res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
            res4 += hypre_conj(y_data[jstart3 + i]) * x_data[i];
         }
         result[j] = res1;
         result[j + 1] = res2;
         result[j + 2] = res3;
         result[j + 3] = res4;
      }
   }
   if (restk == 1)
   {
      res1 = 0;
      jstart = (k - 1) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
      }
      result[k - 1] = res1;
   }
   else if (restk == 2)
   {
      res1 = 0;
      res2 = 0;
      jstart = (k - 2) * size;
      jstart1 = jstart + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
      }
      result[k - 2] = res1;
      result[k - 1] = res2;
   }
   else if (restk == 3)
   {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      jstart = (k - 3) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res1,res2,res3) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res1 += hypre_conj(y_data[jstart + i]) * x_data[i];
         res2 += hypre_conj(y_data[jstart1 + i]) * x_data[i];
         res3 += hypre_conj(y_data[jstart2 + i]) * x_data[i];
      }
      result[k - 3] = res1;
      result[k - 2] = res2;
      result[k - 1] = res3;
   }


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassDotpTwo8
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassDotpTwo8( hypre_Vector *x, hypre_Vector *y,
                                       hypre_Vector **z, HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y)
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex *z_data = hypre_VectorData(z[0]);
   HYPRE_Int      size   = hypre_VectorSize(x);

   HYPRE_Int      i, j, restk;
   HYPRE_Real res_x1;
   HYPRE_Real res_x2;
   HYPRE_Real res_x3;
   HYPRE_Real res_x4;
   HYPRE_Real res_x5;
   HYPRE_Real res_x6;
   HYPRE_Real res_x7;
   HYPRE_Real res_x8;
   HYPRE_Real res_y1;
   HYPRE_Real res_y2;
   HYPRE_Real res_y3;
   HYPRE_Real res_y4;
   HYPRE_Real res_y5;
   HYPRE_Real res_y6;
   HYPRE_Real res_y7;
   HYPRE_Real res_y8;
   HYPRE_Int jstart;
   HYPRE_Int jstart1;
   HYPRE_Int jstart2;
   HYPRE_Int jstart3;
   HYPRE_Int jstart4;
   HYPRE_Int jstart5;
   HYPRE_Int jstart6;
   HYPRE_Int jstart7;

   restk = (k - (k / 8 * 8));

   if (k > 7)
   {
      for (j = 0; j < k - 7; j += 8)
      {
         res_x1 = 0;
         res_x2 = 0;
         res_x3 = 0;
         res_x4 = 0;
         res_x5 = 0;
         res_x6 = 0;
         res_x7 = 0;
         res_x8 = 0;
         res_y1 = 0;
         res_y2 = 0;
         res_y3 = 0;
         res_y4 = 0;
         res_y5 = 0;
         res_y6 = 0;
         res_y7 = 0;
         res_y8 = 0;
         jstart = j * size;
         jstart1 = jstart + size;
         jstart2 = jstart1 + size;
         jstart3 = jstart2 + size;
         jstart4 = jstart3 + size;
         jstart5 = jstart4 + size;
         jstart6 = jstart5 + size;
         jstart7 = jstart6 + size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_x6,res_x7,res_x8,res_y1,res_y2,res_y3,res_y4,res_y5,res_y6,res_y7,res_y8) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
            res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
            res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
            res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
            res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
            res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
            res_x4 += hypre_conj(z_data[jstart3 + i]) * x_data[i];
            res_y4 += hypre_conj(z_data[jstart3 + i]) * y_data[i];
            res_x5 += hypre_conj(z_data[jstart4 + i]) * x_data[i];
            res_y5 += hypre_conj(z_data[jstart4 + i]) * y_data[i];
            res_x6 += hypre_conj(z_data[jstart5 + i]) * x_data[i];
            res_y6 += hypre_conj(z_data[jstart5 + i]) * y_data[i];
            res_x7 += hypre_conj(z_data[jstart6 + i]) * x_data[i];
            res_y7 += hypre_conj(z_data[jstart6 + i]) * y_data[i];
            res_x8 += hypre_conj(z_data[jstart7 + i]) * x_data[i];
            res_y8 += hypre_conj(z_data[jstart7 + i]) * y_data[i];
         }
         result_x[j] = res_x1;
         result_x[j + 1] = res_x2;
         result_x[j + 2] = res_x3;
         result_x[j + 3] = res_x4;
         result_x[j + 4] = res_x5;
         result_x[j + 5] = res_x6;
         result_x[j + 6] = res_x7;
         result_x[j + 7] = res_x8;
         result_y[j] = res_y1;
         result_y[j + 1] = res_y2;
         result_y[j + 2] = res_y3;
         result_y[j + 3] = res_y4;
         result_y[j + 4] = res_y5;
         result_y[j + 5] = res_y6;
         result_y[j + 6] = res_y7;
         result_y[j + 7] = res_y8;
      }
   }
   if (restk == 1)
   {
      res_x1 = 0;
      res_y1 = 0;
      jstart = (k - 1) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_y1) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
      }
      result_x[k - 1] = res_x1;
      result_y[k - 1] = res_y1;
   }
   else if (restk == 2)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_y1 = 0;
      res_y2 = 0;
      jstart = (k - 2) * size;
      jstart1 = jstart + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_y1,res_y2) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
      }
      result_x[k - 2] = res_x1;
      result_x[k - 1] = res_x2;
      result_y[k - 2] = res_y1;
      result_y[k - 1] = res_y2;
   }
   else if (restk == 3)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      jstart = (k - 3) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_y1,res_y2,res_y3) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
         res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
         res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
      }
      result_x[k - 3] = res_x1;
      result_x[k - 2] = res_x2;
      result_x[k - 1] = res_x3;
      result_y[k - 3] = res_y1;
      result_y[k - 2] = res_y2;
      result_y[k - 1] = res_y3;
   }
   else if (restk == 4)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_x4 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      res_y4 = 0;
      jstart = (k - 4) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_y1,res_y2,res_y3,res_y4) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
         res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
         res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
         res_x4 += hypre_conj(z_data[jstart3 + i]) * x_data[i];
         res_y4 += hypre_conj(z_data[jstart3 + i]) * y_data[i];
      }
      result_x[k - 4] = res_x1;
      result_x[k - 3] = res_x2;
      result_x[k - 2] = res_x3;
      result_x[k - 1] = res_x4;
      result_y[k - 4] = res_y1;
      result_y[k - 3] = res_y2;
      result_y[k - 2] = res_y3;
      result_y[k - 1] = res_y4;
   }
   else if (restk == 5)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_x4 = 0;
      res_x5 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      res_y4 = 0;
      res_y5 = 0;
      jstart = (k - 5) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
      jstart4 = jstart3 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_y1,res_y2,res_y3,res_y4,res_y5) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
         res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
         res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
         res_x4 += hypre_conj(z_data[jstart3 + i]) * x_data[i];
         res_y4 += hypre_conj(z_data[jstart3 + i]) * y_data[i];
         res_x5 += hypre_conj(z_data[jstart4 + i]) * x_data[i];
         res_y5 += hypre_conj(z_data[jstart4 + i]) * y_data[i];
      }
      result_x[k - 5] = res_x1;
      result_x[k - 4] = res_x2;
      result_x[k - 3] = res_x3;
      result_x[k - 2] = res_x4;
      result_x[k - 1] = res_x5;
      result_y[k - 5] = res_y1;
      result_y[k - 4] = res_y2;
      result_y[k - 3] = res_y3;
      result_y[k - 2] = res_y4;
      result_y[k - 1] = res_y5;
   }
   else if (restk == 6)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_x4 = 0;
      res_x5 = 0;
      res_x6 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      res_y4 = 0;
      res_y5 = 0;
      res_y6 = 0;
      jstart = (k - 6) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
      jstart4 = jstart3 + size;
      jstart5 = jstart4 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_x6,res_y1,res_y2,res_y3,res_y4,res_y5,res_y6) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
         res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
         res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
         res_x4 += hypre_conj(z_data[jstart3 + i]) * x_data[i];
         res_y4 += hypre_conj(z_data[jstart3 + i]) * y_data[i];
         res_x5 += hypre_conj(z_data[jstart4 + i]) * x_data[i];
         res_y5 += hypre_conj(z_data[jstart4 + i]) * y_data[i];
         res_x6 += hypre_conj(z_data[jstart5 + i]) * x_data[i];
         res_y6 += hypre_conj(z_data[jstart5 + i]) * y_data[i];
      }
      result_x[k - 6] = res_x1;
      result_x[k - 5] = res_x2;
      result_x[k - 4] = res_x3;
      result_x[k - 3] = res_x4;
      result_x[k - 2] = res_x5;
      result_x[k - 1] = res_x6;
      result_y[k - 6] = res_y1;
      result_y[k - 5] = res_y2;
      result_y[k - 4] = res_y3;
      result_y[k - 3] = res_y4;
      result_y[k - 2] = res_y5;
      result_y[k - 1] = res_y6;
   }
   else if (restk == 7)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_x4 = 0;
      res_x5 = 0;
      res_x6 = 0;
      res_x7 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      res_y4 = 0;
      res_y5 = 0;
      res_y6 = 0;
      res_y7 = 0;
      jstart = (k - 7) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
      jstart3 = jstart2 + size;
      jstart4 = jstart3 + size;
      jstart5 = jstart4 + size;
      jstart6 = jstart5 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_x6,res_x7,res_y1,res_y2,res_y3,res_y4,res_y5,res_y6,res_y7) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
         res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
         res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
         res_x4 += hypre_conj(z_data[jstart3 + i]) * x_data[i];
         res_y4 += hypre_conj(z_data[jstart3 + i]) * y_data[i];
         res_x5 += hypre_conj(z_data[jstart4 + i]) * x_data[i];
         res_y5 += hypre_conj(z_data[jstart4 + i]) * y_data[i];
         res_x6 += hypre_conj(z_data[jstart5 + i]) * x_data[i];
         res_y6 += hypre_conj(z_data[jstart5 + i]) * y_data[i];
         res_x7 += hypre_conj(z_data[jstart6 + i]) * x_data[i];
         res_y7 += hypre_conj(z_data[jstart6 + i]) * y_data[i];
      }
      result_x[k - 7] = res_x1;
      result_x[k - 6] = res_x2;
      result_x[k - 5] = res_x3;
      result_x[k - 4] = res_x4;
      result_x[k - 3] = res_x5;
      result_x[k - 2] = res_x6;
      result_x[k - 1] = res_x7;
      result_y[k - 7] = res_y1;
      result_y[k - 6] = res_y2;
      result_y[k - 5] = res_y3;
      result_y[k - 4] = res_y4;
      result_y[k - 3] = res_y5;
      result_y[k - 2] = res_y6;
      result_y[k - 1] = res_y7;
   }


   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassDotpTwo4
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassDotpTwo4( hypre_Vector *x, hypre_Vector *y,
                                       hypre_Vector **z, HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y)
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex *z_data = hypre_VectorData(z[0]);
   HYPRE_Int      size   = hypre_VectorSize(x);

   HYPRE_Int      i, j, restk;
   HYPRE_Real res_x1;
   HYPRE_Real res_x2;
   HYPRE_Real res_x3;
   HYPRE_Real res_x4;
   HYPRE_Real res_y1;
   HYPRE_Real res_y2;
   HYPRE_Real res_y3;
   HYPRE_Real res_y4;
   HYPRE_Int jstart;
   HYPRE_Int jstart1;
   HYPRE_Int jstart2;
   HYPRE_Int jstart3;

   restk = (k - (k / 4 * 4));

   if (k > 3)
   {
      for (j = 0; j < k - 3; j += 4)
      {
         res_x1 = 0;
         res_x2 = 0;
         res_x3 = 0;
         res_x4 = 0;
         res_y1 = 0;
         res_y2 = 0;
         res_y3 = 0;
         res_y4 = 0;
         jstart = j * size;
         jstart1 = jstart + size;
         jstart2 = jstart1 + size;
         jstart3 = jstart2 + size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_y1,res_y2,res_y3,res_y4) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
            res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
            res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
            res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
            res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
            res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
            res_x4 += hypre_conj(z_data[jstart3 + i]) * x_data[i];
            res_y4 += hypre_conj(z_data[jstart3 + i]) * y_data[i];
         }
         result_x[j] = res_x1;
         result_x[j + 1] = res_x2;
         result_x[j + 2] = res_x3;
         result_x[j + 3] = res_x4;
         result_y[j] = res_y1;
         result_y[j + 1] = res_y2;
         result_y[j + 2] = res_y3;
         result_y[j + 3] = res_y4;
      }
   }
   if (restk == 1)
   {
      res_x1 = 0;
      res_y1 = 0;
      jstart = (k - 1) * size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_y1) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
      }
      result_x[k - 1] = res_x1;
      result_y[k - 1] = res_y1;
   }
   else if (restk == 2)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_y1 = 0;
      res_y2 = 0;
      jstart = (k - 2) * size;
      jstart1 = jstart + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_y1,res_y2) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
      }
      result_x[k - 2] = res_x1;
      result_x[k - 1] = res_x2;
      result_y[k - 2] = res_y1;
      result_y[k - 1] = res_y2;
   }
   else if (restk == 3)
   {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      jstart = (k - 3) * size;
      jstart1 = jstart + size;
      jstart2 = jstart1 + size;
#if defined(HYPRE_USING_OPENMP)
      #pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_y1,res_y2,res_y3) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
         res_x1 += hypre_conj(z_data[jstart + i]) * x_data[i];
         res_y1 += hypre_conj(z_data[jstart + i]) * y_data[i];
         res_x2 += hypre_conj(z_data[jstart1 + i]) * x_data[i];
         res_y2 += hypre_conj(z_data[jstart1 + i]) * y_data[i];
         res_x3 += hypre_conj(z_data[jstart2 + i]) * x_data[i];
         res_y3 += hypre_conj(z_data[jstart2 + i]) * y_data[i];
      }
      result_x[k - 3] = res_x1;
      result_x[k - 2] = res_x2;
      result_x[k - 1] = res_x3;
      result_y[k - 3] = res_y1;
      result_y[k - 2] = res_y2;
      result_y[k - 1] = res_y3;
   }


   return hypre_error_flag;
}

HYPRE_Int hypre_SeqVectorMassInnerProd( hypre_Vector *x,
                                        hypre_Vector **y, HYPRE_Int k, HYPRE_Int unroll, HYPRE_Real *result)
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y[0]);
   HYPRE_Int      size   = hypre_VectorSize(x)*hypre_VectorNumVectors(x);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#if defined(HYPRE_USING_CUBLAS)

   HYPRE_Real * d_result = hypre_CTAlloc(HYPRE_Real,k,hypre_VectorMemoryLocation(x));

   double alpha = 1.0, beta = 0.0;
   HYPRE_CUBLAS_CALL( cublasDgemv(hypre_HandleCublasHandle(hypre_handle()), CUBLAS_OP_T, size, k, &alpha,
                                                           y_data, size, x_data, 1, &beta, d_result, 1));

   hypre_TMemcpy(result, d_result, HYPRE_Real, k, HYPRE_MEMORY_HOST, hypre_VectorMemoryLocation(x));
   hypre_TFree(d_result,hypre_VectorMemoryLocation(x));

#else

   hypreDevice_MassInnerProd(k, size, x_data, y_data, result);

#endif


#else

   HYPRE_Real res;
   HYPRE_Int      i, j, jstart;

   if (unroll == 8)
   {
      hypre_SeqVectorMassInnerProd8(x, y, k, result);
      return hypre_error_flag;
   }
   else if (unroll == 4)
   {
      hypre_SeqVectorMassInnerProd4(x, y, k, result);
      return hypre_error_flag;
   }
   else
   {
      for (j = 0; j < k; j++)
      {
         res = 0;
         jstart = j * size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) reduction(+:res) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            res += hypre_conj(y_data[jstart + i]) * x_data[i];
         }
         result[j] = res;
      }
   }

#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassDotpTwo
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SeqVectorMassDotpTwo( hypre_Vector *x, hypre_Vector *y,
                                      hypre_Vector **z, HYPRE_Int k,  HYPRE_Int unroll,
                                      HYPRE_Real *result_x, HYPRE_Real *result_y)
{
   HYPRE_Complex *x_data = hypre_VectorData(x);
   HYPRE_Complex *y_data = hypre_VectorData(y);
   HYPRE_Complex *z_data = hypre_VectorData(z[0]);
   HYPRE_Int      size   = hypre_VectorSize(x);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

   hypreDevice_MassDotpTwo(k, size, x_data, y_data, z_data, result_x, result_y);

#else

   HYPRE_Real res_x, res_y;

   HYPRE_Int      i, j, jstart;

   if (unroll == 8)
   {
      hypre_SeqVectorMassDotpTwo8(x, y, z, k, result_x, result_y);
      return hypre_error_flag;
   }
   else if (unroll == 4)
   {
      hypre_SeqVectorMassDotpTwo4(x, y, z, k, result_x, result_y);
      return hypre_error_flag;
   }
   else
   {
      for (j = 0; j < k; j++)
      {
         res_x = 0; //result_x[j];
         res_y = 0; //result_y[j];
         jstart = j * size;
#if defined(HYPRE_USING_OPENMP)
         #pragma omp parallel for private(i) reduction(+:res_x,res_y) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            res_x += hypre_conj(z_data[jstart + i]) * x_data[i];
            res_y += hypre_conj(z_data[jstart + i]) * y_data[i];
         }
         result_x[j] = res_x;
         result_y[j] = res_y;
      }
   }

#endif

   return hypre_error_flag;
}

