/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre seq_mv mixed-precision interface
 *
 *****************************************************************************/

#include "_hypre_seq_mv.h"

#if defined(HYPRE_MIXED_PRECISION)

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Mixed precision hypre_SeqVectorCopy -- TODO: Needs GPU support - DOK
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_SeqVectorCopy_mp( hypre_Vector *x,
                        hypre_Vector *y )
{
   /*
   #ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
   #endif

      hypre_GpuProfilingPushRange("SeqVectorCopy");
   */
   /* determine type of output vector data  ==> Precision of y. */
   HYPRE_Precision precision_y = hypre_VectorPrecision (y);

   HYPRE_Int      i;

   /* Generic pointer type */
   void               *xp, *yp;

   /* Call standard vector copy if precisions match. */
   if (precision_y == hypre_VectorPrecision (x))
   {
      return HYPRE_VectorCopy_pre(precision_y, (HYPRE_Vector)x, (HYPRE_Vector)y);
   }

   size_t size = hypre_min(hypre_VectorSize(x), hypre_VectorSize(y)) * hypre_VectorNumVectors(x);

   /* Implicit conversion to generic data type (void pointer) */
   xp = hypre_VectorData(x);
   yp = hypre_VectorData(y);
   
   switch (hypre_VectorPrecision (x))
   {
      case HYPRE_REAL_SINGLE:
         switch (precision_y)
         {
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < size; i++)
               {
                  ((hypre_double *)yp)[i] = (hypre_double)((hypre_float *)xp)[i];
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < size; i++)
               {
                  ((hypre_long_double *)yp)[i] = (hypre_long_double)((hypre_float *)xp)[i];
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_DOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < size; i++)
               {
                  ((hypre_float *)yp)[i] = (hypre_float)((hypre_double *)xp)[i];
               }
               break;
            case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < size; i++)
               {
                  ((hypre_long_double *)yp)[i] = (hypre_long_double)((hypre_double *)xp)[i];
               }
               break;
            default:
               break;
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
         switch (precision_y)
         {
            case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < size; i++)
               {
                  ((hypre_float *)yp)[i] = (hypre_float)((hypre_long_double *)xp)[i];
               }
               break;            
            case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
            #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
               for (i = 0; i < size; i++)
               {
                  ((hypre_double *)yp)[i] = (hypre_double)((hypre_long_double *)xp)[i];
               }
               break;
            default:
               break;
         }
      default:
         hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for Vector Copy!\n");
         break;
   }

   /*
   #ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
   #endif
      hypre_GpuProfilingPopRange();
   */
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Mixed-precision hypre_SeqVectorAxpy -- TODO: Needs GPU support - DOK
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorAxpy_mp( hypre_double alpha,
                        hypre_Vector *x,
                        hypre_Vector *y     )
{
   /*
   #ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
   #endif
   */
   /* determine type of output vector data  ==> Precision of y. */
   HYPRE_Precision precision = hypre_VectorPrecision (y);

   void               *xp, *yp;
   
   HYPRE_Int      size   = hypre_VectorSize(x);
   HYPRE_Int      i;

   size *= hypre_VectorNumVectors(x);

   /* Implicit conversion to generic data type (void pointer) */
   xp = hypre_VectorData(x);
   yp = hypre_VectorData(y);   

   switch (precision)
   {
      case HYPRE_REAL_SINGLE:
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         { 
            ((hypre_float *)yp)[i] += (hypre_float)(alpha * ((hypre_double *)xp)[i]);
         }
         break;
      case HYPRE_REAL_DOUBLE:
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            ((hypre_double *)yp)[i] += (hypre_double)(alpha * ((hypre_float *)xp)[i]);
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
         for (i = 0; i < size; i++)
         {
            ((hypre_long_double *)yp)[i] += (hypre_long_double)(alpha * ((hypre_double *)xp)[i]);
         }
         break;
      default:
         hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for Vector Axpy!\n");
   }
   /*
   #ifdef HYPRE_PROFILE
      hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
   #endif
   */
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Convert precision in a mixed precision vector
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SeqVectorConvert_mp (hypre_Vector *v,
                           HYPRE_Precision new_precision)
{
   HYPRE_Precision precision = hypre_VectorPrecision (v);
   void *data = hypre_VectorData(v);
   void *data_mp = NULL;
   HYPRE_Int size = hypre_VectorSize(v);
   HYPRE_MemoryLocation memory_location = hypre_VectorMemoryLocation(v);
   HYPRE_Int i;

   if (new_precision == precision)
      return hypre_error_flag;
   else
   {
      switch (precision)
      {
         case HYPRE_REAL_SINGLE:
         {
            switch (new_precision)
            {
               case HYPRE_REAL_DOUBLE:
               {
                  data_mp = (hypre_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_double *)data_mp)[i] = (hypre_double) ((hypre_float *) data)[i];
               }
               break;
               case HYPRE_REAL_LONGDOUBLE:
               {
                  data_mp = (hypre_long_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_long_double *)data_mp)[i] = (hypre_long_double) ((hypre_float *) data)[i];
               }
               break;
               default:
                  hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
            }
         }
         break;
         case HYPRE_REAL_DOUBLE:
         {
            switch (new_precision)
            {
               case HYPRE_REAL_SINGLE:
               {
                  data_mp = (hypre_float *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_float *)data_mp)[i] = (hypre_float) ((hypre_double *) data)[i];
               }
               break;
               case HYPRE_REAL_LONGDOUBLE:
               {
                  data_mp = (hypre_long_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_long_double *)data_mp)[i] = (hypre_long_double) ((hypre_double *) data)[i];
               }
               break;
               default:
                  hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
            }
         }
         break;
         case HYPRE_REAL_LONGDOUBLE:
         {
            switch (new_precision)
            {
               case HYPRE_REAL_SINGLE:
               {
                  data_mp = (hypre_float *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_float *)data_mp)[i] = (hypre_float) ((hypre_long_double *) data)[i];
               }
               break;
               case HYPRE_REAL_DOUBLE:
               {
                  data_mp = (hypre_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_double *)data_mp)[i] = (hypre_double) ((hypre_long_double *) data)[i];
               }
               break;
               default:
                  hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
            }
         }
         break;
         default:
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
      }
      hypre_Free(data, memory_location);
      hypre_VectorData(v) = data_mp;
      hypre_VectorPrecision(v) = new_precision;
   }
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * Convert precision in a mixed precision matrix
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixConvert_mp (hypre_CSRMatrix *A,
                           HYPRE_Precision new_precision)
{
   HYPRE_Precision precision = hypre_CSRMatrixPrecision (A);
   void *data = hypre_CSRMatrixData(A);
   void *data_mp = NULL;
   HYPRE_Int size = hypre_CSRMatrixI(A)[hypre_CSRMatrixNumRows(A)];
   HYPRE_MemoryLocation memory_location = hypre_CSRMatrixMemoryLocation(A);
   HYPRE_Int i;

   if (new_precision == precision)
      return hypre_error_flag;
   else
   {
      switch (precision)
      {
         case HYPRE_REAL_SINGLE:
         {
            switch (new_precision)
            {
               case HYPRE_REAL_DOUBLE:
               {
                  data_mp = (hypre_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_double *)data_mp)[i] = (hypre_double) ((hypre_float *) data)[i];
               }
               break;
               case HYPRE_REAL_LONGDOUBLE:
               {
                  data_mp = (hypre_long_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_long_double *)data_mp)[i] = (hypre_long_double) ((hypre_float *) data)[i];
               }
               break;
               default:
                  hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
            }
         }
         break;
         case HYPRE_REAL_DOUBLE:
         {
            switch (new_precision)
            {
               case HYPRE_REAL_SINGLE:
               {
                  data_mp = (hypre_float *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_float *)data_mp)[i] = (hypre_float) ((hypre_double *) data)[i];
               }
               break;
               case HYPRE_REAL_LONGDOUBLE:
               {
                  data_mp = (hypre_long_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_long_double *)data_mp)[i] = (hypre_long_double) ((hypre_double *) data)[i];
               }
               break;
               default:
                  hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
            }
         }
         break;
         case HYPRE_REAL_LONGDOUBLE:
         {
            switch (new_precision)
            {
               case HYPRE_REAL_SINGLE:
               {
                  data_mp = (hypre_float *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_float *)data_mp)[i] = (hypre_float) ((hypre_long_double *) data)[i];
               }
               break;
               case HYPRE_REAL_DOUBLE:
               {
                  data_mp = (hypre_double *) hypre_CAlloc ((size_t)size, (size_t)sizeof(hypre_double), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_double *)data_mp)[i] = (hypre_double) ((hypre_long_double *) data)[i];
               }
               break;
               default:
                  hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
            }
         }
         break;
         default:
            hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type!\n");
      }
      hypre_Free(data, memory_location);
      hypre_CSRMatrixData(A) = data_mp;
      hypre_CSRMatrixPrecision(A) = new_precision;
   }
   return hypre_error_flag;
}

#endif
