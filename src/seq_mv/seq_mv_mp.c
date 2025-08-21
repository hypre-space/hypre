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

#include "seq_mv.h"

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
   HYPRE_Precision precision = hypre_VectorPrecision (y);

   HYPRE_Int      i;

   /* Generic pointer type */
   void               *xp, *yp;

   size_t size = hypre_min(hypre_VectorSize(x), hypre_VectorSize(y)) * hypre_VectorNumVectors(x);

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
            ((hypre_float *)yp)[i] = (hypre_float)((hypre_double *)xp)[i];
         }
         break;
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
            ((hypre_long_double *)yp)[i] = (hypre_long_double)((hypre_double *)xp)[i];
         }
         break;
      default:
         hypre_error_w_msg_mp(HYPRE_ERROR_GENERIC, "Error: Undefined precision type for Vector Copy!\n");
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

#endif
