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

#include "_hypre_struct_mv.h"

#if defined(HYPRE_MIXED_PRECISION)

/******************************************************************************
 *
 * Member functions for hypre_StructVector class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * Mixed precision hypre_StructVectorCopy -- -- TODO: Needs GPU support - DOK
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructVectorCopy_mp( hypre_StructVector *x,
                           hypre_StructVector *y )
{
   /* determine type of output vector data  ==> Precision of y. */
   HYPRE_Precision precision_y = hypre_StructVectorPrecision (y);

   /* Generic pointer type */
   void               *xp, *yp;

   HYPRE_Int           i, size;

   /* Call standard vector copy if precisions match. */
   if (precision_y == hypre_StructVectorPrecision (x))
   {
      return HYPRE_StructVectorCopy_pre(precision_y, (HYPRE_StructVector)x, (HYPRE_StructVector)y);
   }

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   size = hypre_StructVectorDataSize(x);

   /* Implicit conversion to generic data type (void pointer) */
   xp = hypre_StructVectorData(x);
   yp = hypre_StructVectorData(y);

   switch (hypre_StructVectorPrecision (x))
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

#endif

