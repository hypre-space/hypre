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
   HYPRE_Precision precision = hypre_StructVectorPrecision (y);

   /* Generic pointer type */
   void               *xp, *yp;

   HYPRE_Int           i, size;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   size = hypre_StructVectorDataSize(x);

   /* Implicit conversion to generic data type (void pointer) */
   xp = hypre_StructVectorData(x);
   yp = hypre_StructVectorData(y);

   switch (precision)
   {
      case HYPRE_REAL_SINGLE:
         for (i = 0; i < size; i++)
         {
            ((hypre_float *)yp)[i] = (hypre_float)((hypre_double *)xp)[i];
         }
         break;
      case HYPRE_REAL_DOUBLE:
         for (i = 0; i < size; i++)
         {
            ((hypre_double *)yp)[i] = (hypre_double)((hypre_float *)xp)[i];
         }
         break;
      case HYPRE_REAL_LONGDOUBLE:
         for (i = 0; i < size; i++)
         {
            ((hypre_long_double *)yp)[i] =
               (hypre_long_double)((hypre_double *)xp)[i];
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

#endif

