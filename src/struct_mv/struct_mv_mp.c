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
 * Mixed precision hypre_StructVectorCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructVectorCopy_mp( hypre_StructVector_mp *x,
                           hypre_StructVector_mp *y )
{
   /* determine type of output vector data  ==> Precision of y. */
   HYPRE_Precision precision = hypre_StructVectorPrecision (y);

   void               *xp, *yp;

   HYPRE_Int           i, size;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   size = hypre_StructVectorDataSize(x);
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

/*--------------------------------------------------------------------------
 * Convert precision in a mixed precision vector
 *--------------------------------------------------------------------------*/

/*HYPRE_Int
hypre_StructVectorConvert_mp (hypre_StructVector_mp *v,
                           HYPRE_Precision new_precision)
{
   HYPRE_Precision precision = hypre_StructVectorPrecision (v);
   hypre_Box          *v_data_box;
   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   void *vp;
   void *vp_mp = NULL;

   HYPRE_Int size = hypre_StructVectorDataSize(v);

   HYPRE_Int           i;

   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(v);

   hypre_SetIndex_flt(unit_stride, 1);

   boxes = hypre_StructGridBoxes( hypre_StructVectorGrid(v) );
   if (new_precision == precision)
      return hypre_error_flag;
   if (new_precision == HYPRE_REAL_DOUBLE)
   {
      vp_mp = (hypre_double *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_double), memory_location);
   }
   else if (new_precision == HYPRE_REAL_SINGLE)
   {
      vp_mp = (hypre_float *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
   }
   else if (new_precision == HYPRE_REAL_LONGDOUBLE)
   {
      vp_mp = (hypre_long_double *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
   }
      hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         v_data_box =
            hypre_BoxArrayBox(hypre_StructVectorDataSpace(v), i);
         vp = hypre_StructVectorBoxData(v, i);

         hypre_BoxGetSize(box, loop_size);

         switch (precision)
         {
            case HYPRE_REAL_SINGLE:
            {
               switch (new_precision)
               {
                  case HYPRE_REAL_DOUBLE:
                  {
            hypre_BoxLoop1Begin(hypre_StructVectorNDim(x), loop_size,
                                x_data_box, start, unit_stride, vi);
            {
               ((hypre_long_double *)(hypre_StructVectorBoxData(y, i))[vi] =
      (hypre_long_double)(hypre_double *)(hypre_StructVectorBoxData(x, i))[vi];
            }
            hypre_BoxLoop1End(vi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_double *)data_mp)[i] = (hypre_double) ((hypre_float *) data)[i];
               }
               break;
               case HYPRE_REAL_LONGDOUBLE:
               {
                  data_mp = (hypre_long_double *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
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
                  data_mp = (hypre_float *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_float *)data_mp)[i] = (hypre_float) ((hypre_double *) data)[i];
               }
               break;
               case HYPRE_REAL_LONGDOUBLE:
               {
                  data_mp = (hypre_long_double *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_long_double), memory_location);
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
                  data_mp = (hypre_float *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_float), memory_location);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
                  for (i = 0; i < size; i++)
                     ((hypre_float *)data_mp)[i] = (hypre_float) ((hypre_long_double *) data)[i];
               }
               break;
               case HYPRE_REAL_DOUBLE:
               {
                  data_mp = (hypre_double *) hypre_CAlloc_dbl ((size_t)size, (size_t)sizeof(hypre_double), memory_location);
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
      hypre_Free_dbl(data, memory_location);
      hypre_StructVectorData(v) = data_mp;
      hypre_StructVectorPrecision(v) = new_precision;
   }
   return hypre_error_flag;
}
*/

#endif

