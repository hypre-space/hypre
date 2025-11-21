/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured inner product routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * hypre_StructInnerProdLocal
 *
 * The vectors x and y may have different base grids, but the grid boxes for
 * each vector (defined by grid, stride, nboxes, boxnums) must be the same.
 * Only nboxes is checked, the rest is assumed to be true.
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_StructInnerProdLocal( hypre_StructVector *x,
                            hypre_StructVector *y )
{
   HYPRE_Int        ndim = hypre_StructVectorNDim(x);

   HYPRE_Real       result = 0.0;

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;

   HYPRE_Int        nboxes;
   hypre_Box       *loop_box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      ustride;

   HYPRE_Int        i;

   nboxes = hypre_StructVectorNBoxes(x);

   /* Return if nboxes is not the same for x and y */
   if (nboxes != hypre_StructVectorNBoxes(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "StructInnerProd: nboxes for x and y do not match!");

      return hypre_error_flag;
   }

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(x, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      x_data_box = hypre_StructVectorGridDataBox(x, i);
      y_data_box = hypre_StructVectorGridDataBox(y, i);

      xp = hypre_StructVectorGridData(x, i);
      yp = hypre_StructVectorGridData(y, i);

      hypre_BoxGetSize(loop_box, loop_size);

#if defined(HYPRE_USING_KOKKOS) || defined(HYPRE_USING_SYCL)
      HYPRE_Real box_sum = 0.0;
#elif defined(HYPRE_USING_RAJA)
      ReduceSum<hypre_raja_reduce_policy, HYPRE_Real> box_sum(0.0);
#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      ReduceSum<HYPRE_Real> box_sum(0.0);
#else
      HYPRE_Real box_sum = 0.0;
#endif

#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
#define HYPRE_BOX_REDUCTION map(tofrom: box_sum) reduction(+:box_sum)
#else
#define HYPRE_BOX_REDUCTION reduction(+:box_sum)
#endif

#define DEVICE_VAR is_device_ptr(yp,xp)
      hypre_BoxLoop2ReductionBegin(ndim, loop_size,
                                   x_data_box, start, ustride, xi,
                                   y_data_box, start, ustride, yi,
                                   box_sum)
      {
         HYPRE_Real tmp = xp[xi] * hypre_conj(yp[yi]);
         box_sum += tmp;
      }
      hypre_BoxLoop2ReductionEnd(xi, yi, box_sum);

      result += (HYPRE_Real) box_sum;
   }

   hypre_BoxDestroy(loop_box);

   return result;
}

/*--------------------------------------------------------------------------
 * hypre_StructInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_StructInnerProd( hypre_StructVector *x,
                       hypre_StructVector *y )
{
   HYPRE_Real local_result;
   HYPRE_Real global_result;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("InnerProd");

   local_result = hypre_StructInnerProdLocal(x, y);

   hypre_MPI_Allreduce(&local_result, &global_result, 1, HYPRE_MPI_REAL, hypre_MPI_SUM,
                       hypre_StructVectorComm(x));

   hypre_IncFLOPCount(2 * hypre_StructVectorGlobalSize(x));

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return global_result;
}
