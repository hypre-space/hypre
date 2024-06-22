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
 * hypre_StructInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_StructInnerProd( hypre_StructVector *x,
                       hypre_StructVector *y )
{
   HYPRE_Real       final_innerprod_result;
   HYPRE_Real       process_result;

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;

   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;

   HYPRE_Int        ndim = hypre_StructVectorNDim(x);
   HYPRE_Int        i;

#if 0 //defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
   const HYPRE_Int  data_location = hypre_StructGridDataLocation(hypre_StructVectorGrid(y));
#endif

   HYPRE_Real       local_result = 0.0;

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(box, loop_size);

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
                                   x_data_box, start, unit_stride, xi,
                                   y_data_box, start, unit_stride, yi,
                                   box_sum)
      {
         HYPRE_Real tmp = xp[xi] * hypre_conj(yp[yi]);
         box_sum += tmp;
      }
      hypre_BoxLoop2ReductionEnd(xi, yi, box_sum);

      local_result += (HYPRE_Real) box_sum;
   }

   process_result = (HYPRE_Real) local_result;

   hypre_MPI_Allreduce(&process_result, &final_innerprod_result, 1,
                       HYPRE_MPI_REAL, hypre_MPI_SUM, hypre_StructVectorComm(x));

   hypre_IncFLOPCount(2 * hypre_StructVectorGlobalSize(x));

   return final_innerprod_result;
}
