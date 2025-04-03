/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "temp_multivector.h"
#include "_hypre_struct_mv.hpp"

HYPRE_Int
hypre_StructVectorSetRandomValues( hypre_StructVector *vector,
                                   HYPRE_Int           seed )
{
   hypre_Box           *v_data_box;
   HYPRE_Real          *vp;
   hypre_BoxArray      *boxes;
   hypre_Box           *box;
   hypre_Index          loop_size;
   hypre_IndexRef       start;
   hypre_Index          unit_stride;
   HYPRE_Int            i;
   HYPRE_Complex       *data            = hypre_StructVectorData(vector);
   HYPRE_Complex       *data_host       = NULL;
   HYPRE_Int            data_size       = hypre_StructVectorDataSize(vector);
   HYPRE_MemoryLocation memory_location = hypre_StructVectorMemoryLocation(vector);

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

   //   srand( seed );
   hypre_SeedRand(seed);

   hypre_SetIndex3(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));

   if (hypre_GetExecPolicy1(memory_location) == HYPRE_EXEC_DEVICE)
   {
      data_host = hypre_CTAlloc(HYPRE_Complex, data_size, HYPRE_MEMORY_HOST);
      hypre_StructVectorData(vector) = data_host;
   }

   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      v_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      vp = hypre_StructVectorBoxData(vector, i);

      hypre_BoxGetSize(box, loop_size);

      hypre_SerialBoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                                v_data_box, start, unit_stride, vi);
      {
         vp[vi] = 2.0 * hypre_Rand() - 1.0;
      }
      hypre_SerialBoxLoop1End(vi);
   }

   if (data_host)
   {
      hypre_TMemcpy(data, data_host, HYPRE_Complex, data_size, memory_location, HYPRE_MEMORY_HOST);
      hypre_StructVectorData(vector) = data;
      hypre_TFree(data_host, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_StructSetRandomValues( void* v, HYPRE_Int seed )
{

   return hypre_StructVectorSetRandomValues( (hypre_StructVector*)v, seed );
}
