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

HYPRE_Int
HYPRE_StructSetupInterpreter( mv_InterfaceInterpreter *i )
{
   i->CreateVector = hypre_StructKrylovCreateVector;
   i->DestroyVector = hypre_StructKrylovDestroyVector;
   i->InnerProd = hypre_StructKrylovInnerProd;
   i->CopyVector = hypre_StructKrylovCopyVector;
   i->ClearVector = hypre_StructKrylovClearVector;
   i->SetRandomValues = hypre_StructSetRandomValues;
   i->ScaleVector = hypre_StructKrylovScaleVector;
   i->Axpy = hypre_StructKrylovAxpy;

   i->CreateMultiVector = mv_TempMultiVectorCreateFromSampleVector;
   i->CopyCreateMultiVector = mv_TempMultiVectorCreateCopy;
   i->DestroyMultiVector = mv_TempMultiVectorDestroy;

   i->Width = mv_TempMultiVectorWidth;
   i->Height = mv_TempMultiVectorHeight;
   i->SetMask = mv_TempMultiVectorSetMask;
   i->CopyMultiVector = mv_TempMultiVectorCopy;
   i->ClearMultiVector = mv_TempMultiVectorClear;
   i->SetRandomVectors = mv_TempMultiVectorSetRandom;
   i->MultiInnerProd = mv_TempMultiVectorByMultiVector;
   i->MultiInnerProdDiag = mv_TempMultiVectorByMultiVectorDiag;
   i->MultiVecMat = mv_TempMultiVectorByMatrix;
   i->MultiVecMatDiag = mv_TempMultiVectorByDiagonal;
   i->MultiAxpy = mv_TempMultiVectorAxpy;
   i->MultiXapy = mv_TempMultiVectorXapy;
   i->Eval = mv_TempMultiVectorEval;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_StructSetupMatvec(HYPRE_MatvecFunctions * mv)
{
   mv->MatvecCreate = hypre_StructKrylovMatvecCreate;
   mv->Matvec = hypre_StructKrylovMatvec;
   mv->MatvecDestroy = hypre_StructKrylovMatvecDestroy;

   mv->MatMultiVecCreate = NULL;
   mv->MatMultiVec = NULL;
   mv->MatMultiVecDestroy = NULL;

   return hypre_error_flag;
}
