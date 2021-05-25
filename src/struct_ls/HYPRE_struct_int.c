/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "temp_multivector.h"
#include "_hypre_struct_mv.hpp"

HYPRE_Int
hypre_StructVectorSetRandomValues( hypre_StructVector *vector,
                                   HYPRE_Int seed )
{
   hypre_Box          *v_data_box;

   HYPRE_Real         *vp;

   hypre_BoxArray     *boxes;
   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   HYPRE_Int           i;

   /*-----------------------------------------------------------------------
    * Set the vector coefficients
    *-----------------------------------------------------------------------*/

//   srand( seed );
   hypre_SeedRand(seed);

   hypre_SetIndex3(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(vector));
   hypre_ForBoxI(i, boxes)
   {
      box      = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      v_data_box =
         hypre_BoxArrayBox(hypre_StructVectorDataSpace(vector), i);
      vp = hypre_StructVectorBoxData(vector, i);

      hypre_BoxGetSize(box, loop_size);

      /* RL TODO: generate on host and copy to device. FIX? */
#if defined(HYPRE_USING_GPU)
      HYPRE_Int loop_n = 1, ii;
      for (ii = 0; ii < hypre_StructVectorNDim(vector); ii++)
      {
         loop_n *= loop_size[ii];
      }

      HYPRE_Real *rand_host   = hypre_TAlloc(HYPRE_Real, loop_n, HYPRE_MEMORY_HOST);
      HYPRE_Real *rand_device = hypre_TAlloc(HYPRE_Real, loop_n, HYPRE_MEMORY_DEVICE);

      ii = 0;
      hypre_SerialBoxLoop0Begin(hypre_StructVectorNDim(vector),loop_size)
      {
         rand_host[ii++] = 2.0*hypre_Rand() - 1.0;
      }
      hypre_SerialBoxLoop0End()
      hypre_TMemcpy(rand_device, rand_host, HYPRE_Real, loop_n,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);
#endif

#define DEVICE_VAR is_device_ptr(vp, rand_device)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(vector), loop_size,
                          v_data_box, start, unit_stride, vi);
      {
#if defined(HYPRE_USING_GPU)
         vp[vi] = rand_device[idx];
#else
         vp[vi] = 2.0*hypre_Rand() - 1.0;
#endif
      }
      hypre_BoxLoop1End(vi);
#undef DEVICE_VAR

#if defined(HYPRE_USING_GPU)
      hypre_TFree(rand_device, HYPRE_MEMORY_DEVICE);
      hypre_TFree(rand_host, HYPRE_MEMORY_HOST);
#endif
   }

   return hypre_error_flag;
}

HYPRE_Int
hypre_StructSetRandomValues( void* v, HYPRE_Int seed ) {

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
