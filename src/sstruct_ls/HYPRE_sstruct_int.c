/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "_hypre_lobpcg_interpreter.h"
#include "_hypre_lobpcg_temp_multivector.h"


/* TODO: This does not belong here */
HYPRE_Int
hypre_SStructSetRandomValues( void* v, HYPRE_Int seed )
{

   return hypre_SStructVectorSetRandomValues( (hypre_SStructVector*)v, seed );
}

HYPRE_Int
HYPRE_SStructSetupInterpreter( mv_InterfaceInterpreter *i )
{
   i->CreateVector = hypre_SStructKrylovCreateVector;
   i->DestroyVector = hypre_SStructKrylovDestroyVector;
   i->InnerProd = hypre_SStructKrylovInnerProd;
   i->CopyVector = hypre_SStructKrylovCopyVector;
   i->ClearVector = hypre_SStructKrylovClearVector;
   i->SetRandomValues = hypre_SStructSetRandomValues;
   i->ScaleVector = hypre_SStructKrylovScaleVector;
   i->Axpy = hypre_SStructKrylovAxpy;

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

   return 0;
}

HYPRE_Int
HYPRE_SStructSetupMatvec(HYPRE_MatvecFunctions * mv)
{
   mv->MatvecCreate = hypre_SStructKrylovMatvecCreate;
   mv->Matvec = hypre_SStructKrylovMatvec;
   mv->MatvecDestroy = hypre_SStructKrylovMatvecDestroy;

   mv->MatMultiVecCreate = NULL;
   mv->MatMultiVec = NULL;
   mv->MatMultiVecDestroy = NULL;

   return 0;
}
