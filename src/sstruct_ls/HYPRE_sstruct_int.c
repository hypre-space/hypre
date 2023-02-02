/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"


HYPRE_Int
hypre_SStructPVectorSetRandomValues( hypre_SStructPVector *pvector, HYPRE_Int seed )
{
   HYPRE_Int ierr = 0;
   HYPRE_Int           nvars = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector *svector;
   HYPRE_Int           var;

   hypre_SeedRand( seed );

   for (var = 0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
      seed = hypre_RandI();
      hypre_StructVectorSetRandomValues(svector, seed);
   }

   return ierr;
}

HYPRE_Int
hypre_SStructVectorSetRandomValues( hypre_SStructVector *vector, HYPRE_Int seed )
{
   HYPRE_Int ierr = 0;
   HYPRE_Int             nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   HYPRE_Int             part;

   hypre_SeedRand( seed );

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
      seed = hypre_RandI();
      hypre_SStructPVectorSetRandomValues(pvector, seed);
   }

   return ierr;
}

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
