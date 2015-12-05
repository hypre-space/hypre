/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/




#include "HYPRE_struct_int.h"
#include "HYPRE_sstruct_int.h"
#include "_hypre_sstruct_ls.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
#include "temp_multivector.h"


int 
hypre_SStructPVectorSetRandomValues( hypre_SStructPVector *pvector, int seed )
{
   int ierr = 0;
   int                 nvars = hypre_SStructPVectorNVars(pvector);
   hypre_StructVector *svector;
   int                 var;

   srand( seed );

   for (var = 0; var < nvars; var++)
   {
      svector = hypre_SStructPVectorSVector(pvector, var);
	  seed = rand();
      hypre_StructVectorSetRandomValues(svector, seed);
   }

   return ierr;
}

int 
hypre_SStructVectorSetRandomValues( hypre_SStructVector *vector, int seed )
{
   int ierr = 0;
   int                   nparts = hypre_SStructVectorNParts(vector);
   hypre_SStructPVector *pvector;
   int                   part;

   srand( seed );

   for (part = 0; part < nparts; part++)
   {
      pvector = hypre_SStructVectorPVector(vector, part);
	  seed = rand();
      hypre_SStructPVectorSetRandomValues(pvector, seed);
   }

   return ierr;
}

int
hypre_SStructSetRandomValues( void* v, int seed ) {

  return hypre_SStructVectorSetRandomValues( (hypre_SStructVector*)v, seed );
}

int
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

int
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
