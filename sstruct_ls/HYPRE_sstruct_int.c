#include "HYPRE_sstruct_int.h"

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
      hypre_SStructVectorSetRandomValues(svector, seed);
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
HYPRE_SStructSetupInterpreter( HYPRE_InterfaceInterpreter *i )
{
  i->CAlloc = hypre_CAlloc;
  i->Free = hypre_SStructKrylovFree;
  i->CommInfo = hypre_SStructKrylovCommInfo;

  i->CreateVector = hypre_SStructKrylovCreateVector;
  i->DestroyVector = hypre_SStructKrylovDestroyVector; 
  i->MatvecCreate = hypre_SStructKrylovMatvecCreate;
  i->Matvec = hypre_SStructKrylovMatvec; 
  i->MatvecDestroy = hypre_SStructKrylovMatvecDestroy;
  i->InnerProd = hypre_SStructKrylovInnerProd; 
  i->CopyVector = hypre_SStructKrylovCopyVector;
  i->ClearVector = hypre_SStructKrylovClearVector;
  i->SetRandomValues = hypre_SStructSetRandomValues;
  i->ScaleVector = hypre_SStructKrylovScaleVector;
  i->Axpy = hypre_SStructKrylovAxpy;
  i->PrintVector = NULL;
  i->ReadVector = NULL;

  i->CreateMultiVector = hypre_TempMultiVectorCreateFromSampleVector;
  i->CopyCreateMultiVector = hypre_TempMultiVectorCreateCopy;
  i->DestroyMultiVector = hypre_TempMultiVectorDestroy;

  i->MatMultiVecCreate = NULL;
  i->MatMultiVec = NULL;
  i->MatMultiVecDestroy = NULL;

  i->Width = hypre_TempMultiVectorWidth;
  i->Height = hypre_TempMultiVectorHeight;
  i->SetMask = hypre_TempMultiVectorSetMask;
  i->CopyMultiVector = hypre_TempMultiVectorCopy;
  i->ClearMultiVector = hypre_TempMultiVectorClear;
  i->SetRandomVectors = hypre_TempMultiVectorSetRandom;
  i->MultiInnerProd = hypre_TempMultiVectorByMultiVector;
  i->MultiInnerProdDiag = hypre_TempMultiVectorByMultiVectorDiag;
  i->MultiVecMat = hypre_TempMultiVectorByMatrix;
  i->MultiVecMatDiag = hypre_TempMultiVectorByDiagonal;
  i->MultiAxpy = hypre_TempMultiVectorAxpy;
  i->MultiXapy = hypre_TempMultiVectorXapy;
  i->Eval = hypre_TempMultiVectorEval;
  i->PrintMultiVector = hypre_TempMultiVectorPrint;
  i->ReadMultiVector = hypre_TempMultiVectorRead;

  return 0;
}

