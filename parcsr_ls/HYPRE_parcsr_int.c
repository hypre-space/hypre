#include "HYPRE_parcsr_int.h"

#include "temp_multivector.h"

int
hypre_ParSetRandomValues( void* v, int seed ) {

  HYPRE_ParVectorSetRandomValues( (HYPRE_ParVector)v, seed );
  return 0;
}

int
hypre_ParPrintVector( void* v, const char* file ) {

  return hypre_ParVectorPrint( (hypre_ParVector*)v, file );
}

void*
hypre_ParReadVector( MPI_Comm comm, const char* file ) {

  return (void*)hypre_ParVectorRead( comm, file );
}

int
HYPRE_ParCSRSetupInterpreter( HYPRE_InterfaceInterpreter *i )
{
  return HYPRE_TempParCSRSetupInterpreter( i );
}

/* The function below is a temporary one that fills the multivector 
   part of the HYPRE_InterfaceInterpreter structure with pointers 
   that come from the temporary implementation of the multivector 
   (cf. temp_multivector.h). 
   It must be eventually replaced with a function that
   provides the respective pointers to properly implemented 
   parcsr multivector functions */

int
HYPRE_TempParCSRSetupInterpreter( HYPRE_InterfaceInterpreter *i )
{
  i->CAlloc = hypre_CAlloc;
  i->Free = hypre_ParKrylovFree;
  i->CommInfo = hypre_ParKrylovCommInfo;

  /* Vector part */

  i->CreateVector = hypre_ParKrylovCreateVector;
  i->DestroyVector = hypre_ParKrylovDestroyVector; 
  i->MatvecCreate = hypre_ParKrylovMatvecCreate;
  i->Matvec = hypre_ParKrylovMatvec; 
  i->MatvecDestroy = hypre_ParKrylovMatvecDestroy;
  i->InnerProd = hypre_ParKrylovInnerProd; 
  i->CopyVector = hypre_ParKrylovCopyVector;
  i->ClearVector = hypre_ParKrylovClearVector;
  i->SetRandomValues = hypre_ParSetRandomValues;
  i->ScaleVector = hypre_ParKrylovScaleVector;
  i->Axpy = hypre_ParKrylovAxpy;
  i->PrintVector = hypre_ParPrintVector;
  i->ReadVector = hypre_ParReadVector;

  /* Multivector part */

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
