#include "HYPRE_parcsr_int.h"

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
  i->CAlloc = hypre_CAlloc;
  i->Free = hypre_ParKrylovFree;
  i->CommInfo = hypre_ParKrylovCommInfo;
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

  return 0;
}
