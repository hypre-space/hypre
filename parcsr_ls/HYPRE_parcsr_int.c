#include <assert.h>

#include "parcsr_ls.h"
#include "HYPRE_parcsr_int.h"
#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"
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

int hypre_ParVectorSize(void * x)
{
  return 0;
}

int
hypre_ParCSRMultiVectorPrint( void* x_, const char* fileName ) {

  int i, ierr;
  mv_TempMultiVector* x;
  char fullName[128];
  
  x = (mv_TempMultiVector*)x_;
  assert( x != NULL );

  ierr = 0;
  for ( i = 0; i < x->numVectors; i++ ) {
    sprintf( fullName, "%s.%d", fileName, i ); 
    ierr = ierr || 
      hypre_ParPrintVector( x->vector[i], fullName );
  }
  return ierr;
}
							
void* 
hypre_ParCSRMultiVectorRead( MPI_Comm comm, void* ii_, const char* fileName ) {

  int i, n, id;
  FILE* fp;
  char fullName[128];
  mv_TempMultiVector* x;
  mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;
  
  MPI_Comm_rank( comm, &id );
  
  n = 0;
  do {
    sprintf( fullName, "%s.%d.%d", fileName, n, id ); 
    if ( (fp = fopen(fullName, "r")) ) {
	  n++;
      fclose( fp );
	}
  } while ( fp );

  if ( n == 0 )
    return NULL;

  x = (mv_TempMultiVector*) malloc(sizeof(mv_TempMultiVector));
  assert( x != NULL );
  
  x->interpreter = ii;

  x->numVectors = n;
  
  x->vector = (void**) calloc( n, sizeof(void*) );
  assert( x->vector != NULL );

  x->ownsVectors = 1;

  for ( i = 0; i < n; i++ ) {
    sprintf( fullName, "%s.%d", fileName, i ); 
    x->vector[i] = hypre_ParReadVector( comm, fullName );
  }

  x->mask = NULL;
  x->ownsMask = 0;

  return x;
}
							
int
aux_maskCount( int n, int* mask ) {

  int i, m;

  if ( mask == NULL )
    return n;

  for ( i = m = 0; i < n; i++ )
    if ( mask[i] )
      m++;

  return m;
}

void
aux_indexFromMask( int n, int* mask, int* index ) {

  long i, j;
  
  if ( mask != NULL ) {
    for ( i = 0, j = 0; i < n; i++ )
      if ( mask[i] )
	index[j++] = i + 1;
  }
  else
    for ( i = 0; i < n; i++ )
      index[i] = i + 1;

}


/* The function below is a temporary one that fills the multivector 
   part of the HYPRE_InterfaceInterpreter structure with pointers 
   that come from the temporary implementation of the multivector 
   (cf. temp_multivector.h). 
   It must be eventually replaced with a function that
   provides the respective pointers to properly implemented 
   parcsr multivector functions */

int
HYPRE_TempParCSRSetupInterpreter( mv_InterfaceInterpreter *i )
{
  /* Vector part */

  i->CreateVector = hypre_ParKrylovCreateVector;
  i->DestroyVector = hypre_ParKrylovDestroyVector; 
  i->InnerProd = hypre_ParKrylovInnerProd; 
  i->CopyVector = hypre_ParKrylovCopyVector;
  i->ClearVector = hypre_ParKrylovClearVector;
  i->SetRandomValues = hypre_ParSetRandomValues;
  i->ScaleVector = hypre_ParKrylovScaleVector;
  i->Axpy = hypre_ParKrylovAxpy;

  /* Multivector part */

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
HYPRE_ParCSRSetupInterpreter( mv_InterfaceInterpreter *i )
{
  return HYPRE_TempParCSRSetupInterpreter( i );
}

int 
HYPRE_ParCSRSetupMatvec(HYPRE_MatvecFunctions * mv)
{
  mv->MatvecCreate = hypre_ParKrylovMatvecCreate;
  mv->Matvec = hypre_ParKrylovMatvec;
  mv->MatvecDestroy = hypre_ParKrylovMatvecDestroy;

  mv->MatMultiVecCreate = NULL;
  mv->MatMultiVec = NULL;
  mv->MatMultiVecDestroy = NULL;

  return 0;
}
