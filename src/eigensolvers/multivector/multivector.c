#include <assert.h>
#include <math.h>

#include "multivector.h"

hypre_MultiVectorPtr 
hypre_MultiVectorCreateFromSampleVector( void* ii_, int n, void* sample ) { 

  hypre_MultiVector* x;
  HYPRE_InterfaceInterpreter* ii = (HYPRE_InterfaceInterpreter*)ii_;

  x = (hypre_MultiVector*) malloc(sizeof(hypre_MultiVector));
  assert( x != NULL );
  
  x->interpreter = ii;
  x->data = (ii->CreateMultiVector)( ii, n, sample );
  x->ownsData = 1;

  return (hypre_MultiVectorPtr)x;

}

hypre_MultiVectorPtr 
hypre_MultiVectorCreateCopy( hypre_MultiVectorPtr x_, int copyValues ) {

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y;
  void* data;
  HYPRE_InterfaceInterpreter* ii;

  assert( x != NULL );
  ii = x->interpreter;

  y = (hypre_MultiVector*) malloc(sizeof(hypre_MultiVector));
  assert( y != NULL );
  
  data = (ii->CopyCreateMultiVector)( x->data, copyValues );

  y->interpreter = ii;
  y->data = data;
  y->ownsData = 1;

  return (hypre_MultiVectorPtr)y;
}

void 
hypre_MultiVectorDestroy( hypre_MultiVectorPtr v_ ) {

  hypre_MultiVector* v = (hypre_MultiVector*)v_;

  if ( v == NULL )
    return;

  if ( v->ownsData )
    (v->interpreter->DestroyMultiVector)( v->data );
  free( v );
}

void
hypre_MultiVectorSetMask( hypre_MultiVectorPtr v_, int* mask ) {

  hypre_MultiVector* v = (hypre_MultiVector*)v_;
  assert( v != NULL );
  (v->interpreter->SetMask)( v->data, mask );
}

int
hypre_MultiVectorWidth( hypre_MultiVectorPtr v_ ) {

  hypre_MultiVector* v = (hypre_MultiVector*)v_;

  if ( v == NULL )
    return 0;

  return (v->interpreter->Width)( v->data );
}

int
hypre_MultiVectorHeight( hypre_MultiVectorPtr v ) {

  return 0; /* not available in some interfaces */
}

void
hypre_MultiVectorClear( hypre_MultiVectorPtr v_ ) {

  hypre_MultiVector* v = (hypre_MultiVector*)v_;
  assert( v != NULL );
  (v->interpreter->ClearMultiVector)( v->data );
}

void
hypre_MultiVectorSetRandom( hypre_MultiVectorPtr v_, int seed ) {

  hypre_MultiVector* v = (hypre_MultiVector*)v_;
  assert( v != NULL );
  (v->interpreter->SetRandomVectors)( v->data, seed );
}

void 
hypre_MultiVectorCopy( hypre_MultiVectorPtr src_, hypre_MultiVectorPtr dest_ ) {

  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;
  assert( src != NULL && dest != NULL );
  (src->interpreter->CopyMultiVector)( src->data, dest->data );
}

void 
hypre_MultiVectorAxpy( double a, hypre_MultiVectorPtr x_, hypre_MultiVectorPtr y_ ) { 
	
  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->MultiAxpy)( a, x->data, y->data );
}

void 
hypre_MultiVectorByMultiVector( hypre_MultiVectorPtr x_, hypre_MultiVectorPtr y_,
				     int xyGHeight, int xyHeight, 
				     int xyWidth, double* xy ) { 
/* xy = x'*y */	

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->MultiInnerProd)
    ( x->data, y->data, xyGHeight, xyHeight, xyWidth, xy );
}

void 
hypre_MultiVectorByMultiVectorDiag( hypre_MultiVectorPtr x_, hypre_MultiVectorPtr y_,
					 int* mask, int n, double* d ) {
/* d = diag(x'*y) */	

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->MultiInnerProdDiag)( x->data, y->data, mask, n, d );
}

void 
hypre_MultiVectorByMatrix( hypre_MultiVectorPtr x_, 
			   int rGHeight, int rHeight, 
			   int rWidth, double* rVal,
			   hypre_MultiVectorPtr y_ ) {

  /* y = x*r */

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->MultiVecMat)
    ( x->data, rGHeight, rHeight, rWidth, rVal, y->data );
}

void 
hypre_MultiVectorXapy( hypre_MultiVectorPtr x_, 
		       int rGHeight, int rHeight, 
		       int rWidth, double* rVal,
		       hypre_MultiVectorPtr y_ ) {

  /* y = y + x*a */

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->MultiXapy)
    ( x->data, rGHeight, rHeight, rWidth, rVal, y->data );
}

void 
hypre_MultiVectorByDiagonal( hypre_MultiVectorPtr x_, 
			     int* mask, int n, double* d,
			     hypre_MultiVectorPtr y_ ) {

  /* y = x*d */

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->MultiVecMatDiag)( x->data, mask, n, d, y->data );
}

void 
hypre_MultiVectorEval( void (*f)( void*, void*, void* ), void* par,
		       hypre_MultiVectorPtr x_, hypre_MultiVectorPtr y_ ) {

  /* y = f(x) computed vector-wise */

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  assert( x != NULL && y != NULL );
  (x->interpreter->Eval)( f, par, x->data, y->data );
}

int
hypre_MultiVectorPrint( hypre_MultiVectorPtr x_, const char* fileName ) {

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  assert( x != NULL );
  return (x->interpreter->PrintMultiVector)( x->data, fileName );
}
							
hypre_MultiVectorPtr 
hypre_MultiVectorRead( MPI_Comm comm, void *ii, const char* fileName ) {

  hypre_MultiVector* x;
  void* xData;

  x = (hypre_MultiVector*) malloc(sizeof(hypre_MultiVector));
  assert( x != NULL );
  
  xData = (((HYPRE_InterfaceInterpreter*)ii)->ReadMultiVector)( comm, ii, fileName );

  x->interpreter = ii;
  x->data = xData;
  x->ownsData = 1;

  return (hypre_MultiVectorPtr)x;

}
							

