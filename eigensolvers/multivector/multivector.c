#include <assert.h>
#include <math.h>

#include "multivector.h"

hypre_MultiVectorPtr 
hypre_MultiVectorCreateFromSampleVector( HYPRE_InterfaceInterpreter* ii, int n, void* sample ) { 

  long i;
  hypre_MultiVector* dest;

  dest = (hypre_MultiVector*) malloc(sizeof(hypre_MultiVector));
  assert( dest != NULL );
  
  dest->interpreter = ii;

  dest->numVectors = n;
  
  dest->vector = (void**) calloc( n, sizeof(void*) );
  assert( dest->vector != NULL );

  dest->ownsVectors = 1;

  for ( i = 0; i < n; i++ )
    dest->vector[i] = (ii->CreateVector)(sample);
  
  return (hypre_MultiVectorPtr)dest;
}

hypre_MultiVectorPtr 
hypre_MultiVectorCreateCopy( hypre_MultiVectorPtr src_, int copyValues ) {

  long i, n;

  hypre_MultiVector* dest;
  hypre_MultiVector* src = (hypre_MultiVector*)src_;

  dest = (hypre_MultiVector*) malloc(sizeof(hypre_MultiVector));
  assert( dest != NULL );
  
  dest->interpreter = src->interpreter;

  n = src->numVectors;

  dest = (hypre_MultiVector*)
	  hypre_MultiVectorCreateFromSampleVector( src->interpreter, n, src->vector[0] );

  if ( copyValues )
    for ( i = 0; i < n; i++ ) {
      (dest->interpreter->CopyVector)(src->vector[i],dest->vector[i]);
  }
  return (hypre_MultiVectorPtr)dest;
}

void 
hypre_MultiVectorDestroy( hypre_MultiVectorPtr v_ ) {

  long i;
  hypre_MultiVector* v = (hypre_MultiVector*)v_;

  if ( v == NULL )
    return;
		
  if ( v->ownsVectors && v->vector != NULL ) {
    for ( i = 0; i < v->numVectors; i++ )
      (v->interpreter->DestroyVector)(v->vector[i]);
    free(v->vector);
  }

  free(v);
}

long
hypre_MultiVectorWidth( hypre_MultiVectorPtr v ) {

  assert( v != NULL );
  return ((hypre_MultiVector*)v)->numVectors;
}

long
hypre_MultiVectorHeight( hypre_MultiVectorPtr v ) {

  assert( v != NULL );
  return 0; /*HYPRE_VECTOR_SIZE(((hypre_MultiVector*)v)->vector[0]);*/
}

void
hypre_MultiVectorClear( hypre_MultiVectorPtr v_ ) {

  long i;
  hypre_MultiVector* v = (hypre_MultiVector*)v_;
  
  for ( i = 0; i < ((hypre_MultiVector*)v)->numVectors; i++ )
    (v->interpreter->ClearVector)(((hypre_MultiVector*)v)->vector[i]);
}

void
hypre_MultiVectorSetRandom( hypre_MultiVectorPtr v_, int seed ) {

  long i;
  hypre_MultiVector* v = (hypre_MultiVector*)v_;

  srand( seed );
  for ( i = 0; i < ((hypre_MultiVector*)v)->numVectors; i++ ) {
    seed = rand();
    (v->interpreter->SetRandomValues)(((hypre_MultiVector*)v)->vector[i],seed);
  }
}

void 
hypre_MultiVectorSetVector( hypre_MultiVectorPtr mv_, int i, void* v ) { 

  hypre_MultiVector* mv = (hypre_MultiVector*)mv_;
  
  (mv->interpreter->CopyVector)(v,((hypre_MultiVector*)mv)->vector[i]);
  
}

void 
hypre_MultiVectorGetVector( hypre_MultiVectorPtr mv_, int i, void* v ) { 

  hypre_MultiVector* mv = (hypre_MultiVector*)mv_;
  
  (mv->interpreter->CopyVector)(((hypre_MultiVector*)mv)->vector[i],v);
  
}

void
hypre_collectVectorPtr( int* indexOrMask, int isMask, 
			   hypre_MultiVector* x, void** px ) {

  long ix, jx;

  if ( indexOrMask != NULL ) {
    if ( isMask ) {
      for ( ix = 0, jx = 0; ix < x->numVectors; ix++ )
	if ( indexOrMask[ix] )
	  px[jx++] = x->vector[ix];
    }
    else {
      for ( ix = 0; ix < x->numVectors; ix++ )
	if ( indexOrMask[ix] < 0 )
	  break;
	else
	  px[ix] = x->vector[indexOrMask[ix]-1];
    }	
  }
  else
    for ( ix = 0; ix < x->numVectors; ix++ )
      px[ix] = x->vector[ix];

}

void 
hypre_MultiVectorCopy( int* srcMask, hypre_MultiVectorPtr src_,
		       int* destMask, hypre_MultiVectorPtr dest_ ) {

  long i, ms, md;
  int srcMaskIsMask;
  int destMaskIsMask;
  void** ps;
  void** pd;
  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;

  ms = aux_indexOrMaskCount( src->numVectors, srcMask, &srcMaskIsMask );
  md = aux_indexOrMaskCount( dest->numVectors, destMask, &destMaskIsMask );
  assert( ms == md );
	
  ps = (void**) calloc( ms, sizeof(void*) );
  assert( ps != NULL );
  pd = (void**) calloc( md, sizeof(void*) );
  assert( pd != NULL );

  hypre_collectVectorPtr( srcMask, srcMaskIsMask, src, ps );
  hypre_collectVectorPtr( destMask, destMaskIsMask, dest, pd );

  for ( i = 0; i < ms; i++ )
    (src->interpreter->CopyVector)(ps[i],pd[i]);

  free(ps);
  free(pd);
}

void 
hypre_MultiVectorAxpy( double a, int *xMask, hypre_MultiVectorPtr x_,
		       int* yMask, hypre_MultiVectorPtr y_ ) { 
	
  long i, mx, my;
  int xMaskIsMask;
  int yMaskIsMask;
  void** px;
  void** py;
  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;

  mx = aux_indexOrMaskCount( x->numVectors, xMask, &xMaskIsMask );
  my = aux_indexOrMaskCount( y->numVectors, yMask, &yMaskIsMask );
  assert( mx == my );

  px = (void**) calloc( mx, sizeof(void*) );
  assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  assert( py != NULL );

  hypre_collectVectorPtr( xMask, xMaskIsMask, x, px );
  hypre_collectVectorPtr( yMask, yMaskIsMask, y, py );

  for ( i = 0; i < mx; i++ )
    (x->interpreter->Axpy)(a,px[i],py[i]);

  free(px);
  free(py);
}

void 
hypre_MultiVectorByMultiVector( int* xMask, hypre_MultiVectorPtr x_,
				     int* yMask, hypre_MultiVectorPtr y_,
				     int xyGHeight, int xyHeight, 
				     int xyWidth, double* xyVal ) { 
/* xy = x'*y */	

  long ix, iy, mx, my, jxy;
  int xMaskIsMask;
  int yMaskIsMask;
  double* p;
  void** px;
  void** py;
  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;

  mx = aux_indexOrMaskCount( x->numVectors, xMask, &xMaskIsMask );
  assert( mx == xyHeight );

  my = aux_indexOrMaskCount( y->numVectors, yMask, &yMaskIsMask );
  assert( my == xyWidth );

  px = (void**) calloc( mx, sizeof(void*) );
  assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  assert( py != NULL );

  hypre_collectVectorPtr( xMask, xMaskIsMask, x, px );
  hypre_collectVectorPtr( yMask, yMaskIsMask, y, py );

  jxy = xyGHeight - xyHeight;
  for ( iy = 0, p = xyVal; iy < my; iy++ ) {
    for ( ix = 0; ix < mx; ix++, p++ )
      *p = (x->interpreter->InnerProd)(px[ix],py[iy]);
    p += jxy;
  }

  free(px);
  free(py);

}

void 
hypre_MultiVectorByMultiVectorDiag( int* xMask, hypre_MultiVectorPtr x_,
					 int* yMask, hypre_MultiVectorPtr y_,
					 int* dMask, int n, double* diag ) {
/* diag = diag(x'*y) */	

  long i, mx, my, m;
  int xMaskIsMask;
  int yMaskIsMask;
  int dMaskIsMask;

  void** px;
  void** py;

  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;

  int* index;

  mx = aux_indexOrMaskCount( x->numVectors, xMask, &xMaskIsMask );
  my = aux_indexOrMaskCount( y->numVectors, yMask, &yMaskIsMask );
  m = aux_indexOrMaskCount( n, dMask, &dMaskIsMask );
  assert( mx == my && mx == m );

  px = (void**) calloc( mx, sizeof(void*) );
  assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  assert( py != NULL );

  hypre_collectVectorPtr( xMask, xMaskIsMask, x, px );
  hypre_collectVectorPtr( yMask, yMaskIsMask, y, py );

  if ( dMaskIsMask ) {
    index = (int*)calloc( m, sizeof(int) );
    aux_indexFromMask( n, dMask, index );
  }
  else
    index = dMask;

  for ( i = 0; i < m; i++ )
    *(diag+index[i]-1) = (x->interpreter->InnerProd)(px[i],py[i]);
  
  if ( dMaskIsMask )
    free(index);
  free(px);
  free(py);

}

void 
hypre_MultiVectorByMatrix( int* srcMask, hypre_MultiVectorPtr src_, 
				int rGHeight, int rHeight, 
				int rWidth, double* rVal,
				int* destMask, hypre_MultiVectorPtr dest_ ) {

  long i, j, jump;
  long ms, md;
  int srcMaskIsMask;
  int destMaskIsMask;
  double* p;

  void** ps;
  void** pd;

  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;

  ms = aux_indexOrMaskCount( src->numVectors, srcMask, &srcMaskIsMask );
  md = aux_indexOrMaskCount( dest->numVectors, destMask, &destMaskIsMask );

  assert( ms == rHeight && md == rWidth );
  
  ps = (void**) calloc( ms, sizeof(void*) );
  assert( ps != NULL );
  pd = (void**) calloc( md, sizeof(void*) );
  assert( pd != NULL );
  
  hypre_collectVectorPtr( srcMask, srcMaskIsMask, src, ps );
  hypre_collectVectorPtr( destMask, destMaskIsMask, dest, pd );

  jump = rGHeight - rHeight;
  for ( j = 0, p = rVal; j < md; j++ ) {
    (src->interpreter->ClearVector)( pd[j] );
    for ( i = 0; i < ms; i++, p++ )
      (src->interpreter->Axpy)(*p,ps[i],pd[j]);
    p += jump;
  }

  free(ps);
  free(pd);
}

void 
hypre_MultiVectorXapy( int* srcMask, hypre_MultiVectorPtr src_, 
				int rGHeight, int rHeight, 
				int rWidth, double* rVal,
				int* destMask, hypre_MultiVectorPtr dest_ ) {

  long i, j, jump;
  long ms, md;
  int srcMaskIsMask;
  int destMaskIsMask;
  double* p;

  void** ps;
  void** pd;

  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;

  ms = aux_indexOrMaskCount( src->numVectors, srcMask, &srcMaskIsMask );
  md = aux_indexOrMaskCount( dest->numVectors, destMask, &destMaskIsMask );

  assert( ms == rHeight && md == rWidth );
  
  ps = (void**) calloc( ms, sizeof(void*) );
  assert( ps != NULL );
  pd = (void**) calloc( md, sizeof(void*) );
  assert( pd != NULL );
  
  hypre_collectVectorPtr( srcMask, srcMaskIsMask, src, ps );
  hypre_collectVectorPtr( destMask, destMaskIsMask, dest, pd );

  jump = rGHeight - rHeight;
  for ( j = 0, p = rVal; j < md; j++ ) {
    for ( i = 0; i < ms; i++, p++ )
      (src->interpreter->Axpy)(*p,ps[i],pd[j]);
    p += jump;
  }

  free(ps);
  free(pd);
}

void 
hypre_MultiVectorByDiagonal( int* srcMask, hypre_MultiVectorPtr src_, 
				  int* diagMask, int n, double* diag,
				  int* destMask, hypre_MultiVectorPtr dest_ ) {

  long j;
  long ms, md, m;
  int srcMaskIsMask;
  int destMaskIsMask;
  int diagMaskIsMask;

  void** ps;
  void** pd;

  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;
  
  int* index;

  ms = aux_indexOrMaskCount( src->numVectors, srcMask, &srcMaskIsMask );
  md = aux_indexOrMaskCount( dest->numVectors, destMask, &destMaskIsMask );
  m = aux_indexOrMaskCount( n, diagMask, &diagMaskIsMask );
	
  assert( ms == m && md == m );

  if ( m < 1 )
    return;

  ps = (void**) calloc( ms, sizeof(void*) );
  assert( ps != NULL );
  pd = (void**) calloc( md, sizeof(void*) );
  assert( pd != NULL );

  if ( diagMaskIsMask ) {
    index = (int*)calloc( m, sizeof(int) );
    aux_indexFromMask( n, diagMask, index );
  }
  else
    index = diagMask;

  hypre_collectVectorPtr( srcMask, srcMaskIsMask, src, ps );
  hypre_collectVectorPtr( destMask, destMaskIsMask, dest, pd );

  for ( j = 0; j < md; j++ ) {
    (src->interpreter->ClearVector)(pd[j]);
    (src->interpreter->Axpy)(diag[index[j]-1],ps[j],pd[j]);
  }

  free(ps);
  free(pd);
  if ( diagMaskIsMask )
    free( index );
}

void 
hypre_MultiVectorExplicitQR( int* xMask, hypre_MultiVectorPtr x_, 
				  int rGHeight, int rHeight, 
				  int rWidth, double* rVal ) {

  long i, j, m;
  int xMaskIsMask;
  double* p;

  void** px;
  hypre_MultiVector* x = (hypre_MultiVector*)x_;

  m = aux_indexOrMaskCount( x->numVectors, xMask, &xMaskIsMask );
  assert( m == rHeight && m == rWidth );

  px = (void**) calloc( m, sizeof(void*) );
  assert( px != NULL );

  hypre_collectVectorPtr( xMask, xMaskIsMask, x, px );

  for ( j = 0, p = rVal; j < m; j++ ) {
    for ( i = 0; i < j; i++, p++ ) {
      *p = (x->interpreter->InnerProd)(px[i],px[j]);
      (x->interpreter->Axpy)(-(*p),px[i],px[j]);
    }
    *p = (x->interpreter->InnerProd)(px[j],px[j]);
    assert( *p > 0 );
    *p = sqrt( *p );
    (x->interpreter->ScaleVector)(1.0/(*p),px[j]);
    p += rGHeight - j;
  }

  free(px);
}

void 
hypre_MultiVectorEval( void (*f)( void*, void*, void* ), void* par,
			    int* xMask, hypre_MultiVectorPtr x_, 
			    int* yMask, hypre_MultiVectorPtr y_ ) {

  long i, mx, my;
  int xMaskIsMask;
  int yMaskIsMask;
  void** px;
  void** py;
  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  hypre_MultiVector* y = (hypre_MultiVector*)y_;
  
  if ( f == NULL ) {
    hypre_MultiVectorCopy( xMask, x_, yMask, y_ );
    return;
  }

  mx = aux_indexOrMaskCount( x->numVectors, xMask, &xMaskIsMask );
  my = aux_indexOrMaskCount( y->numVectors, yMask, &yMaskIsMask );
  assert( mx == my );

  px = (void**) calloc( mx, sizeof(void*) );
  assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  assert( py != NULL );

  hypre_collectVectorPtr( xMask, xMaskIsMask, x, px );
  hypre_collectVectorPtr( yMask, yMaskIsMask, y, py );

  for ( i = 0; i < mx; i++ )
    f( par, (void*)px[i], (void*)py[i] );

  free(px);
  free(py);
}

int
hypre_MultiVectorPrint( hypre_MultiVectorPtr x_, const char* fileName ) {

  int i, ierr;
  hypre_MultiVector* x = (hypre_MultiVector*)x_;
  char fullName[128];
  
  if ( x->interpreter->PrintVector == NULL )
    return 1;

  ierr = 0;
  for ( i = 0; i < x->numVectors; i++ ) {
    sprintf( fullName, "%s.%d", fileName, i ); 
    ierr = ierr || 
      (x->interpreter->PrintVector)( x->vector[i], fullName );
  }
  return ierr;
}
							
hypre_MultiVectorPtr 
hypre_MultiVectorRead( MPI_Comm comm, HYPRE_InterfaceInterpreter* ii, const char* fileName ) {

  int i, n, id;
  FILE* fp;
  char fullName[128];
  hypre_MultiVector* x;
  
  if ( ii->ReadVector == NULL )
    return NULL;

  MPI_Comm_rank( comm, &id );
  
  n = 0;
  do {
    sprintf( fullName, "%s.%d.%d", fileName, n, id ); 
    if ( (fp = fopen(fullName, "r")) ) {
	  n++;
      fclose( fp );
	}
  } while ( fp );

  x = (hypre_MultiVector*) malloc(sizeof(hypre_MultiVector));
  assert( x != NULL );
  
  x->interpreter = ii;

  x->numVectors = n;
  
  x->vector = (void**) calloc( n, sizeof(void*) );
  assert( x->vector != NULL );

  x->ownsVectors = 1;

  for ( i = 0; i < n; i++ ) {
    sprintf( fullName, "%s.%d", fileName, i ); 
    x->vector[i] = (ii->ReadVector)( comm, fullName );
  }

  return (hypre_MultiVectorPtr)x;
}
							
long
aux_indexOrMaskCount( int n, int* indexOrMask, int* isMask ) {

  int i, m;

  *isMask = 1;

  if ( indexOrMask == NULL )
    return n;

  for ( i = 0; i < n; i++ ) {
    if ( indexOrMask[i] != 0 && indexOrMask[i] != 1 ) {
      *isMask = 0;
      break;
    }
  }
	
  if ( *isMask ) {
    for ( i = m = 0; i < n; i++ )
      if ( indexOrMask[i] )
	m++;
  }
  else {
    for ( i = m = 0; i < n; i++, m++ )
      if ( indexOrMask[i] < 0 )
	break;
  }
	
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

