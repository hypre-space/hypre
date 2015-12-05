/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.11 $
 ***********************************************************************EHEADER*/




#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "temp_multivector.h"
#include "interpreter.h"
#include "_hypre_utilities.h"

static void
mv_collectVectorPtr( HYPRE_Int* mask, mv_TempMultiVector* x, void** px ) {

  HYPRE_Int ix, jx;

  if ( mask != NULL ) {
    for ( ix = 0, jx = 0; ix < x->numVectors; ix++ )
      if ( mask[ix] )
	px[jx++] = x->vector[ix];
  }
  else
    for ( ix = 0; ix < x->numVectors; ix++ )
      px[ix] = x->vector[ix];

}

static HYPRE_Int
aux_maskCount( HYPRE_Int n, HYPRE_Int* mask ) {

  HYPRE_Int i, m;

  if ( mask == NULL )
    return n;

  for ( i = m = 0; i < n; i++ )
    if ( mask[i] )
      m++;

  return m;
}

static void
aux_indexFromMask( HYPRE_Int n, HYPRE_Int* mask, HYPRE_Int* index ) {

  HYPRE_Int i, j;
  
  if ( mask != NULL ) {
    for ( i = 0, j = 0; i < n; i++ )
      if ( mask[i] )
	index[j++] = i + 1;
  }
  else
    for ( i = 0; i < n; i++ )
      index[i] = i + 1;

}

/* ------- here goes simple random number generator --------- */

static hypre_ulongint next = 1;

/* RAND_MAX assumed to be 32767 */
static HYPRE_Int myrand(void) {
   next = next * 1103515245 + 12345;
   return((unsigned)(next/65536) % 32768);
}

static void mysrand(unsigned seed) {
   next = seed;
}


void*
mv_TempMultiVectorCreateFromSampleVector( void* ii_, HYPRE_Int n, void* sample ) { 

  HYPRE_Int i;
  mv_TempMultiVector* x;
  mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;

  x = (mv_TempMultiVector*) malloc(sizeof(mv_TempMultiVector));
  hypre_assert( x != NULL );
  
  x->interpreter = ii;
  x->numVectors = n;
  
  x->vector = (void**) calloc( n, sizeof(void*) );
  hypre_assert( x->vector != NULL );

  x->ownsVectors = 1;
  x->mask = NULL;
  x->ownsMask = 0;

  for ( i = 0; i < n; i++ )
    x->vector[i] = (ii->CreateVector)(sample);

  return x;

}

void*
mv_TempMultiVectorCreateCopy( void* src_, HYPRE_Int copyValues ) {

  HYPRE_Int i, n;

  mv_TempMultiVector* src;
  mv_TempMultiVector* dest;

  src = (mv_TempMultiVector*)src_;
  hypre_assert( src != NULL );

  n = src->numVectors;

  dest = mv_TempMultiVectorCreateFromSampleVector( src->interpreter, 
						      n, src->vector[0] );
  if ( copyValues )
    for ( i = 0; i < n; i++ ) {
      (dest->interpreter->CopyVector)(src->vector[i],dest->vector[i]);
  }

  return dest;
}

void 
mv_TempMultiVectorDestroy( void* x_ ) {

  HYPRE_Int i;
  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  if ( x == NULL )
    return;

  if ( x->ownsVectors && x->vector != NULL ) {
    for ( i = 0; i < x->numVectors; i++ )
      (x->interpreter->DestroyVector)(x->vector[i]);
    free(x->vector);
  }
  if ( x->mask && x->ownsMask )
    free(x->mask);
  free(x);
}

HYPRE_Int
mv_TempMultiVectorWidth( void* x_ ) {

  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  if ( x == NULL )
    return 0;

  return x->numVectors;
}

HYPRE_Int
mv_TempMultiVectorHeight( void* x_ ) {
 
  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;
  
  if ( x == NULL )
   return 0; 
 
  return (x->interpreter->VectorSize)(x->vector[0]); 
}

/* this shallow copy of the mask is convenient but not safe;
   a proper copy should be considered */
void
mv_TempMultiVectorSetMask( void* x_, HYPRE_Int* mask ) {

  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  hypre_assert( x != NULL );
  x->mask = mask;
  x->ownsMask = 0;
}

void
mv_TempMultiVectorClear( void* x_ ) {

  HYPRE_Int i;
  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  hypre_assert( x != NULL );

  for ( i = 0; i < x->numVectors; i++ )
    if ( x->mask == NULL || (x->mask)[i] )
      (x->interpreter->ClearVector)(x->vector[i]);
}

void
mv_TempMultiVectorSetRandom( void* x_, HYPRE_Int seed ) {

  HYPRE_Int i;
  mv_TempMultiVector* x = (mv_TempMultiVector*)x_;

  hypre_assert( x != NULL );

  mysrand(seed);

  for ( i = 0; i < x->numVectors; i++ ) {
    if ( x->mask == NULL || (x->mask)[i] ) {
      seed=myrand();
      (x->interpreter->SetRandomValues)(x->vector[i], seed);
    }
  }
}



void 
mv_TempMultiVectorCopy( void* src_, void* dest_ ) {

  HYPRE_Int i, ms, md;
  void** ps;
  void** pd;
  mv_TempMultiVector* src = (mv_TempMultiVector*)src_;
  mv_TempMultiVector* dest = (mv_TempMultiVector*)dest_;

  hypre_assert( src != NULL && dest != NULL );

  ms = aux_maskCount( src->numVectors, src->mask );
  md = aux_maskCount( dest->numVectors, dest->mask );
  hypre_assert( ms == md );
	
  ps = (void**) calloc( ms, sizeof(void*) );
  hypre_assert( ps != NULL );
  pd = (void**) calloc( md, sizeof(void*) );
  hypre_assert( pd != NULL );

  mv_collectVectorPtr( src->mask, src, ps );
  mv_collectVectorPtr( dest->mask, dest, pd );

  for ( i = 0; i < ms; i++ )
    (src->interpreter->CopyVector)(ps[i],pd[i]);

  free(ps);
  free(pd);
}

void 
mv_TempMultiVectorAxpy( double a, void* x_, void* y_ ) { 
	
  HYPRE_Int i, mx, my;
  void** px;
  void** py;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  mx = aux_maskCount( x->numVectors, x->mask );
  my = aux_maskCount( y->numVectors, y->mask );
  hypre_assert( mx == my );

  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );

  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

  for ( i = 0; i < mx; i++ )
    (x->interpreter->Axpy)(a,px[i],py[i]);

  free(px);
  free(py);
}

void 
mv_TempMultiVectorByMultiVector( void* x_, void* y_,
				     HYPRE_Int xyGHeight, HYPRE_Int xyHeight, 
				     HYPRE_Int xyWidth, double* xyVal ) { 
/* xy = x'*y */	

  HYPRE_Int ix, iy, mx, my, jxy;
  double* p;
  void** px;
  void** py;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  mx = aux_maskCount( x->numVectors, x->mask );
  hypre_assert( mx == xyHeight );

  my = aux_maskCount( y->numVectors, y->mask );
  hypre_assert( my == xyWidth );

  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );

  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

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
mv_TempMultiVectorByMultiVectorDiag( void* x_, void* y_,
					HYPRE_Int* mask, HYPRE_Int n, double* diag ) {
/* diag = diag(x'*y) */	

  HYPRE_Int i, mx, my, m;
  void** px;
  void** py;
  HYPRE_Int* index;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  mx = aux_maskCount( x->numVectors, x->mask );
  my = aux_maskCount( y->numVectors, y->mask );
  m = aux_maskCount( n, mask );
  hypre_assert( mx == my && mx == m );

  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );

  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

  index = (HYPRE_Int*)calloc( m, sizeof(HYPRE_Int) );
  aux_indexFromMask( n, mask, index );

  for ( i = 0; i < m; i++ )
    *(diag+index[i]-1) = (x->interpreter->InnerProd)(px[i],py[i]);
  
  free(index);
  free(px);
  free(py);

}

void 
mv_TempMultiVectorByMatrix( void* x_, 
			       HYPRE_Int rGHeight, HYPRE_Int rHeight, 
			       HYPRE_Int rWidth, double* rVal,
			       void* y_ ) {

  HYPRE_Int i, j, jump;
  HYPRE_Int mx, my;
  double* p;
  void** px;
  void** py;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  mx = aux_maskCount( x->numVectors, x->mask );
  my = aux_maskCount( y->numVectors, y->mask );

  hypre_assert( mx == rHeight && my == rWidth );
  
  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );
  
  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

  jump = rGHeight - rHeight;
  for ( j = 0, p = rVal; j < my; j++ ) {
    (x->interpreter->ClearVector)( py[j] );
    for ( i = 0; i < mx; i++, p++ )
      (x->interpreter->Axpy)(*p,px[i],py[j]);
    p += jump;
  }

  free(px);
  free(py);
}

void 
mv_TempMultiVectorXapy( void* x_, 
			   HYPRE_Int rGHeight, HYPRE_Int rHeight, 
			   HYPRE_Int rWidth, double* rVal,
			   void* y_ ) {

  HYPRE_Int i, j, jump;
  HYPRE_Int mx, my;
  double* p;
  void** px;
  void** py;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  mx = aux_maskCount( x->numVectors, x->mask );
  my = aux_maskCount( y->numVectors, y->mask );

  hypre_assert( mx == rHeight && my == rWidth );
  
  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );
  
  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

  jump = rGHeight - rHeight;
  for ( j = 0, p = rVal; j < my; j++ ) {
    for ( i = 0; i < mx; i++, p++ )
      (x->interpreter->Axpy)(*p,px[i],py[j]);
    p += jump;
  }

  free(px);
  free(py);
}

void 
mv_TempMultiVectorByDiagonal( void* x_, 
				HYPRE_Int* mask, HYPRE_Int n, double* diag,
				void* y_ ) {

  HYPRE_Int j;
  HYPRE_Int mx, my, m;
  void** px;
  void** py;
  HYPRE_Int* index;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  mx = aux_maskCount( x->numVectors, x->mask );
  my = aux_maskCount( y->numVectors, y->mask );
  m = aux_maskCount( n, mask );
	
  hypre_assert( mx == m && my == m );

  if ( m < 1 )
    return;

  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );

  index = (HYPRE_Int*)calloc( m, sizeof(HYPRE_Int) );
  aux_indexFromMask( n, mask, index );

  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

  for ( j = 0; j < my; j++ ) {
    (x->interpreter->ClearVector)(py[j]);
    (x->interpreter->Axpy)(diag[index[j]-1],px[j],py[j]);
  }

  free(px);
  free(py);
  free( index );
}

void 
mv_TempMultiVectorEval( void (*f)( void*, void*, void* ), void* par,
			   void* x_, void* y_ ) {

  HYPRE_Int i, mx, my;
  void** px;
  void** py;
  mv_TempMultiVector* x;
  mv_TempMultiVector* y;

  x = (mv_TempMultiVector*)x_;
  y = (mv_TempMultiVector*)y_;
  hypre_assert( x != NULL && y != NULL );

  if ( f == NULL ) {
    mv_TempMultiVectorCopy( x, y );
    return;
  }

  mx = aux_maskCount( x->numVectors, x->mask );
  my = aux_maskCount( y->numVectors, y->mask );
  hypre_assert( mx == my );

  px = (void**) calloc( mx, sizeof(void*) );
  hypre_assert( px != NULL );
  py = (void**) calloc( my, sizeof(void*) );
  hypre_assert( py != NULL );

  mv_collectVectorPtr( x->mask, x, px );
  mv_collectVectorPtr( y->mask, y, py );

  for ( i = 0; i < mx; i++ )
    f( par, (void*)px[i], (void*)py[i] );

  free(px);
  free(py);
}
