#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "fortran.h"
/*  #include "hypre_blas.h"  */

#include "fortran_matrix.h"

/*****************************************************************************/
/* Fortran Matrices */

utilities_FortranMatrix*
utilities_FortranMatrixCreate() {

  utilities_FortranMatrix* mtx;

  mtx = (utilities_FortranMatrix*) malloc( sizeof(utilities_FortranMatrix) );
  assert( mtx != NULL );

  mtx->globalHeight = 0;
  mtx->height = 0;
  mtx->width = 0;
  mtx->value = NULL;
  mtx->ownsValues = 0;

  return mtx;
}

void
utilities_FortranMatrixAllocateData( long  h, long w, 
				     utilities_FortranMatrix* mtx ) {

  assert( h > 0 && w > 0 );
  assert( mtx != NULL );

  if ( mtx->value != NULL && mtx->ownsValues )
    free( mtx->value );

  mtx->value = (double*) calloc( h*w, sizeof(double) );
  assert ( mtx->value != NULL );

  mtx->globalHeight = h;
  mtx->height = h;
  mtx->width = w;
  mtx->ownsValues = 1;
}


void
utilities_FortranMatrixWrap( double* v, long gh, long  h, long w, 
			     utilities_FortranMatrix* mtx ) {

  assert( h > 0 && w > 0 );
  assert( mtx != NULL );

  if ( mtx->value != NULL && mtx->ownsValues )
    free( mtx->value );

  mtx->value = v;
  assert ( mtx->value != NULL );

  mtx->globalHeight = gh;
  mtx->height = h;
  mtx->width = w;
  mtx->ownsValues = 0;
}


void
utilities_FortranMatrixDestroy( utilities_FortranMatrix* mtx ) {

  if ( mtx == NULL )
    return;

  if ( mtx->ownsValues && mtx->value != NULL )
    free(mtx->value);

  free(mtx);
}

long
utilities_FortranMatrixGlobalHeight( utilities_FortranMatrix* mtx ) {

  assert( mtx != NULL );

  return mtx->globalHeight;
}

long
utilities_FortranMatrixHeight( utilities_FortranMatrix* mtx ) {

  assert( mtx != NULL );

  return mtx->height;
}

long
utilities_FortranMatrixWidth( utilities_FortranMatrix* mtx ) {

  assert( mtx != NULL );

  return mtx->width;
}

double*
utilities_FortranMatrixValues( utilities_FortranMatrix* mtx ) {

  assert( mtx != NULL );

  return mtx->value;
}

void
utilities_FortranMatrixClear( utilities_FortranMatrix* mtx ) {

  long i, j, h, w, jump;
  double* p;

  assert( mtx != NULL );

  h = mtx->height;
  w = mtx->width;

  jump = mtx->globalHeight - h;
	
  for ( j = 0, p = mtx->value; j < w; j++ ) {
    for ( i = 0; i < h; i++, p++ )
      *p = 0.0;
    p += jump;
  }
}

void
utilities_FortranMatrixClearL( utilities_FortranMatrix* mtx ) {

  long i, j, k, h, w, jump;
  double* p;

  assert( mtx != NULL );

  h = mtx->height;
  w = mtx->width;

  if ( w > h )
    w = h;

  jump = mtx->globalHeight - h;
	
  for ( j = 0, p = mtx->value; j < w - 1; j++ ) {
    k = j + 1;
    p += k;
    for ( i = k; i < h; i++, p++ )
      *p = 0.0;
    p += jump;
  }
}


void 
utilities_FortranMatrixSetToIdentity( utilities_FortranMatrix* mtx ) {

  long j, h, w, jump;
  double* p;

  assert( mtx != NULL );

  utilities_FortranMatrixClear( mtx );

  h = mtx->height;
  w = mtx->width;

  jump = mtx->globalHeight;

  for ( j = 0, p = mtx->value; j < w && j < h; j++, p += jump )
    *p++ = 1.0;

}

void 
utilities_FortranMatrixTransposeSquare( utilities_FortranMatrix* mtx ) {

  long i, j, g, h, w, jump;
  double* p;
  double* q;
  double tmp;

  assert( mtx != NULL );

  g = mtx->globalHeight;
  h = mtx->height;
  w = mtx->width;

  assert( h == w );

  jump = mtx->globalHeight - h;
	
  for ( j = 0, p = mtx->value; j < w; j++ ) {
    q = p;
    p++;
    q += g;
    for ( i = j + 1; i < h; i++, p++, q += g ) {
      tmp = *p;
      *p = *q;
      *q = tmp;
    }
    p += ++jump;
  }
}

void 
utilities_FortranMatrixSymmetrize( utilities_FortranMatrix* mtx ) {

  long i, j, g, h, w, jump;
  double* p;
  double* q;

  assert( mtx != NULL );

  g = mtx->globalHeight;
  h = mtx->height;
  w = mtx->width;

  assert( h == w );

  jump = mtx->globalHeight - h;
	
  for ( j = 0, p = mtx->value; j < w; j++ ) {
    q = p;
    p++;
    q += g;
    for ( i = j + 1; i < h; i++, p++, q += g )
      *p = *q = (*p + *q)*0.5;
    p += ++jump;
  }
}

void 
utilities_FortranMatrixCopy( utilities_FortranMatrix* src, int t, 
				  utilities_FortranMatrix* dest ) {

  long i, j, h, w;
  long jp, jq, jr;
  double* p;
  double* q;
  double* r;

  assert( src != NULL && dest != NULL );

  h = dest->height;
  w = dest->width;

  jp = dest->globalHeight - h;

  if ( t == 0 ) {
    assert( src->height == h && src->width == w );
    jq = 1;
    jr = src->globalHeight;
  }
  else {
    assert( src->height == w && src->width == h );
    jr = 1;
    jq = src->globalHeight;
  }

  for ( j = 0, p = dest->value, r = src->value; j < w; j++, p += jp, r += jr )
    for ( i = 0, q = r; i < h; i++, p++, q += jq )
      *p = *q;
}

void 
utilities_FortranMatrixIndexCopy( int* index, 
				       utilities_FortranMatrix* src, int t, 
				       utilities_FortranMatrix* dest ) {

  long i, j, h, w;
  long jp, jq, jr;
  double* p;
  double* q;
  double* r;

  assert( src != NULL && dest != NULL );

  h = dest->height;
  w = dest->width;

  jp = dest->globalHeight - h;

  if ( t == 0 ) {
    assert( src->height == h && src->width == w );
    jq = 1;
    jr = src->globalHeight;
  }
  else {
    assert( src->height == w && src->width == h );
    jr = 1;
    jq = src->globalHeight;
  }

  for ( j = 0, p = dest->value; j < w; j++, p += jp ) {
    r = src->value + (index[j]-1)*jr;
    for ( i = 0, q = r; i < h; i++, p++, q += jq )
      *p = *q;
  }
}

void 
utilities_FortranMatrixSetDiagonal( utilities_FortranMatrix* mtx, 
				    utilities_FortranMatrix* vec ) {

  long j, h, w, jump;
  double* p;
  double* q;

  assert( mtx != NULL && vec != NULL );

  h = mtx->height;
  w = mtx->width;

  assert( vec->height >= h );

  jump = mtx->globalHeight + 1;

  for ( j = 0, p = mtx->value, q = vec->value; j < w && j < h; 
	j++, p += jump, q++ )
    *p = *q;

}

void 
utilities_FortranMatrixGetDiagonal( utilities_FortranMatrix* mtx, 
				    utilities_FortranMatrix* vec ) {

  long j, h, w, jump;
  double* p;
  double* q;
  
  assert( mtx != NULL && vec != NULL );

  h = mtx->height;
  w = mtx->width;

  assert( vec->height >= h );

  jump = mtx->globalHeight + 1;

  for ( j = 0, p = mtx->value, q = vec->value; j < w && j < h; 
	j++, p += jump, q++ )
    *q = *p;

}

void 
utilities_FortranMatrixAdd( double a, 
				 utilities_FortranMatrix* mtxA, 
				 utilities_FortranMatrix* mtxB, 
				 utilities_FortranMatrix* mtxC ) {

  long i, j, h, w, jA, jB, jC;
  double *pA;
  double *pB;
  double *pC;

  assert( mtxA != NULL && mtxB != NULL && mtxC != NULL );

  h = mtxA->height;
  w = mtxA->width;

  assert( mtxB->height == h && mtxB->width == w );
  assert( mtxC->height == h && mtxC->width == w );

  jA = mtxA->globalHeight - h;
  jB = mtxB->globalHeight - h;
  jC = mtxC->globalHeight - h;

  pA = mtxA->value;
  pB = mtxB->value;
  pC = mtxC->value;

  if ( a == 0.0 ) {
    for ( j = 0; j < w; j++ ) {
      for ( i = 0; i < h; i++, pA++, pB++, pC++ )
	*pC = *pB;
      pA += jA;
      pB += jB;
      pC += jC;
    }
  } 
  else if ( a == 1.0 ) {
    for ( j = 0; j < w; j++ ) {
      for ( i = 0; i < h; i++, pA++, pB++, pC++ )
	*pC = *pA + *pB;
      pA += jA;
      pB += jB;
      pC += jC;
    }
  }
  else if ( a == -1.0 ) {
    for ( j = 0; j < w; j++ ) {
      for ( i = 0; i < h; i++, pA++, pB++, pC++ )
	*pC = *pB - *pA;
      pA += jA;
      pB += jB;
      pC += jC;
    }
  }
  else {
    for ( j = 0; j < w; j++ ) {
      for ( i = 0; i < h; i++, pA++, pB++, pC++ )
	*pC = *pA * a + *pB;
      pA += jA;
      pB += jB;
      pC += jC;
    }
  }
}

void 
utilities_FortranMatrixDMultiply( utilities_FortranMatrix* vec, 
				       utilities_FortranMatrix* mtx ) {

  long i, j, h, w, jump;
  double* p;
  double* q;

  assert( mtx != NULL && vec != NULL );

  h = mtx->height;
  w = mtx->width;

  assert( vec->height == h );

  jump = mtx->globalHeight - h;

  for ( j = 0, p = mtx->value; j < w; j++ ) {
    for ( i = 0, q = vec->value; i < h; i++, p++, q++ )
      *p = *p * (*q);
    p += jump;
  }

}

void 
utilities_FortranMatrixMultiplyD( utilities_FortranMatrix* mtx, 
				       utilities_FortranMatrix* vec ) {

  long i, j, h, w, jump;
  double* p;
  double* q;

  assert( mtx != NULL && vec != NULL );

  h = mtx->height;
  w = mtx->width;

  assert( vec->height == w );

  jump = mtx->globalHeight - h;

  for ( j = 0, q = vec->value, p = mtx->value; j < w; j++, q++ ) {
    for ( i = 0; i < h; i++, p++)
      *p = *p * (*q);
    p += jump;
  }

}

void 
utilities_FortranMatrixMultiply( utilities_FortranMatrix* mtxA, int tA, 
				      utilities_FortranMatrix* mtxB, int tB,
				      utilities_FortranMatrix* mtxC ) {
  /*#ifdef HYPRE_USING_ESSL*/

  long h, w;
  long i, j, k, l;
  long iA, kA;
  long kB, jB;
  long iC, jC;

  double* pAi0;
  double* pAik;
  double* pB0j;
  double* pBkj;
  double* pC0j;
  double* pCij;

  double s;

  assert( mtxA != NULL && mtxB != NULL && mtxC != NULL );

  h = mtxC->height;
  w = mtxC->width;
  iC = 1;
  jC = mtxC->globalHeight;

  if ( tA == 0 ) {
    assert( mtxA->height == h );
    l = mtxA->width;
    iA = 1;
    kA = mtxA->globalHeight;
  }
  else {
    l = mtxA->height;
    assert( mtxA->width == h );
    kA = 1;
    iA = mtxA->globalHeight;
  }

  if ( tB == 0 ) {
    assert( mtxB->height == l );
    assert( mtxB->width == w );
    kB = 1;
    jB = mtxB->globalHeight;
  }
  else {
    assert( mtxB->width == l );
    assert( mtxB->height == w );
    jB = 1;
    kB = mtxB->globalHeight;
  }

  for ( j = 0, pB0j = mtxB->value, pC0j = mtxC->value; j < w; 
	j++, pB0j += jB, pC0j += jC  )
    for ( i = 0, pCij = pC0j, pAi0 = mtxA->value; i < h; 
	  i++, pCij += iC, pAi0 += iA ) {
      s = 0.0;
      for ( k = 0, pAik = pAi0, pBkj = pB0j; k < l; 
	    k++, pAik += kA, pBkj += kB )
	s += *pAik * (*pBkj);
      *pCij = s;
    }

  /*#else*/

#if 0

  char trA, trB;
  int m, n, k;
  int ldA, ldB, ldC;
  double alpha = 1.0, beta = 0.0;

  m = mtxC->height;
  n = mtxC->width;

  if ( tA ) {
    trA = 't';
    k = mtxA->height;
  }
  else {
    trA = 'n';
    k = mtxA->width;
  }

  trB = tB ? 't' : 'n';

  ldA = mtxA->globalHeight;
  ldB = mtxB->globalHeight;
  ldC = mtxC->globalHeight;

  hypre_F90_NAME_BLAS( dgemm, DGEMM )( &trA, &trB, &m, &n, &k, &alpha, 
				       mtxA->value, &ldA, 
				       mtxB->value, &ldB, &beta, 
				       mtxC->value, &ldC );
#endif

}

double 
utilities_FortranMatrixFNorm( utilities_FortranMatrix* mtx ) {
	
  long i, j, h, w, jump;
  double* p;

  double norm;

  assert( mtx != NULL );

  h = mtx->height;
  w = mtx->width;
  
  jump = mtx->globalHeight - h;

  norm = 0.0;

  for ( j = 0, p = mtx->value; j < w; j++ ) {
    for ( i = 0; i < h; i++, p++ )
      norm += (*p) * (*p);
    p += jump;
  }

  norm = sqrt(norm);
  return norm;
}

double 
utilities_FortranMatrixValue( utilities_FortranMatrix* mtx, 
				     long i, long j ) {

  long k;

  assert( mtx != NULL );

  assert( 1 <= i && i <= mtx->height );
  assert( 1 <= j && j <= mtx->width );

  k = i - 1 + (j - 1)*mtx->globalHeight;
  return mtx->value[k];
}

double* 
utilities_FortranMatrixValuePtr( utilities_FortranMatrix* mtx, 
					 long i, long j ) {

  long k;

  assert( mtx != NULL );

  assert( 1 <= i && i <= mtx->height );
  assert( 1 <= j && j <= mtx->width );

  k = i - 1 + (j - 1)*mtx->globalHeight;
  return mtx->value + k;
}

double 
utilities_FortranMatrixMaxValue( utilities_FortranMatrix* mtx ) {

  long i, j, jump;
  long h, w;
  double* p;
  double maxVal;

  assert( mtx != NULL );

  h = mtx->height;
  w = mtx->width;

  jump = mtx->globalHeight - h;

  maxVal = mtx->value[0];

  for ( j = 0, p = mtx->value; j < w; j++ ) {
    for ( i = 0; i < h; i++, p++ )
      if ( *p > maxVal )
	maxVal = *p;
    p += jump;
  }

  return maxVal;
}

void 
utilities_FortranMatrixSelectBlock( utilities_FortranMatrix* mtx,
					 long iFrom, long iTo, 
					 long jFrom, long jTo,
					 utilities_FortranMatrix* block ) {

  if ( block->value != NULL && block->ownsValues )
    free( block->value );

  block->globalHeight = mtx->globalHeight;
  if ( iTo < iFrom || jTo < jFrom ) {
    block->height = 0;
    block->width = 0;
    block->value = NULL;
    return;
  }
  block->height = iTo - iFrom + 1;
  block->width = jTo - jFrom + 1;
  block->value = mtx->value + iFrom - 1 + (jFrom - 1)*mtx->globalHeight;
  block->ownsValues = 0;
}

void 
utilities_FortranMatrixUpperInv( utilities_FortranMatrix* u ) {

  long i, j, k;
  long n, jc, jd;
  double v;
  double* diag;	/* diag(i) = u(i,i)_original */
  double* pin;	/* &u(i-1,n) */
  double* pii;	/* &u(i,i) */
  double* pij;	/* &u(i,j) */
  double* pik;	/* &u(i,k) */
  double* pkj;	/* &u(k,j) */
  double* pd;	/* &diag(i) */

  n = u->height;
  assert( u->width == n );

  diag = (double*)calloc( n, sizeof(double) );
  assert( diag != NULL );

  jc = u->globalHeight;
  jd = jc + 1;

  pii = u->value;
  pd = diag;
  for ( i = 0; i < n; i++, pii += jd, pd++ ) {
    v = *pd = *pii;
    *pii = 1.0/v;
  }

  pii -= jd;
  pin = pii - 1;
  pii -= jd;
  pd -= 2;
  for ( i = n - 1; i > 0; i--, pii -= jd, pin--, pd-- ) {
    pij = pin;
    for ( j = n; j > i; j--, pij -= jc ) {
      v = 0;
      pik = pii + jc;
      pkj = pij + 1;
      for ( k = i + 1; k <= j; k++, pik += jc, pkj++  ) {
	v -= (*pik) * (*pkj);
      }
      *pij = v/(*pd);
    }
  }

  free( diag );

}

int
utilities_FortranMatrixPrint( utilities_FortranMatrix* mtx, char fileName[] ) {

  long i, j, h, w, jump;
  double* p;
  FILE* fp;

  assert( mtx != NULL );

  if ( !(fp = fopen(fileName,"w")) )
    return 1;

  h = mtx->height;
  w = mtx->width;
  
  fprintf(fp,"%d\n",h);
  fprintf(fp,"%d\n",w);
  
  jump = mtx->globalHeight - h;
	
  for ( j = 0, p = mtx->value; j < w; j++ ) {
    for ( i = 0; i < h; i++, p++ )
      fprintf(fp,"%22.16e\n",*p);
    p += jump;
  }

  fclose(fp);
  return 0;
}

