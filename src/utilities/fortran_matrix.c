/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "fortran_matrix.h"
#include "_hypre_utilities.h"

utilities_FortranMatrix*
utilities_FortranMatrixCreate(void)
{

   utilities_FortranMatrix* mtx;

   mtx = hypre_TAlloc(utilities_FortranMatrix, 1, HYPRE_MEMORY_HOST);
   hypre_assert( mtx != NULL );

   mtx->globalHeight = 0;
   mtx->height = 0;
   mtx->width = 0;
   mtx->value = NULL;
   mtx->ownsValues = 0;

   return mtx;
}

void
utilities_FortranMatrixAllocateData( HYPRE_BigInt  h, HYPRE_BigInt w,
                                     utilities_FortranMatrix* mtx )
{

   hypre_assert( h > 0 && w > 0 );
   hypre_assert( mtx != NULL );

   if ( mtx->value != NULL && mtx->ownsValues )
   {
      hypre_TFree( mtx->value, HYPRE_MEMORY_HOST);
   }

   mtx->value = hypre_CTAlloc(HYPRE_Real,  h * w, HYPRE_MEMORY_HOST);
   hypre_assert ( mtx->value != NULL );

   mtx->globalHeight = h;
   mtx->height = h;
   mtx->width = w;
   mtx->ownsValues = 1;
}


void
utilities_FortranMatrixWrap( HYPRE_Real* v, HYPRE_BigInt gh, HYPRE_BigInt  h, HYPRE_BigInt w,
                             utilities_FortranMatrix* mtx )
{

   hypre_assert( h > 0 && w > 0 );
   hypre_assert( mtx != NULL );

   if ( mtx->value != NULL && mtx->ownsValues )
   {
      hypre_TFree( mtx->value, HYPRE_MEMORY_HOST);
   }

   mtx->value = v;
   hypre_assert ( mtx->value != NULL );

   mtx->globalHeight = gh;
   mtx->height = h;
   mtx->width = w;
   mtx->ownsValues = 0;
}


void
utilities_FortranMatrixDestroy( utilities_FortranMatrix* mtx )
{

   if ( mtx == NULL )
   {
      return;
   }

   if ( mtx->ownsValues && mtx->value != NULL )
   {
      hypre_TFree(mtx->value, HYPRE_MEMORY_HOST);
   }

   hypre_TFree(mtx, HYPRE_MEMORY_HOST);
}

HYPRE_BigInt
utilities_FortranMatrixGlobalHeight( utilities_FortranMatrix* mtx )
{

   hypre_assert( mtx != NULL );

   return mtx->globalHeight;
}

HYPRE_BigInt
utilities_FortranMatrixHeight( utilities_FortranMatrix* mtx )
{

   hypre_assert( mtx != NULL );

   return mtx->height;
}

HYPRE_BigInt
utilities_FortranMatrixWidth( utilities_FortranMatrix* mtx )
{

   hypre_assert( mtx != NULL );

   return mtx->width;
}

HYPRE_Real*
utilities_FortranMatrixValues( utilities_FortranMatrix* mtx )
{

   hypre_assert( mtx != NULL );

   return mtx->value;
}

void
utilities_FortranMatrixClear( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, h, w, jump;
   HYPRE_Real* p;

   hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
      {
         *p = 0.0;
      }
      p += jump;
   }
}

void
utilities_FortranMatrixClearL( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, k, h, w, jump;
   HYPRE_Real* p;

   hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   if ( w > h )
   {
      w = h;
   }

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w - 1; j++ )
   {
      k = j + 1;
      p += k;
      for ( i = k; i < h; i++, p++ )
      {
         *p = 0.0;
      }
      p += jump;
   }
}


void
utilities_FortranMatrixSetToIdentity( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt j, h, w, jump;
   HYPRE_Real* p;

   hypre_assert( mtx != NULL );

   utilities_FortranMatrixClear( mtx );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight;

   for ( j = 0, p = mtx->value; j < w && j < h; j++, p += jump )
   {
      *p++ = 1.0;
   }

}

void
utilities_FortranMatrixTransposeSquare( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, g, h, w, jump;
   HYPRE_Real* p;
   HYPRE_Real* q;
   HYPRE_Real tmp;

   hypre_assert( mtx != NULL );

   g = mtx->globalHeight;
   h = mtx->height;
   w = mtx->width;

   hypre_assert( h == w );

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      q = p;
      p++;
      q += g;
      for ( i = j + 1; i < h; i++, p++, q += g )
      {
         tmp = *p;
         *p = *q;
         *q = tmp;
      }
      p += ++jump;
   }
}

void
utilities_FortranMatrixSymmetrize( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, g, h, w, jump;
   HYPRE_Real* p;
   HYPRE_Real* q;

   hypre_assert( mtx != NULL );

   g = mtx->globalHeight;
   h = mtx->height;
   w = mtx->width;

   hypre_assert( h == w );

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      q = p;
      p++;
      q += g;
      for ( i = j + 1; i < h; i++, p++, q += g )
      {
         *p = *q = (*p + *q) * 0.5;
      }
      p += ++jump;
   }
}

void
utilities_FortranMatrixCopy( utilities_FortranMatrix* src, HYPRE_Int t,
                             utilities_FortranMatrix* dest )
{

   HYPRE_BigInt i, j, h, w;
   HYPRE_BigInt jp, jq, jr;
   HYPRE_Real* p;
   HYPRE_Real* q;
   HYPRE_Real* r;

   hypre_assert( src != NULL && dest != NULL );

   h = dest->height;
   w = dest->width;

   jp = dest->globalHeight - h;

   if ( t == 0 )
   {
      hypre_assert( src->height == h && src->width == w );
      jq = 1;
      jr = src->globalHeight;
   }
   else
   {
      hypre_assert( src->height == w && src->width == h );
      jr = 1;
      jq = src->globalHeight;
   }

   for ( j = 0, p = dest->value, r = src->value; j < w; j++, p += jp, r += jr )
      for ( i = 0, q = r; i < h; i++, p++, q += jq )
      {
         *p = *q;
      }
}

void
utilities_FortranMatrixIndexCopy( HYPRE_Int* index,
                                  utilities_FortranMatrix* src, HYPRE_Int t,
                                  utilities_FortranMatrix* dest )
{

   HYPRE_BigInt i, j, h, w;
   HYPRE_BigInt jp, jq, jr;
   HYPRE_Real* p;
   HYPRE_Real* q;
   HYPRE_Real* r;

   hypre_assert( src != NULL && dest != NULL );

   h = dest->height;
   w = dest->width;

   jp = dest->globalHeight - h;

   if ( t == 0 )
   {
      hypre_assert( src->height == h && src->width == w );
      jq = 1;
      jr = src->globalHeight;
   }
   else
   {
      hypre_assert( src->height == w && src->width == h );
      jr = 1;
      jq = src->globalHeight;
   }

   for ( j = 0, p = dest->value; j < w; j++, p += jp )
   {
      r = src->value + (index[j] - 1) * jr;
      for ( i = 0, q = r; i < h; i++, p++, q += jq )
      {
         *p = *q;
      }
   }
}

void
utilities_FortranMatrixSetDiagonal( utilities_FortranMatrix* mtx,
                                    utilities_FortranMatrix* vec )
{

   HYPRE_BigInt j, h, w, jump;
   HYPRE_Real* p;
   HYPRE_Real* q;

   hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   hypre_assert( vec->height >= h );

   jump = mtx->globalHeight + 1;

   for ( j = 0, p = mtx->value, q = vec->value; j < w && j < h;
         j++, p += jump, q++ )
   {
      *p = *q;
   }

}

void
utilities_FortranMatrixGetDiagonal( utilities_FortranMatrix* mtx,
                                    utilities_FortranMatrix* vec )
{

   HYPRE_BigInt j, h, w, jump;
   HYPRE_Real* p;
   HYPRE_Real* q;

   hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   hypre_assert( vec->height >= h );

   jump = mtx->globalHeight + 1;

   for ( j = 0, p = mtx->value, q = vec->value; j < w && j < h;
         j++, p += jump, q++ )
   {
      *q = *p;
   }

}

void
utilities_FortranMatrixAdd( HYPRE_Real a,
                            utilities_FortranMatrix* mtxA,
                            utilities_FortranMatrix* mtxB,
                            utilities_FortranMatrix* mtxC )
{

   HYPRE_BigInt i, j, h, w, jA, jB, jC;
   HYPRE_Real *pA;
   HYPRE_Real *pB;
   HYPRE_Real *pC;

   hypre_assert( mtxA != NULL && mtxB != NULL && mtxC != NULL );

   h = mtxA->height;
   w = mtxA->width;

   hypre_assert( mtxB->height == h && mtxB->width == w );
   hypre_assert( mtxC->height == h && mtxC->width == w );

   jA = mtxA->globalHeight - h;
   jB = mtxB->globalHeight - h;
   jC = mtxC->globalHeight - h;

   pA = mtxA->value;
   pB = mtxB->value;
   pC = mtxC->value;

   if ( a == 0.0 )
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pB;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
   else if ( a == 1.0 )
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pA + *pB;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
   else if ( a == -1.0 )
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pB - *pA;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
   else
   {
      for ( j = 0; j < w; j++ )
      {
         for ( i = 0; i < h; i++, pA++, pB++, pC++ )
         {
            *pC = *pA * a + *pB;
         }
         pA += jA;
         pB += jB;
         pC += jC;
      }
   }
}

void
utilities_FortranMatrixDMultiply( utilities_FortranMatrix* vec,
                                  utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, h, w, jump;
   HYPRE_Real* p;
   HYPRE_Real* q;

   hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   hypre_assert( vec->height == h );

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0, q = vec->value; i < h; i++, p++, q++ )
      {
         *p = *p * (*q);
      }
      p += jump;
   }

}

void
utilities_FortranMatrixMultiplyD( utilities_FortranMatrix* mtx,
                                  utilities_FortranMatrix* vec )
{

   HYPRE_BigInt i, j, h, w, jump;
   HYPRE_Real* p;
   HYPRE_Real* q;

   hypre_assert( mtx != NULL && vec != NULL );

   h = mtx->height;
   w = mtx->width;

   hypre_assert( vec->height == w );

   jump = mtx->globalHeight - h;

   for ( j = 0, q = vec->value, p = mtx->value; j < w; j++, q++ )
   {
      for ( i = 0; i < h; i++, p++)
      {
         *p = *p * (*q);
      }
      p += jump;
   }

}

void
utilities_FortranMatrixMultiply( utilities_FortranMatrix* mtxA, HYPRE_Int tA,
                                 utilities_FortranMatrix* mtxB, HYPRE_Int tB,
                                 utilities_FortranMatrix* mtxC )
{
   HYPRE_BigInt h, w;
   HYPRE_BigInt i, j, k, l;
   HYPRE_BigInt iA, kA;
   HYPRE_BigInt kB, jB;
   HYPRE_BigInt iC, jC;

   HYPRE_Real* pAi0;
   HYPRE_Real* pAik;
   HYPRE_Real* pB0j;
   HYPRE_Real* pBkj;
   HYPRE_Real* pC0j;
   HYPRE_Real* pCij;

   HYPRE_Real s;

   hypre_assert( mtxA != NULL && mtxB != NULL && mtxC != NULL );

   h = mtxC->height;
   w = mtxC->width;
   iC = 1;
   jC = mtxC->globalHeight;

   if ( tA == 0 )
   {
      hypre_assert( mtxA->height == h );
      l = mtxA->width;
      iA = 1;
      kA = mtxA->globalHeight;
   }
   else
   {
      l = mtxA->height;
      hypre_assert( mtxA->width == h );
      kA = 1;
      iA = mtxA->globalHeight;
   }

   if ( tB == 0 )
   {
      hypre_assert( mtxB->height == l );
      hypre_assert( mtxB->width == w );
      kB = 1;
      jB = mtxB->globalHeight;
   }
   else
   {
      hypre_assert( mtxB->width == l );
      hypre_assert( mtxB->height == w );
      jB = 1;
      kB = mtxB->globalHeight;
   }

   for ( j = 0, pB0j = mtxB->value, pC0j = mtxC->value; j < w;
         j++, pB0j += jB, pC0j += jC  )
      for ( i = 0, pCij = pC0j, pAi0 = mtxA->value; i < h;
            i++, pCij += iC, pAi0 += iA )
      {
         s = 0.0;
         for ( k = 0, pAik = pAi0, pBkj = pB0j; k < l;
               k++, pAik += kA, pBkj += kB )
         {
            s += *pAik * (*pBkj);
         }
         *pCij = s;
      }
}

HYPRE_Real
utilities_FortranMatrixFNorm( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, h, w, jump;
   HYPRE_Real* p;

   HYPRE_Real norm;

   hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight - h;

   norm = 0.0;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
      {
         norm += (*p) * (*p);
      }
      p += jump;
   }

   norm = hypre_sqrt(norm);
   return norm;
}

HYPRE_Real
utilities_FortranMatrixValue( utilities_FortranMatrix* mtx,
                              HYPRE_BigInt i, HYPRE_BigInt j )
{

   HYPRE_BigInt k;

   hypre_assert( mtx != NULL );

   hypre_assert( 1 <= i && i <= mtx->height );
   hypre_assert( 1 <= j && j <= mtx->width );

   k = i - 1 + (j - 1) * mtx->globalHeight;
   return mtx->value[k];
}

HYPRE_Real*
utilities_FortranMatrixValuePtr( utilities_FortranMatrix* mtx,
                                 HYPRE_BigInt i, HYPRE_BigInt j )
{

   HYPRE_BigInt k;

   hypre_assert( mtx != NULL );

   hypre_assert( 1 <= i && i <= mtx->height );
   hypre_assert( 1 <= j && j <= mtx->width );

   k = i - 1 + (j - 1) * mtx->globalHeight;
   return mtx->value + k;
}

HYPRE_Real
utilities_FortranMatrixMaxValue( utilities_FortranMatrix* mtx )
{

   HYPRE_BigInt i, j, jump;
   HYPRE_BigInt h, w;
   HYPRE_Real* p;
   HYPRE_Real maxVal;

   hypre_assert( mtx != NULL );

   h = mtx->height;
   w = mtx->width;

   jump = mtx->globalHeight - h;

   maxVal = mtx->value[0];

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
         if ( *p > maxVal )
         {
            maxVal = *p;
         }
      p += jump;
   }

   return maxVal;
}

void
utilities_FortranMatrixSelectBlock( utilities_FortranMatrix* mtx,
                                    HYPRE_BigInt iFrom, HYPRE_BigInt iTo,
                                    HYPRE_BigInt jFrom, HYPRE_BigInt jTo,
                                    utilities_FortranMatrix* block )
{

   if ( block->value != NULL && block->ownsValues )
   {
      hypre_TFree( block->value, HYPRE_MEMORY_HOST);
   }

   block->globalHeight = mtx->globalHeight;
   if ( iTo < iFrom || jTo < jFrom )
   {
      block->height = 0;
      block->width = 0;
      block->value = NULL;
      return;
   }
   block->height = iTo - iFrom + 1;
   block->width = jTo - jFrom + 1;
   block->value = mtx->value + iFrom - 1 + (jFrom - 1) * mtx->globalHeight;
   block->ownsValues = 0;
}

void
utilities_FortranMatrixUpperInv( utilities_FortranMatrix* u )
{

   HYPRE_BigInt i, j, k;
   HYPRE_BigInt n, jc, jd;
   HYPRE_Real v;
   HYPRE_Real* diag;    /* diag(i) = u(i,i)_original */
   HYPRE_Real* pin;     /* &u(i-1,n) */
   HYPRE_Real* pii;     /* &u(i,i) */
   HYPRE_Real* pij;     /* &u(i,j) */
   HYPRE_Real* pik;     /* &u(i,k) */
   HYPRE_Real* pkj;     /* &u(k,j) */
   HYPRE_Real* pd;      /* &diag(i) */

   n = u->height;
   hypre_assert( u->width == n );

   diag = hypre_CTAlloc(HYPRE_Real,  n, HYPRE_MEMORY_HOST);
   hypre_assert( diag != NULL );

   jc = u->globalHeight;
   jd = jc + 1;

   pii = u->value;
   pd = diag;
   for ( i = 0; i < n; i++, pii += jd, pd++ )
   {
      v = *pd = *pii;
      *pii = 1.0 / v;
   }

   pii -= jd;
   pin = pii - 1;
   pii -= jd;
   pd -= 2;
   for ( i = n - 1; i > 0; i--, pii -= jd, pin--, pd-- )
   {
      pij = pin;
      for ( j = n; j > i; j--, pij -= jc )
      {
         v = 0;
         pik = pii + jc;
         pkj = pij + 1;
         for ( k = i + 1; k <= j; k++, pik += jc, pkj++  )
         {
            v -= (*pik) * (*pkj);
         }
         *pij = v / (*pd);
      }
   }

   hypre_TFree( diag, HYPRE_MEMORY_HOST);

}

HYPRE_Int
utilities_FortranMatrixPrint( utilities_FortranMatrix* mtx, const char *fileName)
{

   HYPRE_BigInt i, j, h, w, jump;
   HYPRE_Real* p;
   FILE* fp;

   hypre_assert( mtx != NULL );

   if ( !(fp = fopen(fileName, "w")) )
   {
      return 1;
   }

   h = mtx->height;
   w = mtx->width;

   hypre_fprintf(fp, "%ld\n", h);
   hypre_fprintf(fp, "%ld\n", w);

   jump = mtx->globalHeight - h;

   for ( j = 0, p = mtx->value; j < w; j++ )
   {
      for ( i = 0; i < h; i++, p++ )
      {
         hypre_fprintf(fp, "%.14e\n", *p);
      }
      p += jump;
   }

   fclose(fp);
   return 0;
}

