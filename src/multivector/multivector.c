/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "multivector.h"
#include "_hypre_utilities.h"

/* abstract multivector */
struct mv_MultiVector
{
   void*  data;      /* the pointer to the actual multivector */
   HYPRE_Int ownsData;

   mv_InterfaceInterpreter* interpreter; /* a structure that defines
                     multivector operations */
} ;

void *
mv_MultiVectorGetData (mv_MultiVectorPtr x)
{
   hypre_assert (x != NULL);
   return x->data;
}

mv_MultiVectorPtr
mv_MultiVectorWrap( mv_InterfaceInterpreter* ii, void * data, HYPRE_Int ownsData )
{
   mv_MultiVectorPtr x;

   x = hypre_TAlloc(struct mv_MultiVector, 1, HYPRE_MEMORY_HOST);
   hypre_assert( x != NULL );

   x->interpreter = ii;
   x->data = data;
   x->ownsData = ownsData;

   return x;
}

mv_MultiVectorPtr
mv_MultiVectorCreateFromSampleVector( void* ii_, HYPRE_Int n, void* sample )
{

   mv_MultiVectorPtr x;
   mv_InterfaceInterpreter* ii = (mv_InterfaceInterpreter*)ii_;

   x = hypre_TAlloc(struct mv_MultiVector, 1, HYPRE_MEMORY_HOST);
   hypre_assert( x != NULL );

   x->interpreter = ii;
   x->data = (ii->CreateMultiVector)( ii, n, sample );
   x->ownsData = 1;

   return x;
}

mv_MultiVectorPtr
mv_MultiVectorCreateCopy( mv_MultiVectorPtr x, HYPRE_Int copyValues )
{

   mv_MultiVectorPtr y;
   void* data;
   mv_InterfaceInterpreter* ii;

   hypre_assert( x != NULL );
   ii = x->interpreter;

   y = hypre_TAlloc(struct mv_MultiVector, 1, HYPRE_MEMORY_HOST);
   hypre_assert( y != NULL );

   data = (ii->CopyCreateMultiVector)( x->data, copyValues );

   y->interpreter = ii;
   y->data = data;
   y->ownsData = 1;

   return y;
}

void
mv_MultiVectorDestroy( mv_MultiVectorPtr v)
{

   if ( v == NULL )
   {
      return;
   }

   if ( v->ownsData )
   {
      (v->interpreter->DestroyMultiVector)( v->data );
   }
   hypre_TFree( v, HYPRE_MEMORY_HOST);
}

void
mv_MultiVectorSetMask( mv_MultiVectorPtr v, HYPRE_Int* mask )
{

   hypre_assert( v != NULL );
   (v->interpreter->SetMask)( v->data, mask );
}

HYPRE_Int
mv_MultiVectorWidth( mv_MultiVectorPtr v )
{

   if ( v == NULL )
   {
      return 0;
   }

   return (v->interpreter->Width)( v->data );
}

HYPRE_Int
mv_MultiVectorHeight( mv_MultiVectorPtr v )
{

   if ( v == NULL )
   {
      return 0;
   }

   return (v->interpreter->Height)(v->data);
}

void
mv_MultiVectorClear( mv_MultiVectorPtr v )
{

   hypre_assert( v != NULL );
   (v->interpreter->ClearMultiVector)( v->data );
}

void
mv_MultiVectorSetRandom( mv_MultiVectorPtr v, HYPRE_Int seed )
{

   hypre_assert( v != NULL );
   (v->interpreter->SetRandomVectors)( v->data, seed );
}

void
mv_MultiVectorCopy( mv_MultiVectorPtr src, mv_MultiVectorPtr dest )
{

   hypre_assert( src != NULL && dest != NULL );
   (src->interpreter->CopyMultiVector)( src->data, dest->data );
}

void
mv_MultiVectorAxpy( HYPRE_Complex a, mv_MultiVectorPtr x, mv_MultiVectorPtr y )
{

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->MultiAxpy)( a, x->data, y->data );
}

void
mv_MultiVectorByMultiVector( mv_MultiVectorPtr x, mv_MultiVectorPtr y,
                             HYPRE_BigInt xyGHeight, HYPRE_Int xyHeight,
                             HYPRE_Int xyWidth, HYPRE_Real* xy )
{
   /* xy = x'*y */

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->MultiInnerProd)
   ( x->data, y->data, xyGHeight, xyHeight, xyWidth, xy );
}

void
mv_MultiVectorByMultiVectorDiag( mv_MultiVectorPtr x, mv_MultiVectorPtr y,
                                 HYPRE_Int* mask, HYPRE_Int n, HYPRE_Real* d )
{
   /* d = diag(x'*y) */

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->MultiInnerProdDiag)( x->data, y->data, mask, n, d );
}

void
mv_MultiVectorByMatrix( mv_MultiVectorPtr x,
                        HYPRE_BigInt rGHeight, HYPRE_Int rHeight,
                        HYPRE_Int rWidth, HYPRE_Complex* rVal,
                        mv_MultiVectorPtr y )
{

   /* y = x*r */

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->MultiVecMat)
   ( x->data, rGHeight, rHeight, rWidth, rVal, y->data );
}

void
mv_MultiVectorXapy( mv_MultiVectorPtr x,
                    HYPRE_BigInt rGHeight, HYPRE_Int rHeight,
                    HYPRE_Int rWidth, HYPRE_Complex* rVal,
                    mv_MultiVectorPtr y )
{

   /* y = y + x*a */

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->MultiXapy)
   ( x->data, rGHeight, rHeight, rWidth, rVal, y->data );
}

void
mv_MultiVectorByDiagonal( mv_MultiVectorPtr x,
                          HYPRE_Int* mask, HYPRE_Int n, HYPRE_Complex* d,
                          mv_MultiVectorPtr y )
{

   /* y = x*d */

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->MultiVecMatDiag)( x->data, mask, n, d, y->data );
}

void
mv_MultiVectorEval( void (*f)( void*, void*, void* ), void* par,
                    mv_MultiVectorPtr x, mv_MultiVectorPtr y )
{

   /* y = f(x) computed vector-wise */

   hypre_assert( x != NULL && y != NULL );
   (x->interpreter->Eval)( f, par, x->data, y->data );
}
