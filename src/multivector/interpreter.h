/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef LOBPCG_INTERFACE_INTERPRETER
#define LOBPCG_INTERFACE_INTERPRETER

#include "_hypre_utilities.h"

typedef struct
{
  /* vector operations */
  void*  (*CreateVector)  ( void *vector );
  HYPRE_Int    (*DestroyVector) ( void *vector );

  HYPRE_Real   (*InnerProd)     ( void *x, void *y );
  HYPRE_Int    (*CopyVector)    ( void *x, void *y );
  HYPRE_Int    (*ClearVector)   ( void *x );
  HYPRE_Int    (*SetRandomValues)   ( void *x, HYPRE_Int seed );
  HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x );
  HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y );
  HYPRE_Int    (*VectorSize)    (void * vector);
  
  /* multivector operations */
  /* do we need the following entry? */
  void*  (*CreateMultiVector)  ( void*, HYPRE_Int n, void *vector );
  void*  (*CopyCreateMultiVector)  ( void *x, HYPRE_Int );
  void    (*DestroyMultiVector) ( void *x );

  HYPRE_Int    (*Width)  ( void *x );
  HYPRE_Int    (*Height) ( void *x );

  void   (*SetMask) ( void *x, HYPRE_Int *mask );

  void   (*CopyMultiVector)    ( void *x, void *y );
  void   (*ClearMultiVector)   ( void *x );
  void   (*SetRandomVectors)   ( void *x, HYPRE_Int seed );
  void   (*MultiInnerProd)     ( void *x, void *y, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Real* );
  void   (*MultiInnerProdDiag) ( void *x, void *y, HYPRE_Int*, HYPRE_Int, HYPRE_Real* );
  void   (*MultiVecMat)        ( void *x, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Complex*, void *y );
  void   (*MultiVecMatDiag)    ( void *x, HYPRE_Int*, HYPRE_Int, HYPRE_Complex*, void *y );
  void   (*MultiAxpy)          ( HYPRE_Complex alpha, void *x, void *y );

  /* do we need the following 2 entries? */
  void   (*MultiXapy)          ( void *x, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Complex*, void *y );
  void   (*Eval)               ( void (*f)( void*, void*, void* ), void*, void *x, void *y );

} mv_InterfaceInterpreter;

#endif
