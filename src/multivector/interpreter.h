/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.6 $
 ***********************************************************************EHEADER*/




#ifndef LOBPCG_INTERFACE_INTERPRETER
#define LOBPCG_INTERFACE_INTERPRETER

#include "_hypre_utilities.h"

typedef struct
{
  /* vector operations */
  void*  (*CreateVector)  ( void *vector );
  HYPRE_Int    (*DestroyVector) ( void *vector );

  double (*InnerProd)     ( void *x, void *y );
  HYPRE_Int    (*CopyVector)    ( void *x, void *y );
  HYPRE_Int    (*ClearVector)   ( void *x );
  HYPRE_Int    (*SetRandomValues)   ( void *x, HYPRE_Int seed );
  HYPRE_Int    (*ScaleVector)   ( double alpha, void *x );
  HYPRE_Int    (*Axpy)          ( double alpha, void *x, void *y );
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
  void   (*MultiInnerProd)     ( void *x, void *y, HYPRE_Int, HYPRE_Int, HYPRE_Int, double* );
  void   (*MultiInnerProdDiag) ( void *x, void *y, HYPRE_Int*, HYPRE_Int, double* );
  void   (*MultiVecMat)        ( void *x, HYPRE_Int, HYPRE_Int, HYPRE_Int, double*, void *y );
  void   (*MultiVecMatDiag)    ( void *x, HYPRE_Int*, HYPRE_Int, double*, void *y );
  void   (*MultiAxpy)          ( double alpha, void *x, void *y );

  /* do we need the following 2 entries? */
  void   (*MultiXapy)          ( void *x, HYPRE_Int, HYPRE_Int, HYPRE_Int, double*, void *y );
  void   (*Eval)               ( void (*f)( void*, void*, void* ), void*, void *x, void *y );

} mv_InterfaceInterpreter;

#endif
