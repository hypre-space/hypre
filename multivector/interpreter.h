/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




#ifndef LOBPCG_INTERFACE_INTERPRETER
#define LOBPCG_INTERFACE_INTERPRETER

typedef struct
{
  /* vector operations */
  void*  (*CreateVector)  ( void *vector );
  int    (*DestroyVector) ( void *vector );

  double (*InnerProd)     ( void *x, void *y );
  int    (*CopyVector)    ( void *x, void *y );
  int    (*ClearVector)   ( void *x );
  int    (*SetRandomValues)   ( void *x, int seed );
  int    (*ScaleVector)   ( double alpha, void *x );
  int    (*Axpy)          ( double alpha, void *x, void *y );
  int    (*VectorSize)    (void * vector);
  
  /* multivector operations */
  /* do we need the following entry? */
  void*  (*CreateMultiVector)  ( void*, int n, void *vector );
  void*  (*CopyCreateMultiVector)  ( void *x, int );
  void    (*DestroyMultiVector) ( void *x );

  int    (*Width)  ( void *x );
  int    (*Height) ( void *x );

  void   (*SetMask) ( void *x, int *mask );

  void   (*CopyMultiVector)    ( void *x, void *y );
  void   (*ClearMultiVector)   ( void *x );
  void   (*SetRandomVectors)   ( void *x, int seed );
  void   (*MultiInnerProd)     ( void *x, void *y, int, int, int, double* );
  void   (*MultiInnerProdDiag) ( void *x, void *y, int*, int, double* );
  void   (*MultiVecMat)        ( void *x, int, int, int, double*, void *y );
  void   (*MultiVecMatDiag)    ( void *x, int*, int, double*, void *y );
  void   (*MultiAxpy)          ( double alpha, void *x, void *y );

  /* do we need the following 2 entries? */
  void   (*MultiXapy)          ( void *x, int, int, int, double*, void *y );
  void   (*Eval)               ( void (*f)( void*, void*, void* ), void*, void *x, void *y );

} mv_InterfaceInterpreter;

#endif
