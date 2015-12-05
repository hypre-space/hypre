/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.9 $
 ***********************************************************************EHEADER*/




#ifndef TEMPORARY_MULTIVECTOR_FUNCTION_PROTOTYPES
#define TEMPORARY_MULTIVECTOR_FUNCTION_PROTOTYPES

#include "interpreter.h"

typedef struct
{
  hypre_longint	 numVectors;
  HYPRE_Int*   mask;
  void** vector;
  HYPRE_Int	 ownsVectors;
  HYPRE_Int    ownsMask;
  
  mv_InterfaceInterpreter* interpreter;
  
} mv_TempMultiVector;

/*typedef struct mv_TempMultiVector* mv_TempMultiVectorPtr;  */
typedef  mv_TempMultiVector* mv_TempMultiVectorPtr;

/*******************************************************************/
/*
The above is a temporary implementation of the hypre_MultiVector
data type, just to get things going with LOBPCG eigensolver.

A more proper implementation would be to define hypre_MultiParVector,
hypre_MultiStructVector and hypre_MultiSStructVector by adding a new 
record

HYPRE_Int numVectors;

in hypre_ParVector, hypre_StructVector and hypre_SStructVector,
and increasing the size of data numVectors times. Respective
modifications of most vector operations are straightforward
(it is strongly suggested that BLAS routines are used wherever
possible), efficient implementation of matrix-by-multivector 
multiplication may be more difficult.

With the above implementation of hypre vectors, the definition
of hypre_MultiVector becomes simply (cf. multivector.h)

typedef struct
{
  void*	multiVector;
  HYPRE_InterfaceInterpreter* interpreter;  
} hypre_MultiVector;

with pointers to abstract multivector functions added to the structure
HYPRE_InterfaceInterpreter (cf. HYPRE_interpreter.h; particular values
are assigned to these pointers by functions 
HYPRE_ParCSRSetupInterpreter, HYPRE_StructSetupInterpreter and
HYPRE_Int HYPRE_SStructSetupInterpreter),
and the abstract multivector functions become simply interfaces
to the actual multivector functions of the form (cf. multivector.c):

void 
hypre_MultiVectorCopy( hypre_MultiVectorPtr src_, hypre_MultiVectorPtr dest_ ) {

  hypre_MultiVector* src = (hypre_MultiVector*)src_;
  hypre_MultiVector* dest = (hypre_MultiVector*)dest_;
  assert( src != NULL && dest != NULL );
  (src->interpreter->CopyMultiVector)( src->data, dest->data );
}


*/
/*********************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

void*
mv_TempMultiVectorCreateFromSampleVector( void*, HYPRE_Int n, void* sample );

void*
mv_TempMultiVectorCreateCopy( void*, HYPRE_Int copyValues );

void 
mv_TempMultiVectorDestroy( void* );

HYPRE_Int
mv_TempMultiVectorWidth( void* v );

HYPRE_Int
mv_TempMultiVectorHeight( void* v );

void
mv_TempMultiVectorSetMask( void* v, HYPRE_Int* mask );

void 
mv_TempMultiVectorClear( void* );

void 
mv_TempMultiVectorSetRandom( void* v, HYPRE_Int seed );

void 
mv_TempMultiVectorCopy( void* src, void* dest );

void 
mv_TempMultiVectorAxpy( double, void*, void* ); 

void 
mv_TempMultiVectorByMultiVector( void*, void*,
				    HYPRE_Int gh, HYPRE_Int h, HYPRE_Int w, double* v );

void 
mv_TempMultiVectorByMultiVectorDiag( void* x, void* y,
					HYPRE_Int* mask, HYPRE_Int n, double* diag );

void 
mv_TempMultiVectorByMatrix( void*, 
			       HYPRE_Int gh, HYPRE_Int h, HYPRE_Int w, double* v,
			       void* );

void 
mv_TempMultiVectorXapy( void* x, 
			   HYPRE_Int gh, HYPRE_Int h, HYPRE_Int w, double* v,
			   void* y );

void mv_TempMultiVectorByDiagonal( void* x, 
				      HYPRE_Int* mask, HYPRE_Int n, double* diag,
				      void* y );

void 
mv_TempMultiVectorEval( void (*f)( void*, void*, void* ), void* par,
			   void* x, void* y );

#ifdef __cplusplus
}
#endif

#endif /* MULTIVECTOR_FUNCTION_PROTOTYPES */

