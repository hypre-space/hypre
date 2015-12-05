/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/


#ifndef FORTRAN_STYLE_MATRIX
#define FORTRAN_STYLE_MATRIX

#include "_hypre_utilities.h"

typedef struct
{
  hypre_longint	globalHeight;
  hypre_longint	height;
  hypre_longint	width;
  double* value;
  HYPRE_Int		ownsValues;
} utilities_FortranMatrix;

#ifdef __cplusplus
extern "C" {
#endif

utilities_FortranMatrix* 
utilities_FortranMatrixCreate(void);
void 
utilities_FortranMatrixAllocateData( hypre_longint h, hypre_longint w, 
				     utilities_FortranMatrix* mtx );
void 
utilities_FortranMatrixWrap( double*, hypre_longint gh, hypre_longint h, hypre_longint w, 
			     utilities_FortranMatrix* mtx );
void 
utilities_FortranMatrixDestroy( utilities_FortranMatrix* mtx );

hypre_longint
utilities_FortranMatrixGlobalHeight( utilities_FortranMatrix* mtx );
hypre_longint
utilities_FortranMatrixHeight( utilities_FortranMatrix* mtx );
hypre_longint
utilities_FortranMatrixWidth( utilities_FortranMatrix* mtx );
double*
utilities_FortranMatrixValues( utilities_FortranMatrix* mtx );

void 
utilities_FortranMatrixClear( utilities_FortranMatrix* mtx );
void 
utilities_FortranMatrixClearL( utilities_FortranMatrix* mtx );
void 
utilities_FortranMatrixSetToIdentity( utilities_FortranMatrix* mtx );

void 
utilities_FortranMatrixTransposeSquare( utilities_FortranMatrix* mtx );
void 
utilities_FortranMatrixSymmetrize( utilities_FortranMatrix* mtx );

void 
utilities_FortranMatrixCopy( utilities_FortranMatrix* src, HYPRE_Int t, 
			     utilities_FortranMatrix* dest );
void 
utilities_FortranMatrixIndexCopy( HYPRE_Int* index, 
				  utilities_FortranMatrix* src, HYPRE_Int t, 
				  utilities_FortranMatrix* dest );

void 
utilities_FortranMatrixSetDiagonal( utilities_FortranMatrix* mtx, 
				    utilities_FortranMatrix* d );
void 
utilities_FortranMatrixGetDiagonal( utilities_FortranMatrix* mtx, 
				    utilities_FortranMatrix* d );
void 
utilities_FortranMatrixAdd( double a, 
			    utilities_FortranMatrix* mtxA, 
			    utilities_FortranMatrix* mtxB, 
			    utilities_FortranMatrix* mtxC );
void 
utilities_FortranMatrixDMultiply( utilities_FortranMatrix* d, 
				  utilities_FortranMatrix* mtx );
void 
utilities_FortranMatrixMultiplyD( utilities_FortranMatrix* mtx, 
				  utilities_FortranMatrix* d );
void 
utilities_FortranMatrixMultiply( utilities_FortranMatrix* mtxA, HYPRE_Int tA, 
				 utilities_FortranMatrix* mtxB, HYPRE_Int tB,
				 utilities_FortranMatrix* mtxC );
double 
utilities_FortranMatrixFNorm( utilities_FortranMatrix* mtx );

double 
utilities_FortranMatrixValue( utilities_FortranMatrix* mtx, 
			      hypre_longint i, hypre_longint j );
double* 
utilities_FortranMatrixValuePtr( utilities_FortranMatrix* mtx, 
				 hypre_longint i, hypre_longint j );
double 
utilities_FortranMatrixMaxValue( utilities_FortranMatrix* mtx );

void 
utilities_FortranMatrixSelectBlock( utilities_FortranMatrix* mtx,
				    hypre_longint iFrom, hypre_longint iTo, 
				    hypre_longint jFrom, hypre_longint jTo,
				    utilities_FortranMatrix* block );
void 
utilities_FortranMatrixUpperInv( utilities_FortranMatrix* u );

HYPRE_Int
utilities_FortranMatrixPrint( utilities_FortranMatrix* mtx, const char *fileName);

#ifdef __cplusplus
}
#endif

#endif /* FORTRAN_STYLE_MATRIX */

