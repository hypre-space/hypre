/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.14 $
 ***********************************************************************EHEADER*/


/******************************************************************************
 *
 * Locally optimal preconditioned conjugate gradient functions
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "lobpcg.h"
#include "fortran_matrix.h"
#include "multivector.h"

static HYPRE_Int
lobpcg_chol( utilities_FortranMatrix* a, 
        HYPRE_Int (*dpotrf) (char *uplo, HYPRE_Int *n, double *a, HYPRE_Int *lda, HYPRE_Int *info) )
{

  HYPRE_Int lda, n;
  double* aval;
  char uplo;
  HYPRE_Int ierr;

  lda = utilities_FortranMatrixGlobalHeight( a );
  n = utilities_FortranMatrixHeight( a );
  aval = utilities_FortranMatrixValues( a );
  uplo = 'U';

  (*dpotrf)( &uplo, &n, aval, &lda, &ierr );

  return ierr;
}

static HYPRE_Int
lobpcg_solveGEVP( 
utilities_FortranMatrix* mtxA, 
utilities_FortranMatrix* mtxB,
utilities_FortranMatrix* eigVal,
HYPRE_Int   (*dsygv) (HYPRE_Int *itype, char *jobz, char *uplo, HYPRE_Int *
        n, double *a, HYPRE_Int *lda, double *b, HYPRE_Int *ldb,
        double *w, double *work, HYPRE_Int *lwork, HYPRE_Int *info)
){

  HYPRE_Int n, lda, ldb, itype, lwork, info;
  char jobz, uplo;
  double* work;
  double* a;
  double* b;
  double* lmd;

  itype = 1;
  jobz = 'V';
  uplo = 'L';
    
  a = utilities_FortranMatrixValues( mtxA );
  b = utilities_FortranMatrixValues( mtxB );
  lmd = utilities_FortranMatrixValues( eigVal );

  n = utilities_FortranMatrixHeight( mtxA );
  lda = utilities_FortranMatrixGlobalHeight( mtxA );
  ldb = utilities_FortranMatrixGlobalHeight( mtxB );
  lwork = 10*n;

  work = (double*)calloc( lwork, sizeof(double) );

  (*dsygv)( &itype, &jobz, &uplo, &n, 
				       a, &lda, b, &ldb,
				       lmd, &work[0], &lwork, &info );

  free( work );
  return info;

}


static void
lobpcg_MultiVectorByMultiVector(
mv_MultiVectorPtr x,
mv_MultiVectorPtr y,
utilities_FortranMatrix* xy
){
  mv_MultiVectorByMultiVector( x, y,
				  utilities_FortranMatrixGlobalHeight( xy ),
				  utilities_FortranMatrixHeight( xy ),
				  utilities_FortranMatrixWidth( xy ),
				  utilities_FortranMatrixValues( xy ) );
}

static void
lobpcg_MultiVectorByMatrix(
mv_MultiVectorPtr x,
utilities_FortranMatrix* r,
mv_MultiVectorPtr y
){
  mv_MultiVectorByMatrix( x, 
			     utilities_FortranMatrixGlobalHeight( r ),
			     utilities_FortranMatrixHeight( r ),
			     utilities_FortranMatrixWidth( r ),
			     utilities_FortranMatrixValues( r ),
			     y );
}

static HYPRE_Int
lobpcg_MultiVectorImplicitQR( 
mv_MultiVectorPtr x, mv_MultiVectorPtr y,
utilities_FortranMatrix* r,
mv_MultiVectorPtr z,
HYPRE_Int (*dpotrf) (char *uplo, HYPRE_Int *n, double *a, HYPRE_Int *lda, HYPRE_Int *info)

){

  /* B-orthonormalizes x using y = B x */

  HYPRE_Int ierr;

  lobpcg_MultiVectorByMultiVector( x, y, r );

  ierr = lobpcg_chol( r,dpotrf );

  if ( ierr != 0 )
    return ierr;

  utilities_FortranMatrixUpperInv( r );
  utilities_FortranMatrixClearL( r );

  mv_MultiVectorCopy( x, z );
  lobpcg_MultiVectorByMatrix( z, r, x );

  return 0;
}

static void
lobpcg_sqrtVector( HYPRE_Int n, HYPRE_Int* mask, double* v ) {

  HYPRE_Int i;

  for ( i = 0; i < n; i++ )
    if ( mask == NULL || mask[i] )
      v[i] = sqrt(v[i]);
}

static HYPRE_Int
lobpcg_checkResiduals( 
utilities_FortranMatrix* resNorms,
utilities_FortranMatrix* lambda,
lobpcg_Tolerance tol,
HYPRE_Int* activeMask
){
  HYPRE_Int i, n;
  HYPRE_Int notConverged;
  double atol;
  double rtol;

  n = utilities_FortranMatrixHeight( resNorms );

  atol = tol.absolute;
  rtol = tol.relative;

  notConverged = 0;
  for ( i = 0; i < n; i++ ) {
    if ( utilities_FortranMatrixValue( resNorms, i + 1, 1 ) >
	 utilities_FortranMatrixValue( lambda, i + 1, 1 )*rtol + atol
	 + DBL_EPSILON ) {
      activeMask[i] = 1; 
      notConverged++;
    }
    else
      activeMask[i] = 0;
  }
  return notConverged;
}

static void
lobpcg_errorMessage( HYPRE_Int verbosityLevel, const char* message )
{
  if ( verbosityLevel ) {
    hypre_fprintf( stderr, "Error in LOBPCG:\n" );
    hypre_fprintf( stderr, "%s", message );
  }
}

HYPRE_Int
lobpcg_solve( mv_MultiVectorPtr blockVectorX,
	      void* operatorAData,
	      void (*operatorA)( void*, void*, void* ),
	      void* operatorBData,
	      void (*operatorB)( void*, void*, void* ),
	      void* operatorTData,
	      void (*operatorT)( void*, void*, void* ),
	      mv_MultiVectorPtr blockVectorY,
              lobpcg_BLASLAPACKFunctions blap_fn,
	      lobpcg_Tolerance tolerance,
	      HYPRE_Int maxIterations,
	      HYPRE_Int verbosityLevel,
	      HYPRE_Int* iterationNumber,

/* eigenvalues; "lambda_values" should point to array  containing <blocksize> doubles where <blocksi
ze> is the width of multivector "blockVectorX" */
              double * lambda_values,

/* eigenvalues history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matrix s
tored
in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see next
argument; If you don't need eigenvalues history, provide NULL in this entry */
              double * lambdaHistory_values,

/* global height of the matrix (stored in fotran-style)  specified by previous argument */
              HYPRE_Int lambdaHistory_gh,

/* residual norms; argument should point to array of <blocksize> doubles */
              double * residualNorms_values,

/* residual norms history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matri
x
stored in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see
next
argument If you don't need residual norms history, provide NULL in this entry */
              double * residualNormsHistory_values ,

/* global height of the matrix (stored in fotran-style)  specified by previous argument */
              HYPRE_Int residualNormsHistory_gh

){

  HYPRE_Int				sizeX; /* number of eigenvectors */
  HYPRE_Int				sizeY; /* number of constraints */
  HYPRE_Int				sizeR; /* number of residuals used */
  HYPRE_Int				sizeP; /* number of conj. directions used */
  HYPRE_Int				sizeA; /* size of the Gram matrix for A */
  HYPRE_Int				sizeX3; /* 3*sizeX */

  HYPRE_Int				firstR; /* first line of the Gram block
					   corresponding to residuals */
  HYPRE_Int				lastR; /* last line of this block */
  HYPRE_Int				firstP; /* same for conjugate directions */
  HYPRE_Int				lastP;

  HYPRE_Int				noTFlag; /* nonzero: no preconditioner */
  HYPRE_Int				noBFlag; /* nonzero: no operator B */
  HYPRE_Int				noYFlag; /* nonzero: no constaints */

  HYPRE_Int				exitFlag; /* 1: problem size is too small,
					     2: block size < 1,
					     3: linearly dependent constraints,
					     -1: requested accuracy not 
					     achieved */

  HYPRE_Int*				activeMask; /* soft locking mask */

  HYPRE_Int				i; /* short loop counter */

#if 0
  hypre_longint				n; /* dimension 1 of X */
  /* had to remove because n is not available in some interfaces */ 
#endif 

  mv_MultiVectorPtr		blockVectorR; /* residuals */
  mv_MultiVectorPtr		blockVectorP; /* conjugate directions */

  mv_MultiVectorPtr		blockVectorW; /* auxiliary block vector */

  mv_MultiVectorPtr		blockVectorAX; /* A*X */
  mv_MultiVectorPtr		blockVectorAR; /* A*R */
  mv_MultiVectorPtr		blockVectorAP; /* A*P */

  mv_MultiVectorPtr		blockVectorBX; /* B*X */
  mv_MultiVectorPtr		blockVectorBR; /* B*R */
  mv_MultiVectorPtr		blockVectorBP; /* B*P */

  mv_MultiVectorPtr		blockVectorBY; /* B*Y */

  utilities_FortranMatrix*	gramA; /* Gram matrix for A */
  utilities_FortranMatrix*	gramB; /* Gram matrix for B */
  utilities_FortranMatrix*	lambdaAB; /* eigenvalues of 
					     gramA u = lambda gram B u */
  utilities_FortranMatrix*	lambdaX; /* first sizeX eigenvalues in
					    lambdaAB (ref) */

  utilities_FortranMatrix*	gramXAX; /* XX block of gramA (ref) */
  utilities_FortranMatrix*	gramRAX; /* XR block of gramA (ref) */
  utilities_FortranMatrix*	gramPAX; /* XP block of gramA (ref) */

  utilities_FortranMatrix*	gramRAR; /* RR block of gramA (ref) */
  utilities_FortranMatrix*	gramPAR; /* RP block of gramA (ref) */
	
  utilities_FortranMatrix*	gramPAP; /* PP block of gramA (ref) */

  utilities_FortranMatrix*	gramXBX; /* XX block of gramB (ref) */
  utilities_FortranMatrix*	gramRBX; /* XR block of gramB (ref) */
  utilities_FortranMatrix*	gramPBX; /* XP block of gramB (ref) */

  utilities_FortranMatrix*	gramRBR; /* RR block of gramB (ref) */
  utilities_FortranMatrix*	gramPBR; /* RP block of gramB (ref) */
	
  utilities_FortranMatrix*	gramPBP; /* PP block of gramB (ref) */
	
  utilities_FortranMatrix*	gramYBY; /* Matrices for constraints */
  utilities_FortranMatrix*	gramYBX; 
  utilities_FortranMatrix*	tempYBX; 
  utilities_FortranMatrix*	gramYBR; /* ref. */
  utilities_FortranMatrix*	tempYBR; /* ref. */ 

  utilities_FortranMatrix*	coordX; /* coordinates of the first sizeX
					   Ritz vectors in the XRP basis */
  utilities_FortranMatrix*	coordXX; /* coordinates of the above in X */
  utilities_FortranMatrix*	coordRX; /* coordinates of the above in R */
  utilities_FortranMatrix*	coordPX; /* coordinates of the above in P */

  utilities_FortranMatrix*	upperR; /* R factor in QR-fact. (ref) */
  utilities_FortranMatrix*	historyColumn; /* reference to a column
						  in history matrices */
  utilities_FortranMatrix* lambda;
  utilities_FortranMatrix* lambdaHistory;
  utilities_FortranMatrix* residualNorms;
  utilities_FortranMatrix* residualNormsHistory;

  /* initialization */

  exitFlag = 0;
  *iterationNumber = 0;
  noTFlag = operatorT == NULL;
  noBFlag = operatorB == NULL;

  sizeY = mv_MultiVectorWidth( blockVectorY );
  noYFlag = sizeY == 0;

  sizeX = mv_MultiVectorWidth( blockVectorX );

  lambda = utilities_FortranMatrixCreate();
  utilities_FortranMatrixWrap(lambda_values, sizeX, sizeX, 1, lambda);

/* prepare to process eigenvalues history, if user has provided non-NULL as "lambdaHistory_values" a
rgument */
  if (lambdaHistory_values!=NULL)
  {
      lambdaHistory = utilities_FortranMatrixCreate();
      utilities_FortranMatrixWrap(lambdaHistory_values, lambdaHistory_gh, sizeX,
                                    maxIterations+1, lambdaHistory);
  }
  else
      lambdaHistory = NULL;

  residualNorms = utilities_FortranMatrixCreate();
  utilities_FortranMatrixWrap(residualNorms_values, sizeX, sizeX, 1, residualNorms);

/* prepare to process residuals history, if user has provided non-NULL as "residualNormsHistory_valu
es" argument */
  if (residualNormsHistory_values!=NULL)
  {
      residualNormsHistory = utilities_FortranMatrixCreate();
      utilities_FortranMatrixWrap(residualNormsHistory_values, residualNormsHistory_gh,
                                 sizeX, maxIterations+1,residualNormsHistory);
  }
  else
      residualNormsHistory = NULL;

#if 0
  /* had to remove because n is not available in some interfaces */ 
  n = mv_MultiVectorHeight( blockVectorX );

  if ( n < 5*sizeX ) {
    exitFlag = PROBLEM_SIZE_TOO_SMALL;
    lobpcg_errorMessage( verbosityLevel,
			 "Problem size too small compared to block size\n" );
    return exitFlag;
  }
#endif
  
  if ( sizeX < 1 ) {
    exitFlag = WRONG_BLOCK_SIZE;
    lobpcg_errorMessage( verbosityLevel,
			 "The bloc size is wrong.\n" );
    return exitFlag;
  }

  gramYBY = utilities_FortranMatrixCreate();
  gramYBX = utilities_FortranMatrixCreate();
  tempYBX = utilities_FortranMatrixCreate();
  gramYBR = utilities_FortranMatrixCreate();
  tempYBR = utilities_FortranMatrixCreate();

  blockVectorW = mv_MultiVectorCreateCopy( blockVectorX, 0 );

  if ( !noYFlag ) {
    utilities_FortranMatrixAllocateData( sizeY, sizeY, gramYBY );
    utilities_FortranMatrixAllocateData( sizeY, sizeX, gramYBX );
    utilities_FortranMatrixAllocateData( sizeY, sizeX, tempYBX );
    blockVectorBY = blockVectorY;
    if ( !noBFlag ) {      
      blockVectorBY = mv_MultiVectorCreateCopy( blockVectorY, 0 );
      operatorB( operatorBData, mv_MultiVectorGetData(blockVectorY), 
                 mv_MultiVectorGetData(blockVectorBY) );
    };

    lobpcg_MultiVectorByMultiVector( blockVectorBY, blockVectorY, gramYBY );
    exitFlag = lobpcg_chol( gramYBY, blap_fn.dpotrf );
    if ( exitFlag != 0 ) {
      if ( verbosityLevel )
	hypre_printf("Cannot handle linear dependent constraints\n");
      utilities_FortranMatrixDestroy( gramYBY );
      utilities_FortranMatrixDestroy( gramYBX );
      utilities_FortranMatrixDestroy( tempYBX );
      utilities_FortranMatrixDestroy( gramYBR );
      utilities_FortranMatrixDestroy( tempYBR );
      if ( !noBFlag )
	mv_MultiVectorDestroy( blockVectorBY );
      mv_MultiVectorDestroy( blockVectorW );
      return WRONG_CONSTRAINTS;
    }      
    utilities_FortranMatrixUpperInv( gramYBY );
    utilities_FortranMatrixClearL( gramYBY );

    /* apply the constraints to the initial X */
    lobpcg_MultiVectorByMultiVector( blockVectorBY, blockVectorX, gramYBX );
    utilities_FortranMatrixMultiply( gramYBY, 1, gramYBX, 0, tempYBX );
    utilities_FortranMatrixMultiply( gramYBY, 0, tempYBX, 0, gramYBX );
    lobpcg_MultiVectorByMatrix( blockVectorY, gramYBX, blockVectorW );
    mv_MultiVectorAxpy( -1.0, blockVectorW, blockVectorX );
  }

  if ( verbosityLevel ) {
    hypre_printf("\nSolving ");
    if ( noBFlag )
      hypre_printf("standard");
    else
      hypre_printf("generalized");
    hypre_printf(" eigenvalue problem with");
    if ( noTFlag )
      hypre_printf("out");
    hypre_printf(" preconditioning\n\n");
    hypre_printf("block size %d\n\n", sizeX );
    if ( noYFlag )
      hypre_printf("No constraints\n\n");
    else {
      if ( sizeY > 1 )
	hypre_printf("%d constraints\n\n", sizeY);
      else
	hypre_printf("%d constraint\n\n", sizeY);
    }
  }

  /* creating fortran matrix shells */

  gramA = utilities_FortranMatrixCreate();
  gramB = utilities_FortranMatrixCreate();
  lambdaAB = utilities_FortranMatrixCreate();
  lambdaX = utilities_FortranMatrixCreate();

  gramXAX = utilities_FortranMatrixCreate();
  gramRAX = utilities_FortranMatrixCreate();
  gramPAX = utilities_FortranMatrixCreate();

  gramRAR = utilities_FortranMatrixCreate();
  gramPAR = utilities_FortranMatrixCreate();
	
  gramPAP = utilities_FortranMatrixCreate();

  gramXBX = utilities_FortranMatrixCreate();
  gramRBX = utilities_FortranMatrixCreate();
  gramPBX = utilities_FortranMatrixCreate();

  gramRBR = utilities_FortranMatrixCreate();
  gramPBR = utilities_FortranMatrixCreate();
	
  gramPBP = utilities_FortranMatrixCreate();
	
  coordX = utilities_FortranMatrixCreate();
  coordXX = utilities_FortranMatrixCreate();
  coordRX = utilities_FortranMatrixCreate();
  coordPX = utilities_FortranMatrixCreate();

  upperR = utilities_FortranMatrixCreate();
  historyColumn = utilities_FortranMatrixCreate();
  
  /* initializing soft locking mask */
  activeMask = (HYPRE_Int*)calloc( sizeX, sizeof(HYPRE_Int) );
  hypre_assert( activeMask != NULL );
  for ( i = 0; i < sizeX; i++ )
    activeMask[i] = 1;

  /* allocate memory for Gram matrices and the Ritz values */
  sizeX3 = 3*sizeX;
  utilities_FortranMatrixAllocateData( sizeX3, sizeX3, gramA );
  utilities_FortranMatrixAllocateData( sizeX3, sizeX3, gramB );
  utilities_FortranMatrixAllocateData( sizeX3, 1, lambdaAB );

  /* creating block vectors R, P, AX, AR, AP, BX, BR, BP and W */
  blockVectorR = mv_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorP = mv_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAX = mv_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAR = mv_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAP = mv_MultiVectorCreateCopy( blockVectorX, 0 );

  if ( !noBFlag ) {
    blockVectorBX = mv_MultiVectorCreateCopy( blockVectorX, 0 );
    blockVectorBR = mv_MultiVectorCreateCopy( blockVectorX, 0 );
    blockVectorBP = mv_MultiVectorCreateCopy( blockVectorX, 0 );
  }
  else {
    blockVectorBX = blockVectorX;
    blockVectorBR = blockVectorR;
    blockVectorBP = blockVectorP;
  }

  mv_MultiVectorSetMask( blockVectorR, activeMask );
  mv_MultiVectorSetMask( blockVectorP, activeMask );
  mv_MultiVectorSetMask( blockVectorAR, activeMask );
  mv_MultiVectorSetMask( blockVectorAP, activeMask );
  if ( !noBFlag ) {
    mv_MultiVectorSetMask( blockVectorBR, activeMask );
    mv_MultiVectorSetMask( blockVectorBP, activeMask );
  }
  mv_MultiVectorSetMask( blockVectorW, activeMask );

  /* B-orthonormaliization of X */
  /* selecting a block in gramB for R factor upperR */
  utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, upperR );
  if ( !noBFlag ) {
    operatorB( operatorBData, mv_MultiVectorGetData(blockVectorX), 
                              mv_MultiVectorGetData(blockVectorBX) );
  }
  exitFlag = lobpcg_MultiVectorImplicitQR( blockVectorX, blockVectorBX, 
					   upperR, blockVectorW,blap_fn.dpotrf );
  if ( exitFlag ) {
    lobpcg_errorMessage( verbosityLevel, "Bad initial vectors: orthonormalization failed\n" );
    if ( verbosityLevel )
      hypre_printf("DPOTRF INFO = %d\n", exitFlag);
  }
  else {

    if ( !noBFlag ) { /* update BX */
      lobpcg_MultiVectorByMatrix( blockVectorBX, upperR, blockVectorW );
      mv_MultiVectorCopy( blockVectorW, blockVectorBX );
    }

    operatorA( operatorAData, mv_MultiVectorGetData(blockVectorX), 
               mv_MultiVectorGetData(blockVectorAX) );

    /* gramXAX = X'*AX */
    utilities_FortranMatrixSelectBlock( gramA, 1, sizeX, 1, sizeX, gramXAX );
    lobpcg_MultiVectorByMultiVector( blockVectorX, blockVectorAX, gramXAX );
    utilities_FortranMatrixSymmetrize( gramXAX );

    /* gramXBX = X'*X */ 
    utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, gramXBX );
    lobpcg_MultiVectorByMultiVector( blockVectorX, blockVectorBX, gramXBX );
    utilities_FortranMatrixSymmetrize( gramXBX );
    /*  utilities_FortranMatrixSetToIdentity( gramXBX );*/ /* X may be bad! */
    
    if ( (exitFlag = lobpcg_solveGEVP( gramXAX, gramXBX, lambda,blap_fn.dsygv)) != 0 ) {
      lobpcg_errorMessage( verbosityLevel, 
			   "Bad problem: Rayleigh-Ritz in the initial subspace failed\n" );
      if ( verbosityLevel )
	hypre_printf("DSYGV INFO = %d\n", exitFlag);
    }
    else {
      utilities_FortranMatrixSelectBlock( gramXAX, 1, sizeX, 1, sizeX, coordX );

      lobpcg_MultiVectorByMatrix( blockVectorX, coordX, blockVectorW );
      mv_MultiVectorCopy( blockVectorW, blockVectorX );

      lobpcg_MultiVectorByMatrix( blockVectorAX, coordX, blockVectorW );
      mv_MultiVectorCopy( blockVectorW, blockVectorAX );

      if ( !noBFlag ) {
	lobpcg_MultiVectorByMatrix( blockVectorBX, coordX, blockVectorW );
	mv_MultiVectorCopy( blockVectorW, blockVectorBX );
      }

      /*
      lobpcg_MultiVectorByMultiVector( blockVectorBX, blockVectorX, upperR );
      utilities_FortranMatrixPrint( upperR, "xbx.dat" );
      utilities_FortranMatrixPrint( lambda, "lmd.dat" );
      */

      mv_MultiVectorByDiagonal( blockVectorBX, 
				   NULL, sizeX, 
				   utilities_FortranMatrixValues( lambda ),
				   blockVectorR );

      mv_MultiVectorAxpy( -1.0, blockVectorAX, blockVectorR );

      mv_MultiVectorByMultiVectorDiag( blockVectorR, blockVectorR, 
					  NULL, sizeX, 
					  utilities_FortranMatrixValues( residualNorms ) ); 

      lobpcg_sqrtVector( sizeX, NULL, 
			 utilities_FortranMatrixValues( residualNorms ) );

      if ( lambdaHistory != NULL ) {
	utilities_FortranMatrixSelectBlock( lambdaHistory, 1, sizeX, 1, 1, 
					    historyColumn );
	utilities_FortranMatrixCopy( lambda, 0, historyColumn );
      }
	
      if ( residualNormsHistory != NULL ) {
	utilities_FortranMatrixSelectBlock( residualNormsHistory, 1, sizeX, 1, 1, 
					    historyColumn );
	utilities_FortranMatrixCopy( residualNorms, 0, historyColumn );
      }
	
      if ( verbosityLevel == 2 ) {
	hypre_printf("\n");
	for (i = 1; i <= sizeX; i++ ) 
	  hypre_printf("Initial eigenvalues lambda %22.14e\n",
		 utilities_FortranMatrixValue( lambda, i, 1) );
	for (i = 1; i <= sizeX; i++) 
	  hypre_printf("Initial residuals %12.6e\n",
		 utilities_FortranMatrixValue( residualNorms, i, 1) );
      }
      else if ( verbosityLevel == 1 )
	hypre_printf("\nInitial Max. Residual %22.14e\n",
	       utilities_FortranMatrixMaxValue( residualNorms ) );
    }
  }

  for ( *iterationNumber = 1; exitFlag == 0 && *iterationNumber <= maxIterations; 
	(*iterationNumber)++ ) {
    
    sizeR = lobpcg_checkResiduals( residualNorms, lambda, tolerance, 
				   activeMask );
    if ( sizeR < 1 )
      break;

/* following code added by Ilya Lashuk on March 22, 2005; with current 
   multivector implementation mask needs to be reset after it has changed on each vector
   mask applies to */
   
    mv_MultiVectorSetMask( blockVectorR, activeMask );
    mv_MultiVectorSetMask( blockVectorP, activeMask );
    mv_MultiVectorSetMask( blockVectorAR, activeMask );
    mv_MultiVectorSetMask( blockVectorAP, activeMask );
    if ( !noBFlag ) {
      mv_MultiVectorSetMask( blockVectorBR, activeMask );
      mv_MultiVectorSetMask( blockVectorBP, activeMask );
    }
    mv_MultiVectorSetMask( blockVectorW, activeMask );

/* ***** end of added code ***** */

    if ( !noTFlag ) {
      operatorT( operatorTData, mv_MultiVectorGetData(blockVectorR), 
                 mv_MultiVectorGetData(blockVectorW) );
      mv_MultiVectorCopy( blockVectorW, blockVectorR );
    }

    if ( !noYFlag ) { /* apply the constraints to R  */
      utilities_FortranMatrixSelectBlock( gramYBX, 1, sizeY, 1, sizeR, gramYBR );
      utilities_FortranMatrixSelectBlock( tempYBX, 1, sizeY, 1, sizeR, tempYBR );

      lobpcg_MultiVectorByMultiVector( blockVectorBY, blockVectorR, gramYBR );
      utilities_FortranMatrixMultiply( gramYBY, 1, gramYBR, 0, tempYBR );
      utilities_FortranMatrixMultiply( gramYBY, 0, tempYBR, 0, gramYBR );
      lobpcg_MultiVectorByMatrix( blockVectorY, gramYBR, blockVectorW );
      mv_MultiVectorAxpy( -1.0, blockVectorW, blockVectorR );
    }

    firstR = sizeX + 1;
    lastR = sizeX + sizeR;
    firstP = lastR + 1;

    utilities_FortranMatrixSelectBlock( gramB, firstR, lastR, firstR, lastR, upperR );

    if ( !noBFlag ) {
      operatorB( operatorBData, mv_MultiVectorGetData(blockVectorR), 
                 mv_MultiVectorGetData(blockVectorBR) );
    }
    exitFlag = lobpcg_MultiVectorImplicitQR( blockVectorR, blockVectorBR, 
					     upperR, blockVectorW,blap_fn.dpotrf );
    if ( exitFlag ) {
      lobpcg_errorMessage( verbosityLevel, "Orthonormalization of residuals failed\n" );
      if ( verbosityLevel )
	hypre_printf("DPOTRF INFO = %d\n", exitFlag);
      break;
    }

    if ( !noBFlag ) { /* update BR */
      lobpcg_MultiVectorByMatrix( blockVectorBR, upperR, blockVectorW );
      mv_MultiVectorCopy( blockVectorW, blockVectorBR );
    }

    /* AR = A*R */
    operatorA( operatorAData, mv_MultiVectorGetData(blockVectorR), 
               mv_MultiVectorGetData(blockVectorAR) );

    if ( *iterationNumber > 1 ) {

      sizeP = sizeR;
      lastP = lastR + sizeP;

      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstP, lastP, upperR );

      exitFlag = lobpcg_MultiVectorImplicitQR( blockVectorP, blockVectorBP, 
					       upperR, blockVectorW,blap_fn.dpotrf );
      if ( exitFlag ) {
	/*
	lobpcg_errorMessage( verbosityLevel, "Orthonormalization of P failed\n" );
	if ( verbosityLevel )
	  hypre_printf("DPOTRF INFO = %d\n", exitFlag);
	*/
	sizeP = 0;
      }
      else {
			
	if ( !noBFlag ) { /* update BP */
	  lobpcg_MultiVectorByMatrix( blockVectorBP, upperR, blockVectorW );
	  mv_MultiVectorCopy( blockVectorW, blockVectorBP );
	}

	/* update AP */
	lobpcg_MultiVectorByMatrix( blockVectorAP, upperR, blockVectorW );
	mv_MultiVectorCopy( blockVectorW, blockVectorAP );
      }
    }
    else {
      
      sizeP = 0;
      lastP = lastR;
    }

    sizeA = lastR + sizeP;

    utilities_FortranMatrixSelectBlock( gramA, 1, sizeX, 1, sizeX, gramXAX );
    utilities_FortranMatrixSelectBlock( gramA, firstR, lastR, 1, sizeX, 
					gramRAX );
    utilities_FortranMatrixSelectBlock( gramA, firstR, lastR, firstR, lastR, 
					gramRAR );

    utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, gramXBX );
    utilities_FortranMatrixSelectBlock( gramB, firstR, lastR, 1, sizeX, 
					gramRBX );
    utilities_FortranMatrixSelectBlock( gramB, firstR, lastR, firstR, lastR, 
					gramRBR );

    utilities_FortranMatrixClear( gramXAX );
    utilities_FortranMatrixSetDiagonal( gramXAX, lambda );
		
    lobpcg_MultiVectorByMultiVector( blockVectorR, blockVectorAX, gramRAX );
		
    lobpcg_MultiVectorByMultiVector( blockVectorR, blockVectorAR, gramRAR );
    utilities_FortranMatrixSymmetrize( gramRAR );

    utilities_FortranMatrixSetToIdentity( gramXBX );

    lobpcg_MultiVectorByMultiVector( blockVectorR, blockVectorBX, gramRBX );
		
    utilities_FortranMatrixSetToIdentity( gramRBR );

    if ( *iterationNumber > 1 ) {
      
      utilities_FortranMatrixSelectBlock( gramA, firstP, lastP, 1, sizeX, gramPAX );
      utilities_FortranMatrixSelectBlock( gramA, firstP, lastP, firstR, lastR, gramPAR );
      utilities_FortranMatrixSelectBlock( gramA, firstP, lastP, firstP, lastP, gramPAP );
			
      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, 1, sizeX, gramPBX );
      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstR, lastR, gramPBR );
      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstP, lastP, gramPBP );

      lobpcg_MultiVectorByMultiVector( blockVectorP, blockVectorAX, gramPAX );
			
      lobpcg_MultiVectorByMultiVector( blockVectorP, blockVectorAR, gramPAR );
			
      lobpcg_MultiVectorByMultiVector( blockVectorP, blockVectorAP, gramPAP );
      utilities_FortranMatrixSymmetrize( gramPAP );

      lobpcg_MultiVectorByMultiVector( blockVectorP, blockVectorBX, gramPBX );
			
      lobpcg_MultiVectorByMultiVector( blockVectorP, blockVectorBR, gramPBR );
		
      utilities_FortranMatrixSetToIdentity( gramPBP );
    }
	
    utilities_FortranMatrixSelectBlock( gramA, 1, sizeA, 1, sizeA, gramXAX );
    utilities_FortranMatrixSelectBlock( gramB, 1, sizeA, 1, sizeA, gramXBX );

    if ( (exitFlag = lobpcg_solveGEVP( gramXAX, gramXBX, lambdaAB, blap_fn.dsygv )) != 0 ) {
      lobpcg_errorMessage( verbosityLevel, "GEVP solver failure\n" );
      (*iterationNumber)--;
      /* if ( verbosityLevel )
	 hypre_printf("INFO = %d\n", exitFlag);*/
      break;
    }

    utilities_FortranMatrixSelectBlock( lambdaAB, 1, sizeX, 1, 1, lambdaX );
    utilities_FortranMatrixCopy( lambdaX, 0, lambda );

    utilities_FortranMatrixSelectBlock( gramA, 1, sizeA, 1, sizeX, coordX );

    utilities_FortranMatrixSelectBlock( coordX, 1, sizeX, 1, sizeX, coordXX );
    utilities_FortranMatrixSelectBlock( coordX, firstR, lastR, 1, sizeX, coordRX );

    if ( *iterationNumber > 1 ) {
	
      utilities_FortranMatrixSelectBlock( coordX, firstP, lastP, 1, sizeX, coordPX );

      mv_MultiVectorSetMask( blockVectorW, NULL );
      lobpcg_MultiVectorByMatrix( blockVectorP, coordPX, blockVectorW );
      mv_MultiVectorSetMask( blockVectorP, NULL );
      mv_MultiVectorCopy( blockVectorW, blockVectorP );

      lobpcg_MultiVectorByMatrix( blockVectorAP, coordPX, blockVectorW );
      mv_MultiVectorSetMask( blockVectorAP, NULL );
      mv_MultiVectorCopy( blockVectorW, blockVectorAP );

      if ( !noBFlag ) {
	lobpcg_MultiVectorByMatrix( blockVectorBP, coordPX, blockVectorW );
	mv_MultiVectorSetMask( blockVectorBP, NULL );
	mv_MultiVectorCopy( blockVectorW, blockVectorBP );
      }

      lobpcg_MultiVectorByMatrix( blockVectorR, coordRX, blockVectorW );
      mv_MultiVectorAxpy( 1.0, blockVectorW, blockVectorP );
			
      lobpcg_MultiVectorByMatrix( blockVectorAR, coordRX, blockVectorW );
      mv_MultiVectorAxpy( 1.0, blockVectorW, blockVectorAP );

      if ( !noBFlag ) {
	lobpcg_MultiVectorByMatrix( blockVectorBR, coordRX, blockVectorW );
	mv_MultiVectorAxpy( 1.0, blockVectorW, blockVectorBP );
      }

    }
    else {
      
      mv_MultiVectorSetMask( blockVectorP, NULL );
      lobpcg_MultiVectorByMatrix( blockVectorR, coordRX, blockVectorP );
			
      mv_MultiVectorSetMask( blockVectorAP, NULL );
      lobpcg_MultiVectorByMatrix( blockVectorAR, coordRX, blockVectorAP );

      if ( !noBFlag ) {
	mv_MultiVectorSetMask( blockVectorBP, NULL );
	lobpcg_MultiVectorByMatrix( blockVectorBR, coordRX, blockVectorBP );
      }
		
    }

/* follwing line is bug fix in Google Rev 8 of code, by ilya.lashuk Aug 29,2008   */
    mv_MultiVectorSetMask( blockVectorW, NULL );
    
    mv_MultiVectorCopy( blockVectorX, blockVectorW );
    lobpcg_MultiVectorByMatrix( blockVectorW, coordXX, blockVectorX );
    mv_MultiVectorAxpy( 1.0, blockVectorP, blockVectorX );

    mv_MultiVectorCopy( blockVectorAX, blockVectorW );
    lobpcg_MultiVectorByMatrix( blockVectorW, coordXX, blockVectorAX );
    mv_MultiVectorAxpy( 1.0, blockVectorAP, blockVectorAX );

    if ( !noBFlag ) {
      mv_MultiVectorCopy( blockVectorBX, blockVectorW );
      lobpcg_MultiVectorByMatrix( blockVectorW, coordXX, blockVectorBX );
      mv_MultiVectorAxpy( 1.0, blockVectorBP, blockVectorBX );
    }

    mv_MultiVectorSetMask( blockVectorAX, activeMask );
    mv_MultiVectorSetMask( blockVectorBX, activeMask );

    mv_MultiVectorByDiagonal( blockVectorBX, 
				 activeMask, sizeX, 
				 utilities_FortranMatrixValues( lambda ),
				 blockVectorR );

    mv_MultiVectorAxpy( -1.0, blockVectorAX, blockVectorR );

    mv_MultiVectorByMultiVectorDiag(	blockVectorR, blockVectorR, 
					activeMask, sizeX, 
					utilities_FortranMatrixValues( residualNorms ) );
    lobpcg_sqrtVector( 	sizeX, activeMask, 
			utilities_FortranMatrixValues( residualNorms ) );

    i = *iterationNumber + 1;
    if ( lambdaHistory != NULL ) {
      utilities_FortranMatrixSelectBlock( lambdaHistory, 1, sizeX, i, i, 
					  historyColumn );
      utilities_FortranMatrixCopy( lambda, 0, historyColumn );
    }
	
    if ( residualNormsHistory != NULL ) {
      utilities_FortranMatrixSelectBlock( residualNormsHistory, 1, sizeX, i, i, 
					  historyColumn );
      utilities_FortranMatrixCopy( residualNorms, 0, historyColumn );
    }
    
    if ( verbosityLevel == 2 ) {
      hypre_printf( "Iteration %d \tbsize %d\n", *iterationNumber, sizeR );
      for ( i = 1; i <= sizeX; i++ ) 
	hypre_printf("Eigenvalue lambda %22.14e\n",
	       utilities_FortranMatrixValue( lambda, i, 1) );
      for ( i = 1; i <= sizeX; i++ ) 
	hypre_printf("Residual %12.6e\n",
	       utilities_FortranMatrixValue( residualNorms, i, 1) );
    }
    else if ( verbosityLevel == 1 )
      hypre_printf("Iteration %d \tbsize %d \tmaxres %22.14e\n",
	     *iterationNumber, sizeR, 
	     utilities_FortranMatrixMaxValue( residualNorms ) );

    mv_MultiVectorSetMask( blockVectorAX, NULL );
    mv_MultiVectorSetMask( blockVectorBX, NULL );
    mv_MultiVectorSetMask( blockVectorAP, activeMask );
    mv_MultiVectorSetMask( blockVectorBP, activeMask );
    mv_MultiVectorSetMask( blockVectorP, activeMask );
    mv_MultiVectorSetMask( blockVectorW, activeMask );

  }

  if ( exitFlag != 0 || *iterationNumber > maxIterations )
    exitFlag = REQUESTED_ACCURACY_NOT_ACHIEVED;

  (*iterationNumber)--;
	
  if ( verbosityLevel == 1 ) {
    hypre_printf("\n");
    for ( i = 1; i <= sizeX; i++ ) 
      hypre_printf("Eigenvalue lambda %22.14e\n",
	     utilities_FortranMatrixValue( lambda, i, 1) );
    for ( i = 1; i <= sizeX; i++ ) 
      hypre_printf("Residual %22.14e\n",
	     utilities_FortranMatrixValue( residualNorms, i, 1) );
    hypre_printf("\n%d iterations\n", *iterationNumber );
  }

  mv_MultiVectorDestroy( blockVectorR );
  mv_MultiVectorDestroy( blockVectorP );
  mv_MultiVectorDestroy( blockVectorAX );
  mv_MultiVectorDestroy( blockVectorAR );
  mv_MultiVectorDestroy( blockVectorAP );
  if ( !noBFlag ) {
    mv_MultiVectorDestroy( blockVectorBX );
    mv_MultiVectorDestroy( blockVectorBR );
    mv_MultiVectorDestroy( blockVectorBP );
    if ( !noYFlag )
      mv_MultiVectorDestroy( blockVectorBY );
  }
  mv_MultiVectorDestroy( blockVectorW );

  utilities_FortranMatrixDestroy( gramA );
  utilities_FortranMatrixDestroy( gramB );
  utilities_FortranMatrixDestroy( lambdaAB );
  utilities_FortranMatrixDestroy( lambdaX );
  
  utilities_FortranMatrixDestroy( gramXAX );
  utilities_FortranMatrixDestroy( gramRAX );
  utilities_FortranMatrixDestroy( gramPAX );
  utilities_FortranMatrixDestroy( gramRAR );
  utilities_FortranMatrixDestroy( gramPAR );
  utilities_FortranMatrixDestroy( gramPAP );
  
  utilities_FortranMatrixDestroy( gramXBX );
  utilities_FortranMatrixDestroy( gramRBX );
  utilities_FortranMatrixDestroy( gramPBX );
  utilities_FortranMatrixDestroy( gramRBR );
  utilities_FortranMatrixDestroy( gramPBR );
  utilities_FortranMatrixDestroy( gramPBP );

  utilities_FortranMatrixDestroy( gramYBY );
  utilities_FortranMatrixDestroy( gramYBX );
  utilities_FortranMatrixDestroy( tempYBX );
  utilities_FortranMatrixDestroy( gramYBR );
  utilities_FortranMatrixDestroy( tempYBR );

  utilities_FortranMatrixDestroy( coordX );
  utilities_FortranMatrixDestroy( coordXX );
  utilities_FortranMatrixDestroy( coordRX );
  utilities_FortranMatrixDestroy( coordPX );

  utilities_FortranMatrixDestroy( upperR );
  utilities_FortranMatrixDestroy( historyColumn );

  utilities_FortranMatrixDestroy( lambda );
  utilities_FortranMatrixDestroy( lambdaHistory );
  utilities_FortranMatrixDestroy( residualNorms );
  utilities_FortranMatrixDestroy( residualNormsHistory );	

  free( activeMask );

  return exitFlag;
}
