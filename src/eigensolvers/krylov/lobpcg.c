/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Locally optimal preconditioned conjugate gradient functions
 *
 * Evgueni Ovtchinnikov -- 21 Apr 2004 / 12 May 2004
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "lobpcg.h"

int
lobpcg_initialize( lobpcg_Data* data ) {

  (data->tolerance).absolute	= 1.0e-06;
  (data->tolerance).relative	= 0.0;
  (data->maxIterations)		= 500;
  (data->precondUsageMode)	= 0;
  (data->verbosityLevel)	= 0;
  (data->eigenvaluesHistory)	= utilities_FortranMatrixCreate();  
  (data->residualNorms)		= utilities_FortranMatrixCreate();    
  (data->residualNormsHistory)	= utilities_FortranMatrixCreate();

  return 0;
}

int
lobpcg_clean( lobpcg_Data* data ) {
  
  utilities_FortranMatrixDestroy( data->eigenvaluesHistory );  
  utilities_FortranMatrixDestroy( data->residualNorms );  
  utilities_FortranMatrixDestroy( data->residualNormsHistory );

  return 0;
}

int
lobpcg_solve( hypre_MultiVectorPtr blockVectorX, /* eigenvectors */
	      void* operatorAData, /* data for A */
	      void (*operatorA)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr ),
	      void* operatorBData, /* data for B */
	      void (*operatorB)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr ),
	      void* operatorTData, /* data for T */
	      void (*operatorT)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr ),
	      hypre_MultiVectorPtr blockVectorY, /* constraints */
	      lobpcg_Tolerance tolerance, /* tolerance */
	      int maxIterations, /* max # of iterations */
	      int verbosityLevel, /* 0: no print, 1: print initial
				     and final eigenvalues and residuals,
				     iteration number, number of 
				     non-convergent eigenpairs and
				     maximal residual on each iteration */
	      int* iterationNumber, /* pointer to the iteration number */
	      utilities_FortranMatrix* lambda, /* eigenvalues */
	      utilities_FortranMatrix* lambdaHistory, /* eigenvalues 
							 history */
	      utilities_FortranMatrix* residualNorms, /* residual norms */
	      utilities_FortranMatrix* residualNormsHistory /* residual
							       norms
							       history */
){

  int				sizeX; /* number of eigenvectors */
  int				sizeY; /* number of constraints */
  int				sizeR; /* number of residuals used */
  int				sizeP; /* number of conj. directions used */
  int				sizeA; /* size of the Gram matrix for A */
  int				sizeX3; /* 3*sizeX */

  int				firstR; /* first line of the Gram block
					   corresponding to residuals */
  int				lastR; /* last line of this block */
  int				firstP; /* same for conjugate directions */
  int				lastP;

  int				noTFlag; /* nonzero: no preconditioner */
  int				noBFlag; /* nonzero: no operator B */
  int				noYFlag; /* nonzero: no constaints */

  int				exitFlag; /* 1: problem size is too small,
					     2: block size < 1,
					     3: linearly dependent constraints,
					     -1: requested accuracy not 
					     achieved */

  int*				activeMask; /* soft locking mask */

  int				i; /* short loop counter */

#if 0
  long				n; /* dimension 1 of X */
  /* had to remove because n is not available in some interfaces */ 
#endif 

  hypre_MultiVectorPtr		blockVectorR; /* residuals */
  hypre_MultiVectorPtr		blockVectorP; /* conjugate directions */

  hypre_MultiVectorPtr		blockVectorW; /* auxiliary block vector */

  hypre_MultiVectorPtr		blockVectorAX; /* A*X */
  hypre_MultiVectorPtr		blockVectorAR; /* A*R */
  hypre_MultiVectorPtr		blockVectorAP; /* A*P */

  hypre_MultiVectorPtr		blockVectorBX; /* B*X */
  hypre_MultiVectorPtr		blockVectorBR; /* B*R */
  hypre_MultiVectorPtr		blockVectorBP; /* B*P */

  hypre_MultiVectorPtr		blockVectorBY; /* B*Y */

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

  /* initialization */

  exitFlag = 0;
  *iterationNumber = 0;
  noTFlag = operatorT == NULL;
  noBFlag = operatorB == NULL;

  sizeY = hypre_MultiVectorWidth( blockVectorY );
  noYFlag = sizeY == 0;

  sizeX = hypre_MultiVectorWidth( blockVectorX );

#if 0
  /* had to remove because n is not available in some interfaces */ 
  n = hypre_MultiVectorHeight( blockVectorX );

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

  blockVectorW = hypre_MultiVectorCreateCopy( blockVectorX, 0 );

  if ( !noYFlag ) {
    utilities_FortranMatrixAllocateData( sizeY, sizeY, gramYBY );
    utilities_FortranMatrixAllocateData( sizeY, sizeX, gramYBX );
    utilities_FortranMatrixAllocateData( sizeY, sizeX, tempYBX );
    if ( !noBFlag ) {
      operatorB( operatorBData, blockVectorY, blockVectorBY );
      blockVectorBY = hypre_MultiVectorCreateCopy( blockVectorY, 0 );
    }
    else
      blockVectorBY = blockVectorY;

    lobpcg_MultiVectorByMultiVector( blockVectorBY, blockVectorY, gramYBY );
    exitFlag = lobpcg_chol( gramYBY );
    if ( exitFlag != 0 ) {
      if ( verbosityLevel )
	printf("Cannot handle linear dependent constraints\n");
      utilities_FortranMatrixDestroy( gramYBY );
      utilities_FortranMatrixDestroy( gramYBX );
      utilities_FortranMatrixDestroy( tempYBX );
      utilities_FortranMatrixDestroy( gramYBR );
      utilities_FortranMatrixDestroy( tempYBR );
      if ( !noBFlag )
	hypre_MultiVectorDestroy( blockVectorBY );
      hypre_MultiVectorDestroy( blockVectorW );
      return WRONG_CONSTRAINTS;
    }      
    utilities_FortranMatrixUpperInv( gramYBY );
    utilities_FortranMatrixClearL( gramYBY );

    /* apply the constraints to the initial X */
    lobpcg_MultiVectorByMultiVector( blockVectorBY, blockVectorX, gramYBX );
    utilities_FortranMatrixMultiply( gramYBY, 1, gramYBX, 0, tempYBX );
    utilities_FortranMatrixMultiply( gramYBY, 0, tempYBX, 0, gramYBX );
    lobpcg_MultiVectorByMatrix( blockVectorY, gramYBX, blockVectorW );
    hypre_MultiVectorAxpy( -1.0, blockVectorW, blockVectorX );
  }

  if ( verbosityLevel ) {
    printf("\nSolving ");
    if ( noBFlag )
      printf("standard");
    else
      printf("generalized");
    printf(" eigenvalue problem with");
    if ( noTFlag )
      printf("out");
    printf(" preconditioning\n\n");
    printf("block size %d\n\n", sizeX );
    if ( noYFlag )
      printf("No constraints\n\n");
    else {
      if ( sizeY > 1 )
	printf("%d constraints\n\n", sizeY);
      else
	printf("%d constraint\n\n", sizeY);
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
  activeMask = (int*)calloc( sizeX, sizeof(int) );
  assert( activeMask != NULL );
  for ( i = 0; i < sizeX; i++ )
    activeMask[i] = 1;

  /* allocate memory for Gram matrices and the Ritz values */
  sizeX3 = 3*sizeX;
  utilities_FortranMatrixAllocateData( sizeX3, sizeX3, gramA );
  utilities_FortranMatrixAllocateData( sizeX3, sizeX3, gramB );
  utilities_FortranMatrixAllocateData( sizeX3, 1, lambdaAB );

  /* creating block vectors R, P, AX, AR, AP, BX, BR, BP and W */
  blockVectorR = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorP = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAX = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAR = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAP = hypre_MultiVectorCreateCopy( blockVectorX, 0 );

  if ( !noBFlag ) {
    blockVectorBX = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
    blockVectorBR = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
    blockVectorBP = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  }
  else {
    blockVectorBX = blockVectorX;
    blockVectorBR = blockVectorR;
    blockVectorBP = blockVectorP;
  }

  hypre_MultiVectorSetMask( blockVectorR, activeMask );
  hypre_MultiVectorSetMask( blockVectorP, activeMask );
  hypre_MultiVectorSetMask( blockVectorAR, activeMask );
  hypre_MultiVectorSetMask( blockVectorAP, activeMask );
  if ( !noBFlag ) {
    hypre_MultiVectorSetMask( blockVectorBR, activeMask );
    hypre_MultiVectorSetMask( blockVectorBP, activeMask );
  }
  hypre_MultiVectorSetMask( blockVectorW, activeMask );

  /* B-orthonormaliization of X */
  /* selecting a block in gramB for R factor upperR */
  utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, upperR );
  if ( !noBFlag ) {
    operatorB( operatorBData, blockVectorX, blockVectorBX );
  }
  exitFlag = lobpcg_MultiVectorImplicitQR( blockVectorX, blockVectorBX, 
					   upperR, blockVectorW );
  if ( exitFlag ) {
    lobpcg_errorMessage( verbosityLevel, "Bad initial vectors: orthonormalization failed\n" );
    if ( verbosityLevel )
      printf("DPOTRF INFO = %d\n", exitFlag);
  }
  else {

    if ( !noBFlag ) { /* update BX */
      lobpcg_MultiVectorByMatrix( blockVectorBX, upperR, blockVectorW );
      hypre_MultiVectorCopy( blockVectorW, blockVectorBX );
    }

    operatorA( operatorAData, blockVectorX, blockVectorAX );

    /* gramXAX = X'*AX */
    utilities_FortranMatrixSelectBlock( gramA, 1, sizeX, 1, sizeX, gramXAX );
    lobpcg_MultiVectorByMultiVector( blockVectorX, blockVectorAX, gramXAX );
    utilities_FortranMatrixSymmetrize( gramXAX );

    /* gramXBX = X'*X */ 
    utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, gramXBX );
    lobpcg_MultiVectorByMultiVector( blockVectorX, blockVectorBX, gramXBX );
    utilities_FortranMatrixSymmetrize( gramXBX );
    /*  utilities_FortranMatrixSetToIdentity( gramXBX );*/ /* X may be bad! */
    
    if ( (exitFlag = lobpcg_solveGEVP( gramXAX, gramXBX, lambda )) != 0 ) {
      lobpcg_errorMessage( verbosityLevel, 
			   "Bad problem: Rayleigh-Ritz in the initial subspace failed\n" );
      if ( verbosityLevel )
	printf("DSYGV INFO = %d\n", exitFlag);
    }
    else {
      utilities_FortranMatrixSelectBlock( gramXAX, 1, sizeX, 1, sizeX, coordX );

      lobpcg_MultiVectorByMatrix( blockVectorX, coordX, blockVectorW );
      hypre_MultiVectorCopy( blockVectorW, blockVectorX );

      lobpcg_MultiVectorByMatrix( blockVectorAX, coordX, blockVectorW );
      hypre_MultiVectorCopy( blockVectorW, blockVectorAX );

      if ( !noBFlag ) {
	lobpcg_MultiVectorByMatrix( blockVectorBX, coordX, blockVectorW );
	hypre_MultiVectorCopy( blockVectorW, blockVectorBX );
      }

      /*
      lobpcg_MultiVectorByMultiVector( blockVectorBX, blockVectorX, upperR );
      utilities_FortranMatrixPrint( upperR, "xbx.dat" );
      utilities_FortranMatrixPrint( lambda, "lmd.dat" );
      */

      hypre_MultiVectorByDiagonal( blockVectorBX, 
				   NULL, sizeX, 
				   utilities_FortranMatrixValues( lambda ),
				   blockVectorR );

      hypre_MultiVectorAxpy( -1.0, blockVectorAX, blockVectorR );

      hypre_MultiVectorByMultiVectorDiag( blockVectorR, blockVectorR, 
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
	printf("\n");
	for (i = 1; i <= sizeX; i++ ) 
	  printf("Initial eigenvalues lambda %22.16e\n",
		 utilities_FortranMatrixValue( lambda, i, 1) );
	for (i = 1; i <= sizeX; i++) 
	  printf("Initial residuals %12.6e\n",
		 utilities_FortranMatrixValue( residualNorms, i, 1) );
      }
      else if ( verbosityLevel == 1 )
	printf("\nInitial Max. Residual %22.16e\n",
	       utilities_FortranMatrixMaxValue( residualNorms ) );
    }
  }

  for ( *iterationNumber = 1; exitFlag == 0 && *iterationNumber <= maxIterations; 
	(*iterationNumber)++ ) {
    
    sizeR = lobpcg_checkResiduals( residualNorms, lambda, tolerance, 
				   activeMask );
    if ( sizeR < 1 )
      break;

    if ( !noTFlag ) {
      operatorT( operatorTData, blockVectorR, blockVectorW );
      hypre_MultiVectorCopy( blockVectorW, blockVectorR );
    }

    if ( !noYFlag ) { /* apply the constraints to R  */
      utilities_FortranMatrixSelectBlock( gramYBX, 1, sizeY, 1, sizeR, gramYBR );
      utilities_FortranMatrixSelectBlock( tempYBX, 1, sizeY, 1, sizeR, tempYBR );

      lobpcg_MultiVectorByMultiVector( blockVectorBY, blockVectorR, gramYBR );
      utilities_FortranMatrixMultiply( gramYBY, 1, gramYBR, 0, tempYBR );
      utilities_FortranMatrixMultiply( gramYBY, 0, tempYBR, 0, gramYBR );
      lobpcg_MultiVectorByMatrix( blockVectorY, gramYBR, blockVectorW );
      hypre_MultiVectorAxpy( -1.0, blockVectorW, blockVectorR );
    }

    firstR = sizeX + 1;
    lastR = sizeX + sizeR;
    firstP = lastR + 1;

    utilities_FortranMatrixSelectBlock( gramB, firstR, lastR, firstR, lastR, upperR );

    if ( !noBFlag ) {
      operatorB( operatorBData, blockVectorR, blockVectorBR );
    }
    exitFlag = lobpcg_MultiVectorImplicitQR( blockVectorR, blockVectorBR, 
					     upperR, blockVectorW );
    if ( exitFlag ) {
      lobpcg_errorMessage( verbosityLevel, "Orthonormalization of residuals failed\n" );
      if ( verbosityLevel )
	printf("DPOTRF INFO = %d\n", exitFlag);
      break;
    }

    if ( !noBFlag ) { /* update BR */
      lobpcg_MultiVectorByMatrix( blockVectorBR, upperR, blockVectorW );
      hypre_MultiVectorCopy( blockVectorW, blockVectorBR );
    }

    /* AR = A*R */
    operatorA( operatorAData, blockVectorR, blockVectorAR );

    if ( *iterationNumber > 1 ) {

      sizeP = sizeR;
      lastP = lastR + sizeP;

      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstP, lastP, upperR );

      exitFlag = lobpcg_MultiVectorImplicitQR( blockVectorP, blockVectorBP, 
					       upperR, blockVectorW );
      if ( exitFlag ) {
	/*
	lobpcg_errorMessage( verbosityLevel, "Orthonormalization of P failed\n" );
	if ( verbosityLevel )
	  printf("DPOTRF INFO = %d\n", exitFlag);
	*/
	sizeP = 0;
      }
      else {
			
	if ( !noBFlag ) { /* update BP */
	  lobpcg_MultiVectorByMatrix( blockVectorBP, upperR, blockVectorW );
	  hypre_MultiVectorCopy( blockVectorW, blockVectorBP );
	}

	/* update AP */
	lobpcg_MultiVectorByMatrix( blockVectorAP, upperR, blockVectorW );
	hypre_MultiVectorCopy( blockVectorW, blockVectorAP );
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

    if ( (exitFlag = lobpcg_solveGEVP( gramXAX, gramXBX, lambdaAB )) != 0 ) {
      lobpcg_errorMessage( verbosityLevel, "GEVP solver failure\n" );
      (*iterationNumber)--;
      /* if ( verbosityLevel )
	 printf("INFO = %d\n", exitFlag);*/
      break;
    }

    utilities_FortranMatrixSelectBlock( lambdaAB, 1, sizeX, 1, 1, lambdaX );
    utilities_FortranMatrixCopy( lambdaX, 0, lambda );

    utilities_FortranMatrixSelectBlock( gramA, 1, sizeA, 1, sizeX, coordX );

    utilities_FortranMatrixSelectBlock( coordX, 1, sizeX, 1, sizeX, coordXX );
    utilities_FortranMatrixSelectBlock( coordX, firstR, lastR, 1, sizeX, coordRX );

    if ( *iterationNumber > 1 ) {
	
      utilities_FortranMatrixSelectBlock( coordX, firstP, lastP, 1, sizeX, coordPX );

      hypre_MultiVectorSetMask( blockVectorW, NULL );
      lobpcg_MultiVectorByMatrix( blockVectorP, coordPX, blockVectorW );
      hypre_MultiVectorSetMask( blockVectorP, NULL );
      hypre_MultiVectorCopy( blockVectorW, blockVectorP );

      lobpcg_MultiVectorByMatrix( blockVectorAP, coordPX, blockVectorW );
      hypre_MultiVectorSetMask( blockVectorAP, NULL );
      hypre_MultiVectorCopy( blockVectorW, blockVectorAP );

      if ( !noBFlag ) {
	lobpcg_MultiVectorByMatrix( blockVectorBP, coordPX, blockVectorW );
	hypre_MultiVectorSetMask( blockVectorBP, NULL );
	hypre_MultiVectorCopy( blockVectorW, blockVectorBP );
      }

      lobpcg_MultiVectorByMatrix( blockVectorR, coordRX, blockVectorW );
      hypre_MultiVectorAxpy( 1.0, blockVectorW, blockVectorP );
			
      lobpcg_MultiVectorByMatrix( blockVectorAR, coordRX, blockVectorW );
      hypre_MultiVectorAxpy( 1.0, blockVectorW, blockVectorAP );

      if ( !noBFlag ) {
	lobpcg_MultiVectorByMatrix( blockVectorBR, coordRX, blockVectorW );
	hypre_MultiVectorAxpy( 1.0, blockVectorW, blockVectorBP );
      }

    }
    else {
      
      hypre_MultiVectorSetMask( blockVectorP, NULL );
      lobpcg_MultiVectorByMatrix( blockVectorR, coordRX, blockVectorP );
			
      hypre_MultiVectorSetMask( blockVectorAP, NULL );
      lobpcg_MultiVectorByMatrix( blockVectorAR, coordRX, blockVectorAP );

      if ( !noBFlag ) {
	hypre_MultiVectorSetMask( blockVectorBP, NULL );
	lobpcg_MultiVectorByMatrix( blockVectorBR, coordRX, blockVectorBP );
      }
		
    }
		
    hypre_MultiVectorCopy( blockVectorX, blockVectorW );
    lobpcg_MultiVectorByMatrix( blockVectorW, coordXX, blockVectorX );
    hypre_MultiVectorAxpy( 1.0, blockVectorP, blockVectorX );

    hypre_MultiVectorCopy( blockVectorAX, blockVectorW );
    lobpcg_MultiVectorByMatrix( blockVectorW, coordXX, blockVectorAX );
    hypre_MultiVectorAxpy( 1.0, blockVectorAP, blockVectorAX );

    if ( !noBFlag ) {
      hypre_MultiVectorCopy( blockVectorBX, blockVectorW );
      lobpcg_MultiVectorByMatrix( blockVectorW, coordXX, blockVectorBX );
      hypre_MultiVectorAxpy( 1.0, blockVectorBP, blockVectorBX );
    }

    hypre_MultiVectorSetMask( blockVectorAX, activeMask );
    hypre_MultiVectorSetMask( blockVectorBX, activeMask );

    hypre_MultiVectorByDiagonal( blockVectorBX, 
				 activeMask, sizeX, 
				 utilities_FortranMatrixValues( lambda ),
				 blockVectorR );

    hypre_MultiVectorAxpy( -1.0, blockVectorAX, blockVectorR );

    hypre_MultiVectorByMultiVectorDiag(	blockVectorR, blockVectorR, 
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
      printf( "Iteration %d \tbsize %d\n", *iterationNumber, sizeR );
      for ( i = 1; i <= sizeX; i++ ) 
	printf("Eigenvalue lambda %22.16e\n",
	       utilities_FortranMatrixValue( lambda, i, 1) );
      for ( i = 1; i <= sizeX; i++ ) 
	printf("Residual %12.6e\n",
	       utilities_FortranMatrixValue( residualNorms, i, 1) );
    }
    else if ( verbosityLevel == 1 )
      printf("Iteration %d \tbsize %d \tmaxres %22.16e\n",
	     *iterationNumber, sizeR, 
	     utilities_FortranMatrixMaxValue( residualNorms ) );

    hypre_MultiVectorSetMask( blockVectorAX, NULL );
    hypre_MultiVectorSetMask( blockVectorBX, NULL );
    hypre_MultiVectorSetMask( blockVectorAP, activeMask );
    hypre_MultiVectorSetMask( blockVectorBP, activeMask );
    hypre_MultiVectorSetMask( blockVectorP, activeMask );
    hypre_MultiVectorSetMask( blockVectorW, activeMask );

  }

  if ( exitFlag != 0 || *iterationNumber > maxIterations )
    exitFlag = REQUESTED_ACCURACY_NOT_ACHIEVED;

  (*iterationNumber)--;
	
  if ( verbosityLevel == 1 ) {
    printf("\n");
    for ( i = 1; i <= sizeX; i++ ) 
      printf("Eigenvalue lambda %22.16e\n",
	     utilities_FortranMatrixValue( lambda, i, 1) );
    for ( i = 1; i <= sizeX; i++ ) 
      printf("Residual %22.16e\n",
	     utilities_FortranMatrixValue( residualNorms, i, 1) );
    printf("\n%d iterations\n", *iterationNumber );
  }

  hypre_MultiVectorDestroy( blockVectorR );
  hypre_MultiVectorDestroy( blockVectorP );
  hypre_MultiVectorDestroy( blockVectorAX );
  hypre_MultiVectorDestroy( blockVectorAR );
  hypre_MultiVectorDestroy( blockVectorAP );
  if ( !noBFlag ) {
    hypre_MultiVectorDestroy( blockVectorBX );
    hypre_MultiVectorDestroy( blockVectorBR );
    hypre_MultiVectorDestroy( blockVectorBP );
    if ( !noYFlag )
      hypre_MultiVectorDestroy( blockVectorBY );
  }
  hypre_MultiVectorDestroy( blockVectorW );

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
	
  free( activeMask );

  return exitFlag;
}

void
lobpcg_MultiVectorByMultiVector(
hypre_MultiVectorPtr x,
hypre_MultiVectorPtr y,
utilities_FortranMatrix* xy
){
  hypre_MultiVectorByMultiVector( x, y,
				  utilities_FortranMatrixGlobalHeight( xy ),
				  utilities_FortranMatrixHeight( xy ),
				  utilities_FortranMatrixWidth( xy ),
				  utilities_FortranMatrixValues( xy ) );
}

void
lobpcg_MultiVectorByMatrix(
hypre_MultiVectorPtr x,
utilities_FortranMatrix* r,
hypre_MultiVectorPtr y
){
  hypre_MultiVectorByMatrix( x, 
			     utilities_FortranMatrixGlobalHeight( r ),
			     utilities_FortranMatrixHeight( r ),
			     utilities_FortranMatrixWidth( r ),
			     utilities_FortranMatrixValues( r ),
			     y );
}

int
lobpcg_MultiVectorImplicitQR( 
hypre_MultiVectorPtr x, hypre_MultiVectorPtr y,
utilities_FortranMatrix* r,
hypre_MultiVectorPtr z
){

  /* B-orthonormalizes x using y = B x */

  int ierr;

  lobpcg_MultiVectorByMultiVector( x, y, r );

  ierr = lobpcg_chol( r );

  if ( ierr != 0 )
    return ierr;

  utilities_FortranMatrixUpperInv( r );
  utilities_FortranMatrixClearL( r );

  hypre_MultiVectorCopy( x, z );
  lobpcg_MultiVectorByMatrix( z, r, x );

  return 0;
}

void
lobpcg_sqrtVector( int n, int* mask, double* v ) {

  int i;

  for ( i = 0; i < n; i++ )
    if ( mask == NULL || mask[i] )
      v[i] = sqrt(v[i]);
}

int
lobpcg_checkResiduals( 
utilities_FortranMatrix* resNorms,
utilities_FortranMatrix* lambda,
lobpcg_Tolerance tol,
int* activeMask
){
  int i, n;
  int notConverged;
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

void
lobpcg_errorMessage( int verbosityLevel, char* message )
{
  if ( verbosityLevel ) {
    fprintf( stderr, "Error in LOBPCG:\n" );
    fprintf( stderr, message );
  }
}


