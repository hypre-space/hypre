/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
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
	      void (*operatorA)( void*, void*, void* ), /* A */
	      void* operatorBData, /* data for B */
	      void (*operatorB)( void*, void*, void* ), /* B */
	      void* operatorTData, /* data for T */
	      void (*operatorT)( void*, void*, void* ), /* T */
	      hypre_MultiVectorPtr blockVectorY, /* constrains */
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
  int				exitFlag; /* 1: problem size is too small,
					     2: problem size < 1,
					     -1: requested accuracy not 
					     achieved */

  int*				activeMask; /* soft locking mask */

  int				i; /* short loop counter */
  long				n; /* dimension 1 of X */

  hypre_MultiVectorPtr		blockVectorR; /* residuals */
  hypre_MultiVectorPtr		blockVectorP; /* conjugate directions */

  hypre_MultiVectorPtr		blockVectorW; /* auxiliary block vector */

  hypre_MultiVectorPtr		blockVectorAX; /* A*X */
  hypre_MultiVectorPtr		blockVectorAR; /* A*R */
  hypre_MultiVectorPtr		blockVectorAP; /* A*P */

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

  sizeX = hypre_MultiVectorWidth( blockVectorX );
  n = hypre_MultiVectorHeight( blockVectorX );

  /* had to remove because n is not available in some interfaces */ /*
								      if ( n < 5*sizeX ) {
								      exitFlag = PROBLEM_SIZE_TOO_SMALL;
								      lobpcg_errorMessage( verbosityLevel,
								      "The problem size is too small compared to the block size for LOBPCG.\n" );
								      return exitFlag;
								      }
								    */
  if ( sizeX < 1 ) {
    exitFlag = WRONG_BLOCK_SIZE;
    lobpcg_errorMessage( verbosityLevel,
			 "The bloc size is wrong.\n" );
    return exitFlag;
  }

  /* creating empty fortran matrices */

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

  /* allocate memory for Gram matrices and the eigenvalues
     of gramA u = lambda gramB u */
  sizeX3 = 3*sizeX;
  utilities_FortranMatrixAllocateData( sizeX3, sizeX3, gramA );
  utilities_FortranMatrixAllocateData( sizeX3, sizeX3, gramB );
  utilities_FortranMatrixAllocateData( sizeX3, 1, lambdaAB );

  /* creating block vectors R, P, AX, AR, AP and W */
  blockVectorR = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorP = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAX = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAR = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorAP = hypre_MultiVectorCreateCopy( blockVectorX, 0 );
  blockVectorW = hypre_MultiVectorCreateCopy( blockVectorX, 0 );

  /* selecting a block in gramB for upperR */
  utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, upperR );
  utilities_FortranMatrixClear( upperR ); /* upperR = 0 */
  /* QR-factorization of X, upperR is R-factor  */
  hypre_MultiVectorExplicitQR( NULL, blockVectorX, sizeX3, sizeX, sizeX,
			       utilities_FortranMatrixValues( upperR ) );

  hypre_MultiVectorEval( operatorA, operatorAData, 
			 NULL, blockVectorX,
			 NULL, blockVectorAX ); /* AX = A*X */

  /* gramXAX = X'*AX */
  utilities_FortranMatrixSelectBlock( gramA, 1, sizeX, 1, sizeX, gramXAX );
  lobpcg_MultiVectorByMultiVector( NULL, blockVectorX, 
				   NULL, blockVectorAX, gramXAX );
  utilities_FortranMatrixSymmetrize( gramXAX );

  /* gramXBX = X'*X */ 
  utilities_FortranMatrixSelectBlock( gramB, 1, sizeX, 1, sizeX, gramXBX );
  lobpcg_MultiVectorByMultiVector( NULL, blockVectorX, 
				   NULL, blockVectorX, gramXBX );
  utilities_FortranMatrixSymmetrize( gramXBX );
  /*  utilities_FortranMatrixSetToIdentity( gramXBX );*/ /* X may be bad! */

  if ( (exitFlag = lobpcg_solveGEVP( gramXAX, gramXBX, lambda )) != 0 ) {
    lobpcg_errorMessage( verbosityLevel, "GEVP solver failure\n" );
    if ( verbosityLevel )
      printf("INFO = %d\n", exitFlag);
  }
  else {
    utilities_FortranMatrixSelectBlock( gramXAX, 1, sizeX, 1, sizeX, coordX );

    lobpcg_MultiVectorByMatrix( NULL, blockVectorX, coordX, NULL, blockVectorW );
    hypre_MultiVectorCopy( NULL, blockVectorW, NULL, blockVectorX );

    lobpcg_MultiVectorByMatrix( NULL, blockVectorAX, coordX, 
				NULL, blockVectorW );
    hypre_MultiVectorCopy( NULL, blockVectorW, NULL, blockVectorAX );

    hypre_MultiVectorByDiagonal( NULL, blockVectorX, 
				 NULL, sizeX, 
				 utilities_FortranMatrixValues( lambda ),
				 NULL, blockVectorR );

    hypre_MultiVectorAxpy( -1.0, NULL, blockVectorAX, NULL, blockVectorR );

    hypre_MultiVectorByMultiVectorDiag( NULL, blockVectorR, NULL, blockVectorR, 
					NULL, sizeX, 
					utilities_FortranMatrixValues( residualNorms ) ); 
    lobpcg_sqrtVector( sizeX, activeMask, 
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

  for ( *iterationNumber = 1; exitFlag == 0 && *iterationNumber <= maxIterations; 
	(*iterationNumber)++ ) {
    
    sizeR = lobpcg_checkResiduals( residualNorms, lambda, tolerance, 
				   activeMask );
    if ( sizeR < 1 )
      break;

    if ( !noTFlag ) {
      hypre_MultiVectorEval( operatorT, operatorTData, 
			     activeMask, blockVectorR,
			     activeMask, blockVectorW );
      hypre_MultiVectorCopy( activeMask, blockVectorW, 
			     activeMask, blockVectorR );
    }

    firstR = sizeX + 1;
    lastR = sizeX + sizeR;
    firstP = lastR + 1;

    utilities_FortranMatrixSelectBlock( gramB, firstR, lastR, firstR, lastR, 
					upperR );

    utilities_FortranMatrixClear( upperR );
    hypre_MultiVectorExplicitQR( activeMask, blockVectorR, 
				 sizeX3, sizeR, sizeR,
				 utilities_FortranMatrixValues( upperR ) );
    hypre_MultiVectorEval( operatorA, operatorAData, 
			   activeMask, blockVectorR,
			   activeMask, blockVectorAR );

    if ( *iterationNumber > 1 ) {

      sizeP = sizeR;
      lastP = lastR + sizeP;

      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstP, lastP, 
					  upperR );
      utilities_FortranMatrixClear( upperR );
      hypre_MultiVectorExplicitQR( activeMask, blockVectorP, 
				   sizeX3, sizeP, sizeP,
				   utilities_FortranMatrixValues( upperR ) );
      utilities_FortranMatrixUpperInv( upperR );
			
      lobpcg_MultiVectorByMatrix( activeMask, blockVectorAP, upperR, 
				  activeMask, blockVectorW );
      hypre_MultiVectorCopy( activeMask, blockVectorW, 
			     activeMask, blockVectorAP );
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
		
    lobpcg_MultiVectorByMultiVector( 
				    activeMask, blockVectorR,
				    NULL, blockVectorAX, 
				    gramRAX );
		
    lobpcg_MultiVectorByMultiVector( activeMask, blockVectorR, 
				     activeMask, blockVectorAR, 
				     gramRAR );
    utilities_FortranMatrixSymmetrize( gramRAR );

    utilities_FortranMatrixSetToIdentity( gramXBX );

    lobpcg_MultiVectorByMultiVector( 
				    activeMask, blockVectorR, 
				    NULL, blockVectorX, 
				    gramRBX );
		
    utilities_FortranMatrixSetToIdentity( gramRBR );

    if ( *iterationNumber > 1 ) {
      
      utilities_FortranMatrixSelectBlock( gramA, firstP, lastP, 1, sizeX, 
					  gramPAX );
      utilities_FortranMatrixSelectBlock( gramA, firstP, lastP, firstR, lastR, 
					  gramPAR );
      utilities_FortranMatrixSelectBlock( gramA, firstP, lastP, firstP, lastP, 
					  gramPAP );
			
      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, 1, sizeX, 
					  gramPBX );
      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstR, lastR, 
					  gramPBR );
      utilities_FortranMatrixSelectBlock( gramB, firstP, lastP, firstP, lastP, 
					  gramPBP );

      lobpcg_MultiVectorByMultiVector( 
				      activeMask, blockVectorP, 
				      NULL, blockVectorAX, 
				      gramPAX );
			
      lobpcg_MultiVectorByMultiVector( 
				      activeMask, blockVectorP, 
				      activeMask, blockVectorAR, 
				      gramPAR );
			
      lobpcg_MultiVectorByMultiVector( activeMask, blockVectorP, 
				       activeMask, blockVectorAP, 
				       gramPAP );
      utilities_FortranMatrixSymmetrize( gramPAP );

      lobpcg_MultiVectorByMultiVector( 
				      activeMask, blockVectorP, 
				      NULL, blockVectorX, 
				      gramPBX );
			
      lobpcg_MultiVectorByMultiVector( 
				      activeMask, blockVectorP, 
				      activeMask, blockVectorR, 
				      gramPBR );
		
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
    utilities_FortranMatrixSelectBlock( coordX, firstR, lastR, 1, sizeX, 
					coordRX );

    if ( *iterationNumber > 1 ) {
	
      utilities_FortranMatrixSelectBlock( coordX, firstP, lastP, 1, sizeX, 
					  coordPX );

      lobpcg_MultiVectorByMatrix( activeMask, blockVectorP, coordPX, 
				  NULL, blockVectorW );
      hypre_MultiVectorCopy( NULL, blockVectorW, NULL, blockVectorP );

      lobpcg_MultiVectorByMatrix( activeMask, blockVectorAP, coordPX, 
				  NULL, blockVectorW );
      hypre_MultiVectorCopy( NULL, blockVectorW, NULL, blockVectorAP );

      lobpcg_MultiVectorByMatrix( activeMask, blockVectorR, coordRX, 
				  NULL, blockVectorW );
      hypre_MultiVectorAxpy( 1.0, NULL, blockVectorW, NULL, blockVectorP );
			
      lobpcg_MultiVectorByMatrix( activeMask, blockVectorAR, coordRX, 
				  NULL, blockVectorW );
      hypre_MultiVectorAxpy( 1.0, NULL, blockVectorW, NULL, blockVectorAP );
		
    }
    else {
      
      lobpcg_MultiVectorByMatrix( activeMask, blockVectorR, coordRX, 
				  NULL, blockVectorP );
			
      lobpcg_MultiVectorByMatrix( activeMask, blockVectorAR, coordRX, 
				  NULL, blockVectorAP );
		
    }
		
    hypre_MultiVectorCopy( NULL, blockVectorX, NULL, blockVectorW );
    lobpcg_MultiVectorByMatrix( NULL, blockVectorW, coordXX, 
				NULL, blockVectorX );
    hypre_MultiVectorAxpy( 1.0, NULL, blockVectorP, NULL, blockVectorX );

    hypre_MultiVectorCopy( NULL, blockVectorAX, NULL, blockVectorW );
    lobpcg_MultiVectorByMatrix( NULL, blockVectorW, coordXX, 
				NULL, blockVectorAX );
    hypre_MultiVectorAxpy( 1.0, NULL, blockVectorAP, NULL, blockVectorAX );

    hypre_MultiVectorByDiagonal( activeMask, blockVectorX, 
				 activeMask, sizeX, 
				 utilities_FortranMatrixValues( lambda ),
				 activeMask, blockVectorR );

    hypre_MultiVectorAxpy( -1.0, activeMask, blockVectorAX, 
			   activeMask, blockVectorR );

    hypre_MultiVectorByMultiVectorDiag(	activeMask, blockVectorR, 
					activeMask, blockVectorR, 
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
int* xMask, hypre_MultiVectorPtr x,
int* yMask, hypre_MultiVectorPtr y,
utilities_FortranMatrix* xy
){
  hypre_MultiVectorByMultiVector( xMask, x, yMask, y,
				  utilities_FortranMatrixGlobalHeight( xy ),
				  utilities_FortranMatrixHeight( xy ),
				  utilities_FortranMatrixWidth( xy ),
				  utilities_FortranMatrixValues( xy ) );
}

void
lobpcg_MultiVectorByMatrix(
int* xMask, hypre_MultiVectorPtr x,
utilities_FortranMatrix* r,
int* yMask, hypre_MultiVectorPtr y
){
  hypre_MultiVectorByMatrix( xMask, x, 
			     utilities_FortranMatrixGlobalHeight( r ),
			     utilities_FortranMatrixHeight( r ),
			     utilities_FortranMatrixWidth( r ),
			     utilities_FortranMatrixValues( r ),
			     yMask, y );
}

void
lobpcg_sqrtVector( int n, int* mask, double* v ) {

  int i;

  for ( i = 0; i < n; i++ )
    if ( mask[i] )
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
  if ( verbosityLevel )
    fprintf( stderr, message );
}


