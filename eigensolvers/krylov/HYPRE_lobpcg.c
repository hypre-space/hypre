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
 * HYPRE_LOBPCG interface
 *
 *****************************************************************************/

#include "HYPRE_lobpcg.h"

void
HYPRE_LOBPCGCreate( HYPRE_InterfaceInterpreter* ii, HYPRE_Solver* solver )
{
  hypre_LOBPCGData *pcg_data;

  pcg_data = (hypre_LOBPCGData*)(ii->CAlloc)( 1, sizeof(hypre_LOBPCGData) );

  (pcg_data->precondFunctions).Precond = NULL;
  (pcg_data->precondFunctions).PrecondSetup = NULL;

   /* set defaults */

  (pcg_data->interpreter)               = ii;

  (pcg_data->matvecData)	       	= NULL;
  (pcg_data->B)	       			= NULL;
  (pcg_data->matvecDataB)	       	= NULL;
  (pcg_data->T)	       			= NULL;
  (pcg_data->matvecDataT)	       	= NULL;
  (pcg_data->precondData)	       	= NULL;

  lobpcg_initialize( &(pcg_data->lobpcgData) );

  *solver = (HYPRE_Solver)pcg_data;
}

int 
HYPRE_LOBPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_LOBPCGDestroy( (void *) solver ) );
}

int 
HYPRE_LOBPCGSetup( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_LOBPCGSetup( solver, A, b, x ) );
}

int 
HYPRE_LOBPCGSetupB( HYPRE_Solver solver,
                HYPRE_Matrix B,
                HYPRE_Vector x      )
{
   return( hypre_LOBPCGSetupB( solver, B, x ) );
}

int 
HYPRE_LOBPCGSetupT( HYPRE_Solver solver,
                HYPRE_Matrix T,
                HYPRE_Vector x      )
{
   return( hypre_LOBPCGSetupT( solver, T, x ) );
}

int 
HYPRE_LOBPCGSolve( HYPRE_Solver solver, hypre_MultiVectorPtr con, 
		   hypre_MultiVectorPtr vec, double* val )
{
   return( hypre_LOBPCGSolve( (void *) solver, con, vec, val ) );
}

int
HYPRE_LOBPCGSetTol( HYPRE_Solver solver, double tol )
{
   return( hypre_LOBPCGSetTol( (void *) solver, tol ) );
}

int
HYPRE_LOBPCGSetMaxIter( HYPRE_Solver solver, int max_iter )
{
   return( hypre_LOBPCGSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_LOBPCGSetPrecondUsageMode( HYPRE_Solver solver, int mode )
{
   return( hypre_LOBPCGSetPrecondUsageMode( (void *) solver, mode ) );
}

int
HYPRE_LOBPCGSetPrecond( HYPRE_Solver         solver,
                     HYPRE_PtrToSolverFcn precond,
                     HYPRE_PtrToSolverFcn precond_setup,
                     HYPRE_Solver         precond_solver )
{
   return( hypre_LOBPCGSetPrecond( (void *) solver,
                                precond, precond_setup,
                                (void *) precond_solver ) );
}

int
HYPRE_LOBPCGGetPrecond( HYPRE_Solver  solver,
                     HYPRE_Solver *precond_data_ptr )
{
   return( hypre_LOBPCGGetPrecond( (void *)     solver,
                                (HYPRE_Solver *) precond_data_ptr ) );
}

int
HYPRE_LOBPCGSetPrintLevel( HYPRE_Solver solver, int level )
{
   return( hypre_LOBPCGSetPrintLevel( (void*)solver, level ) );
}

utilities_FortranMatrix*
HYPRE_LOBPCGResidualNorms( HYPRE_Solver solver )
{
  return ( hypre_LOBPCGResidualNorms( (void*)solver ) );
}

utilities_FortranMatrix*
HYPRE_LOBPCGResidualNormsHistory( HYPRE_Solver solver )
{
  return ( hypre_LOBPCGResidualNormsHistory( (void*)solver ) );
}

utilities_FortranMatrix*
HYPRE_LOBPCGEigenvaluesHistory( HYPRE_Solver solver )
{
  return ( hypre_LOBPCGEigenvaluesHistory( (void*)solver ) );
}

int
HYPRE_LOBPCGIterations( HYPRE_Solver solver )
{
  return ( hypre_LOBPCGIterations( (void*)solver ) );
}

int
hypre_LOBPCGDestroy( void *pcg_vdata )
{
  hypre_LOBPCGData      *pcg_data      = pcg_vdata;
  HYPRE_InterfaceInterpreter* ii       = pcg_data->interpreter;
  int ierr = 0;

  if (pcg_data) {
    if ( pcg_data->matvecData != NULL ) {
      (*(ii->MatvecDestroy))(pcg_data->matvecData);
      pcg_data->matvecData = NULL;
    }
    if ( pcg_data->matvecDataB != NULL ) {
      (*(ii->MatvecDestroy))(pcg_data->matvecDataB);
      pcg_data->matvecDataB = NULL;
    }
    if ( pcg_data->matvecDataT != NULL ) {
      (*(ii->MatvecDestroy))(pcg_data->matvecDataT);
      pcg_data->matvecDataT = NULL;
    }
    
    lobpcg_clean( &(pcg_data->lobpcgData) );

    (ii->Free)( pcg_vdata );
  }

  return(ierr);
}

int
hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  HYPRE_InterfaceInterpreter* ii = pcg_data->interpreter;
  int  (*precond_setup)() = (pcg_data->precondFunctions).PrecondSetup;
  void *precond_data = (pcg_data->precondData);
  int ierr = 0;

  (pcg_data->A) = A;

  if ( pcg_data->matvecData != NULL )
    (*(ii->MatvecDestroy))(pcg_data->matvecData);
  (pcg_data->matvecData) = (*(ii->MatvecCreate))(A, x);

  if ( precond_setup != NULL )
    if ( pcg_data->T == NULL )
      precond_setup(precond_data, A, b, x);
    else
      precond_setup(precond_data, pcg_data->T, b, x);

  return ierr;
}

int
hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  HYPRE_InterfaceInterpreter* ii = pcg_data->interpreter;
  int (*precond_setup)() = (pcg_data->precondFunctions).PrecondSetup;
  void *precond_data = (pcg_data->precondData);
  int ierr = 0;

  (pcg_data->B) = B;

  if ( pcg_data->matvecDataB != NULL )
    (*(ii->MatvecDestroy))(pcg_data -> matvecDataB);
  (pcg_data->matvecDataB) = (*(ii->MatvecCreate))(B, x);
  if ( B != NULL )
    (pcg_data->matvecDataB) = (*(ii->MatvecCreate))(B, x);
  else
    (pcg_data->matvecDataB) = NULL;

  return ierr;
}

int
hypre_LOBPCGSetupT( void *pcg_vdata, void *T, void *x )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  HYPRE_InterfaceInterpreter* ii = pcg_data->interpreter;
  int (*precond_setup)() = (pcg_data->precondFunctions).PrecondSetup;
  void *precond_data = (pcg_data->precondData);
  int ierr = 0;

  (pcg_data -> T) = T;

  if ( pcg_data->matvecDataT != NULL )
    (*(ii->MatvecDestroy))(pcg_data->matvecDataT);
  if ( T != NULL )
    (pcg_data->matvecDataT) = (*(ii->MatvecCreate))(T, x);
  else
    (pcg_data->matvecDataT) = NULL;

  return ierr;
}

int
hypre_LOBPCGSetTol( void* pcg_vdata, double tol )
{
  hypre_LOBPCGData *pcg_data	= pcg_vdata;
  int	       	   ierr		= 0;

  lobpcg_absoluteTolerance(pcg_data->lobpcgData) = tol;
 
  return ierr;
}

int
hypre_LOBPCGSetMaxIter( void* pcg_vdata, int max_iter  )
{
  hypre_LOBPCGData *pcg_data	= pcg_vdata;
  int		   ierr		= 0;
 
  lobpcg_maxIterations(pcg_data->lobpcgData) = max_iter;
 
  return ierr;
}

int
hypre_LOBPCGSetPrecondUsageMode( void* pcg_vdata, int mode  )
{
  hypre_LOBPCGData *pcg_data	= pcg_vdata;
  int	      	   ierr		= 0;
 
  lobpcg_precondUsageMode(pcg_data->lobpcgData) = mode;
 
  return ierr;
}

int
hypre_LOBPCGGetPrecond( void         *pcg_vdata,
			HYPRE_Solver *precond_data_ptr )
{
  hypre_LOBPCGData*	pcg_data	= pcg_vdata;
  int	      		ierr		= 0;

  *precond_data_ptr = (HYPRE_Solver)(pcg_data -> precondData);

  return ierr;
}

int
hypre_LOBPCGSetPrecond( void  *pcg_vdata,
			int  (*precond)(),
			int  (*precond_setup)(),
			void  *precond_data )
{
  hypre_LOBPCGData* pcg_data = pcg_vdata;
  int ierr = 0;
 
  (pcg_data->precondFunctions).Precond      = precond;
  (pcg_data->precondFunctions).PrecondSetup = precond_setup;
  (pcg_data->precondData)                   = precond_data;
 
  return ierr;
}

int
hypre_LOBPCGSetPrintLevel( void *pcg_vdata, int level )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  int		   ierr	     = 0;
 
  lobpcg_verbosityLevel(pcg_data->lobpcgData) = level;
 
  return ierr;
}

void
hypre_LOBPCGPreconditioner( void *vdata, void* x, void* y ) {

  hypre_LOBPCGData *data = vdata;
  HYPRE_InterfaceInterpreter* ii = data->interpreter;
  int (*precond)() = (data->precondFunctions).Precond;

  if ( precond == NULL ) {
    (*(ii->CopyVector))(x,y);
	return;
  }

  if ( lobpcg_precondUsageMode(data->lobpcgData) == 0 )
    (*(ii->ClearVector))(y);
  else
    (*(ii->CopyVector))(x,y);
  
  if ( data->T == NULL )
    precond(data->precondData, data->A, x, y);
  else
    precond(data->precondData, data->T, x, y);
}

void
hypre_LOBPCGOperatorA( void *pcg_vdata, void* x, void* y ) {

  hypre_LOBPCGData*           pcg_data    = pcg_vdata;
  HYPRE_InterfaceInterpreter* ii          = pcg_data->interpreter;
  void*	              	      matvec_data = (pcg_data -> matvecData);

  (*(ii->Matvec))(matvec_data, 1.0, pcg_data->A, x, 0.0, y);
}

void
hypre_LOBPCGOperatorB( void *pcg_vdata, void* x, void* y ) {

  hypre_LOBPCGData*           pcg_data    = pcg_vdata;
  HYPRE_InterfaceInterpreter* ii          = pcg_data->interpreter;
  void*                       matvec_data = (pcg_data -> matvecDataB);

  if ( pcg_data->B == NULL ) {
    (*(ii->CopyVector))(x, y);

    /* a test */
    /*
    (*(ii->ScaleVector))(2.0, y);
    */
 
    return;
  }

  (*(ii->Matvec))(matvec_data, 1.0, pcg_data->B, x, 0.0, y);
}

void
hypre_LOBPCGMultiPreconditioner( void *data, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y ) {

  hypre_MultiVectorEval( hypre_LOBPCGPreconditioner, data, x, y );
}

void
hypre_LOBPCGMultiOperatorA( void *data, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y ) {

  hypre_MultiVectorEval( hypre_LOBPCGOperatorA, data, x, y );
}

void
hypre_LOBPCGMultiOperatorB( void *data, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y ) {

  hypre_MultiVectorEval( hypre_LOBPCGOperatorB, data, x, y );
}

int
hypre_LOBPCGSolve( void *vdata, 
		   hypre_MultiVectorPtr con, 
		   hypre_MultiVectorPtr vec, 
		   double* val ) {

  int ierr;
  hypre_LOBPCGData* data = vdata;
  int (*precond)() = (data->precondFunctions).Precond;
  void* opB = data->B;
  
  void (*prec)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr );
  void (*operatorA)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr );
  void (*operatorB)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr );

  int maxit = lobpcg_maxIterations(data->lobpcgData);
  int verb  = lobpcg_verbosityLevel(data->lobpcgData);

  int n	= hypre_MultiVectorWidth( vec );
   
  utilities_FortranMatrix* lambda;
  utilities_FortranMatrix* lambdaHistory;
  utilities_FortranMatrix* residuals;
  utilities_FortranMatrix* residualsHistory;
  
  lambda = utilities_FortranMatrixCreate();

  lambdaHistory	= lobpcg_eigenvaluesHistory(data->lobpcgData);
  residuals = lobpcg_residualNorms(data->lobpcgData);
  residualsHistory = lobpcg_residualNormsHistory(data->lobpcgData);

  utilities_FortranMatrixWrap( val, n, n, 1, lambda );

  utilities_FortranMatrixAllocateData( n, maxit + 1,	lambdaHistory );
  utilities_FortranMatrixAllocateData( n, 1,		residuals );
  utilities_FortranMatrixAllocateData( n, maxit + 1,	residualsHistory );

  if ( precond != NULL )
    prec = hypre_LOBPCGMultiPreconditioner;
  else
    prec = NULL;

  operatorA = hypre_LOBPCGMultiOperatorA;

  if ( opB != NULL )
    operatorB = hypre_LOBPCGMultiOperatorB;
  else
    operatorB = NULL;

  ierr = lobpcg_solve( vec, 
		       vdata, operatorA, 
		       vdata, operatorB,
		       vdata, prec,
		       con,
		       lobpcg_tolerance(data->lobpcgData), maxit, verb,
		       &(lobpcg_iterationNumber(data->lobpcgData)), 
		       lambda, lambdaHistory, 
		       residuals, residualsHistory );

  utilities_FortranMatrixDestroy(lambda);

  return ierr;
}

utilities_FortranMatrix*
hypre_LOBPCGResidualNorms( void *vdata )
{
  hypre_LOBPCGData *data = vdata; 
  return (lobpcg_residualNorms(data->lobpcgData));
}

utilities_FortranMatrix*
hypre_LOBPCGResidualNormsHistory( void *vdata )
{
  hypre_LOBPCGData *data = vdata; 
  return (lobpcg_residualNormsHistory(data->lobpcgData));
}

utilities_FortranMatrix*
hypre_LOBPCGEigenvaluesHistory( void *vdata )
{
  hypre_LOBPCGData *data = vdata; 
  return (lobpcg_eigenvaluesHistory(data->lobpcgData));
}

int
hypre_LOBPCGIterations( void* vdata )
{
  hypre_LOBPCGData *data = vdata;
  return (lobpcg_iterationNumber(data->lobpcgData));
}

int
lobpcg_solveGEVP( 
utilities_FortranMatrix* mtxA, 
utilities_FortranMatrix* mtxB,
utilities_FortranMatrix* eigVal
){

  int n, lda, ldb, itype, lwork, info;
  char jobz, uplo;
  double* work;
  double* a;
  double* b;
  double* lmd;

  a = utilities_FortranMatrixValues( mtxA );
  b = utilities_FortranMatrixValues( mtxB );
  lmd = utilities_FortranMatrixValues( eigVal );

  n = utilities_FortranMatrixHeight( mtxA );
  lda = utilities_FortranMatrixGlobalHeight( mtxA );
  ldb = utilities_FortranMatrixGlobalHeight( mtxB );
  lwork = 10*n;

  work = (double*)calloc( lwork, sizeof(double) );

#ifdef HYPRE_USING_ESSL

  info = 0;
  dsygv( 1, a, lda, b, ldb, lmd, a, lda, n, work, lwork );

#else

  itype = 1;
  jobz = 'V';
  uplo = 'L';
    
  hypre_F90_NAME_BLAS( dsygv, DSYGV )( &itype, &jobz, &uplo, &n, 
				       a, &lda, b, &ldb,
				       lmd, &work[0], &lwork, &info );

#endif

  free( work );
  return info;

}

int
lobpcg_chol( utilities_FortranMatrix* a ) {

  int lda, n;
  double* aval;
  char uplo;
  int ierr;

  lda = utilities_FortranMatrixGlobalHeight( a );
  n = utilities_FortranMatrixHeight( a );
  aval = utilities_FortranMatrixValues( a );
  uplo = 'U';

#ifdef HYPRE_USING_ESSL

  dpotrf( &uplo, n, aval, lda, &ierr );

#else

  hypre_F90_NAME_BLAS( dpotrf, DPOTRF )( &uplo, &n, aval, &lda, &ierr );

#endif

  return ierr;
}
