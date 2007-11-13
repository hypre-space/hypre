/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_LOBPCG interface
 *
 *****************************************************************************/

#include "_hypre_utilities.h"

#include "HYPRE_config.h"
#ifdef HYPRE_USING_ESSL

#include <essl.h>

#else

#include "fortran.h"
int hypre_F90_NAME_BLAS(dsygv, DSYGV)
( int *itype, char *jobz, char *uplo, int *n,
  double *a, int *lda, double *b, int *ldb, double *w,
  double *work, int *lwork, /*@out@*/ int *info
);
int hypre_F90_NAME_BLAS( dpotrf, DPOTRF )
( char* uplo, int* n, double* aval, int* lda, int* ierr );

#endif

#include "HYPRE_lobpcg.h"
#include "lobpcg.h"

#include "interpreter.h"
#include "HYPRE_MatvecFunctions.h"

typedef struct
{
  int    (*Precond)();
  int    (*PrecondSetup)();

} hypre_LOBPCGPrecond;

typedef struct
{
  lobpcg_Tolerance              tolerance;
  int                           maxIterations;
  int                           verbosityLevel;
  int                           precondUsageMode;
  int                           iterationNumber;
  utilities_FortranMatrix*      eigenvaluesHistory;
  utilities_FortranMatrix*      residualNorms;
  utilities_FortranMatrix*      residualNormsHistory;

} lobpcg_Data;

#define lobpcg_tolerance(data)            ((data).tolerance)
#define lobpcg_absoluteTolerance(data)    ((data).tolerance.absolute)
#define lobpcg_relativeTolerance(data)    ((data).tolerance.relative)
#define lobpcg_maxIterations(data)        ((data).maxIterations)
#define lobpcg_verbosityLevel(data)       ((data).verbosityLevel)
#define lobpcg_precondUsageMode(data)     ((data).precondUsageMode)
#define lobpcg_iterationNumber(data)      ((data).iterationNumber)
#define lobpcg_eigenvaluesHistory(data)   ((data).eigenvaluesHistory)
#define lobpcg_residualNorms(data)        ((data).residualNorms)
#define lobpcg_residualNormsHistory(data) ((data).residualNormsHistory)

typedef struct
{

  lobpcg_Data                   lobpcgData;

  mv_InterfaceInterpreter*      interpreter;

  void*                         A;
  void*                         matvecData;
  void*                         precondData;

  void*                         B;
  void*                         matvecDataB;
  void*                         T;
  void*                         matvecDataT;

  hypre_LOBPCGPrecond           precondFunctions;

  HYPRE_MatvecFunctions*        matvecFunctions;

} hypre_LOBPCGData;

static int dsygv_interface (int *itype, char *jobz, char *uplo, int *
                    n, double *a, int *lda, double *b, int *ldb,
                    double *w, double *work, int *lwork, int *info)
{
#ifdef HYPRE_USING_ESSL
   dsygv(*itype, a, *lda, b, *ldb, w, a, *lda, *n, work, *lwork );
#else
   hypre_F90_NAME_BLAS( dsygv, DSYGV )( itype, jobz, uplo, n, 
		       a, lda, b, ldb,
		       w, work, lwork, info );
#endif
   return 0;
}

static int dpotrf_interface (char *uplo, int *n, double *a, int *
                    lda, int *info)
{
#ifdef HYPRE_USING_ESSL
   dpotrf(uplo, *n, a, *lda, info);
#else
   hypre_F90_NAME_BLAS( dpotrf, DPOTRF )(uplo, n, a, lda, info);
#endif
   return 0;
}


int
lobpcg_initialize( lobpcg_Data* data ) {

  (data->tolerance).absolute    = 1.0e-06;
  (data->tolerance).relative    = 0.0;
  (data->maxIterations)         = 500;
  (data->precondUsageMode)      = 0;
  (data->verbosityLevel)        = 0;
  (data->eigenvaluesHistory)    = utilities_FortranMatrixCreate();
  (data->residualNorms)         = utilities_FortranMatrixCreate();
  (data->residualNormsHistory)  = utilities_FortranMatrixCreate();

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
hypre_LOBPCGDestroy( void *pcg_vdata )
{
  hypre_LOBPCGData      *pcg_data      = pcg_vdata;
  HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
  int ierr = 0;

  if (pcg_data) {
    if ( pcg_data->matvecData != NULL ) {
      (*(mv->MatvecDestroy))(pcg_data->matvecData);
      pcg_data->matvecData = NULL;
    }
    if ( pcg_data->matvecDataB != NULL ) {
      (*(mv->MatvecDestroy))(pcg_data->matvecDataB);
      pcg_data->matvecDataB = NULL;
    }
    if ( pcg_data->matvecDataT != NULL ) {
      (*(mv->MatvecDestroy))(pcg_data->matvecDataT);
      pcg_data->matvecDataT = NULL;
    }
    
    lobpcg_clean( &(pcg_data->lobpcgData) );

    hypre_TFree( pcg_vdata );
  }

  return(ierr);
}

int
hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
  int  (*precond_setup)() = (pcg_data->precondFunctions).PrecondSetup;
  void *precond_data = (pcg_data->precondData);
  int ierr = 0;

  (pcg_data->A) = A;

  if ( pcg_data->matvecData != NULL )
    (*(mv->MatvecDestroy))(pcg_data->matvecData);
  (pcg_data->matvecData) = (*(mv->MatvecCreate))(A, x);

  if ( precond_setup != NULL ) {
    if ( pcg_data->T == NULL )
      precond_setup(precond_data, A, b, x);
    else
      precond_setup(precond_data, pcg_data->T, b, x);
  }

  return ierr;
}

int
hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
  int ierr = 0;

  (pcg_data->B) = B;

  if ( pcg_data->matvecDataB != NULL )
    (*(mv->MatvecDestroy))(pcg_data -> matvecDataB);
  (pcg_data->matvecDataB) = (*(mv->MatvecCreate))(B, x);
  if ( B != NULL )
    (pcg_data->matvecDataB) = (*(mv->MatvecCreate))(B, x);
  else
    (pcg_data->matvecDataB) = NULL;

  return ierr;
}

int
hypre_LOBPCGSetupT( void *pcg_vdata, void *T, void *x )
{
  hypre_LOBPCGData *pcg_data = pcg_vdata;
  HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
  int ierr = 0;

  (pcg_data -> T) = T;

  if ( pcg_data->matvecDataT != NULL )
    (*(mv->MatvecDestroy))(pcg_data->matvecDataT);
  if ( T != NULL )
    (pcg_data->matvecDataT) = (*(mv->MatvecCreate))(T, x);
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
  mv_InterfaceInterpreter* ii = data->interpreter;
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
  HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
  void*	              	      matvec_data = (pcg_data -> matvecData);

  (*(mv->Matvec))(matvec_data, 1.0, pcg_data->A, x, 0.0, y);
}

void
hypre_LOBPCGOperatorB( void *pcg_vdata, void* x, void* y ) {

  hypre_LOBPCGData*           pcg_data    = pcg_vdata;
  mv_InterfaceInterpreter* ii          = pcg_data->interpreter;
  HYPRE_MatvecFunctions * mv = pcg_data->matvecFunctions;
  void*                       matvec_data = (pcg_data -> matvecDataB);

  if ( pcg_data->B == NULL ) {
    (*(ii->CopyVector))(x, y);

    /* a test */
    /*
    (*(ii->ScaleVector))(2.0, y);
    */
 
    return;
  }

  (*(mv->Matvec))(matvec_data, 1.0, pcg_data->B, x, 0.0, y);
}

void
hypre_LOBPCGMultiPreconditioner( void *data, void * x, void*  y ) {

  hypre_LOBPCGData *pcg_data = data;
  mv_InterfaceInterpreter* ii = pcg_data->interpreter; 
  
  ii->Eval( hypre_LOBPCGPreconditioner, data, x, y );
}

void
hypre_LOBPCGMultiOperatorA( void *data, void * x, void*  y ) {

  hypre_LOBPCGData *pcg_data = data;
  mv_InterfaceInterpreter* ii = pcg_data->interpreter;
  
  ii->Eval( hypre_LOBPCGOperatorA, data, x, y );
}

void
hypre_LOBPCGMultiOperatorB( void *data, void * x, void*  y ) {

  hypre_LOBPCGData *pcg_data = data;
  mv_InterfaceInterpreter* ii = pcg_data->interpreter;
  
  ii->Eval( hypre_LOBPCGOperatorB, data, x, y );
}

int
hypre_LOBPCGSolve( void *vdata, 
		   mv_MultiVectorPtr con, 
		   mv_MultiVectorPtr vec, 
		   double* val ) {

  int ierr;
  hypre_LOBPCGData* data = vdata;
  int (*precond)() = (data->precondFunctions).Precond;
  void* opB = data->B;
  
  void (*prec)( void*, void*, void* );
  void (*operatorA)( void*, void*, void* );
  void (*operatorB)( void*, void*, void* );

  int maxit = lobpcg_maxIterations(data->lobpcgData);
  int verb  = lobpcg_verbosityLevel(data->lobpcgData);

  int n	= mv_MultiVectorWidth( vec );
  lobpcg_BLASLAPACKFunctions blap_fn;
   
  utilities_FortranMatrix* lambdaHistory;
  utilities_FortranMatrix* residuals;
  utilities_FortranMatrix* residualsHistory;
  
  lambdaHistory	= lobpcg_eigenvaluesHistory(data->lobpcgData);
  residuals = lobpcg_residualNorms(data->lobpcgData);
  residualsHistory = lobpcg_residualNormsHistory(data->lobpcgData);

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

  blap_fn.dsygv = dsygv_interface;
  blap_fn.dpotrf = dpotrf_interface;
  
  ierr = lobpcg_solve( vec, 
		       vdata, operatorA, 
		       vdata, operatorB,
		       vdata, prec,
		       con,
                       blap_fn,
		       lobpcg_tolerance(data->lobpcgData), maxit, verb,
		       &(lobpcg_iterationNumber(data->lobpcgData)),
                       val, 
                       utilities_FortranMatrixValues(lambdaHistory),
	               utilities_FortranMatrixGlobalHeight(lambdaHistory),
                       utilities_FortranMatrixValues(residuals),
                       utilities_FortranMatrixValues(residualsHistory),
                       utilities_FortranMatrixGlobalHeight(residualsHistory)
		       );
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


void
HYPRE_LOBPCGCreate( mv_InterfaceInterpreter* ii, HYPRE_MatvecFunctions* mv, 
                    HYPRE_Solver* solver )
{
  hypre_LOBPCGData *pcg_data;

  pcg_data = hypre_CTAlloc(hypre_LOBPCGData,1);

  (pcg_data->precondFunctions).Precond = NULL;
  (pcg_data->precondFunctions).PrecondSetup = NULL;

   /* set defaults */

  (pcg_data->interpreter)               = ii;
  pcg_data->matvecFunctions             = mv;

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
HYPRE_LOBPCGSolve( HYPRE_Solver solver, mv_MultiVectorPtr con, 
		   mv_MultiVectorPtr vec, double* val )
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

void
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

