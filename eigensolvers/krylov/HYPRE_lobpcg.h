/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "krylov.h"

#include "HYPRE_interpreter.h"

#include "lobpcg.h"

#ifndef HYPRE_LOBPCG_SOLVER
#define HYPRE_LOBPCG_SOLVER

typedef struct
{
  int    (*Precond)();
  int    (*PrecondSetup)();

} hypre_LOBPCGPrecond;

typedef struct
{

  lobpcg_Data	       	        lobpcgData;

  HYPRE_InterfaceInterpreter*   interpreter;

  void*			       	A;
  void*			       	matvecData;
  void*			       	precondData;

  void*			       	B;
  void*			       	matvecDataB;
  void*			       	T;
  void*			       	matvecDataT;

  hypre_LOBPCGPrecond	        precondFunctions;
   
} hypre_LOBPCGData;

#ifdef __cplusplus
extern "C" {
#endif


/* HYPRE_pcg.c */
void
HYPRE_LOBPCGCreate( HYPRE_InterfaceInterpreter*, HYPRE_Solver* );

int 
HYPRE_LOBPCGDestroy( HYPRE_Solver solver );

int 
HYPRE_LOBPCGSetup( HYPRE_Solver solver, 
		   HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );
int 
HYPRE_LOBPCGSetupB( HYPRE_Solver solver, 
		   HYPRE_Matrix A, HYPRE_Vector x );
int 
HYPRE_LOBPCGSetupT( HYPRE_Solver solver, 
		   HYPRE_Matrix A, HYPRE_Vector x );
int 
HYPRE_LOBPCGSolve( HYPRE_Solver, hypre_MultiVectorPtr, 
		   hypre_MultiVectorPtr, double* );

int 
HYPRE_LOBPCGSetTol( HYPRE_Solver solver, double tol );

int 
HYPRE_LOBPCGSetMaxIter( HYPRE_Solver solver, int max_iter );

int 
HYPRE_LOBPCGSetPrecondUsageMode( HYPRE_Solver solver, int mode );

int 
HYPRE_LOBPCGSetPrecond( HYPRE_Solver solver, 
			HYPRE_PtrToSolverFcn precond, 
			HYPRE_PtrToSolverFcn precond_setup, 
			HYPRE_Solver precond_solver );
int 
HYPRE_LOBPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );

int 
HYPRE_LOBPCGSetPrintLevel( HYPRE_Solver solver , int level );

utilities_FortranMatrix*
HYPRE_LOBPCGResidualNorms( HYPRE_Solver solver );

utilities_FortranMatrix*
HYPRE_LOBPCGResidualNormsHistory( HYPRE_Solver solver );

utilities_FortranMatrix*
HYPRE_LOBPCGEigenvaluesHistory( HYPRE_Solver solver );

int
HYPRE_LOBPCGIterations( HYPRE_Solver solver );

int 
hypre_LOBPCGDestroy( void *pcg_vdata );

int 
hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x );

int 
hypre_LOBPCGSetupB( void *pcg_vdata, void *A, void *x );

int 
hypre_LOBPCGSetupT( void *pcg_vdata, void *A, void *x );

int 
hypre_LOBPCGSetTol( void *pcg_vdata, double tol );

int 
hypre_LOBPCGSetMaxIter( void *pcg_vdata, int max_iter );

int 
hypre_LOBPCGGetPrecond( void *pcg_vdata, HYPRE_Solver *precond_data_ptr );

int 
hypre_LOBPCGSetPrecond( void *pcg_vdata, 
			int (*precond )(), 
			int (*precond_setup )(), 
			void *precond_data );

int 
hypre_LOBPCGSetPrintLevel( void *pcg_vdata, int level );

int 
hypre_LOBPCGSolve( void *pcg_vdata, hypre_MultiVectorPtr, 
		   hypre_MultiVectorPtr, double* );

int
lobpcg_solveGEVP( 
utilities_FortranMatrix* mtxA, 
utilities_FortranMatrix* mtxB,
utilities_FortranMatrix* eigVal
);

utilities_FortranMatrix*
hypre_LOBPCGResidualNorms( void *pcg_vdata );

utilities_FortranMatrix*
hypre_LOBPCGResidualNormsHistory( void *pcg_vdata );

utilities_FortranMatrix*
hypre_LOBPCGEigenvaluesHistory( void *pcg_vdata );

int
hypre_LOBPCGIterations( void* pcg_vdata );

#include "HYPRE_config.h"

#ifdef HYPRE_USING_ESSL

#include <essl.h>

#else

#include "fortran.h"

void hypre_F90_NAME_BLAS(dsygv, DSYGV)
( int *itype, char *jobz, char *uplo, int *n,
  double *a, int *lda, double *b, int *ldb, double *w, 
  double *work, int *lwork, /*@out@*/ int *info
);

#endif

#ifdef __cplusplus
}
#endif

#endif /* HYPRE_LOBPCG_SOLVER */

