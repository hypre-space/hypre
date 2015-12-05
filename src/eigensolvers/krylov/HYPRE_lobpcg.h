/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 1.3 $
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


  /* HYPRE_lobpcg.c */

  /* LOBPCG Constructor */
void
HYPRE_LOBPCGCreate( HYPRE_InterfaceInterpreter*, HYPRE_Solver* );

  /* LOBPCG Destructor */
int 
HYPRE_LOBPCGDestroy( HYPRE_Solver solver );

  /* Sets the preconditioner; if not called, preconditioning is not used */
int 
HYPRE_LOBPCGSetPrecond( HYPRE_Solver solver, 
			HYPRE_PtrToSolverFcn precond, 
			HYPRE_PtrToSolverFcn precond_setup, 
			HYPRE_Solver precond_solver );
int 
HYPRE_LOBPCGGetPrecond( HYPRE_Solver solver , HYPRE_Solver *precond_data_ptr );

  /* Sets up A and the preconditioner, if there is one (see above) */
int 
HYPRE_LOBPCGSetup( HYPRE_Solver solver, 
		   HYPRE_Matrix A, HYPRE_Vector b, HYPRE_Vector x );

  /* Sets up B; if not called, B = I */
int 
HYPRE_LOBPCGSetupB( HYPRE_Solver solver, 
		   HYPRE_Matrix B, HYPRE_Vector x );

  /* If called, makes the preconditionig to be applyed to Tx = b, not Ax = b */
int 
HYPRE_LOBPCGSetupT( HYPRE_Solver solver, 
		   HYPRE_Matrix T, HYPRE_Vector x );

  /* Solves A x = lambda B x, y'x = 0 */
int 
HYPRE_LOBPCGSolve( HYPRE_Solver data, hypre_MultiVectorPtr y, 
		   hypre_MultiVectorPtr x, double* lambda );

  /* Sets the absolute tolerance */
int 
HYPRE_LOBPCGSetTol( HYPRE_Solver solver, double tol );

  /* Sets the maximal number of iterations */
int 
HYPRE_LOBPCGSetMaxIter( HYPRE_Solver solver, int maxIter );

  /* Defines which initial guess for inner PCG iterations to use:
     mode = 0: use zero initial guess, otherwise use RHS */
int 
HYPRE_LOBPCGSetPrecondUsageMode( HYPRE_Solver solver, int mode );

  /* Sets the level of printout */
int 
HYPRE_LOBPCGSetPrintLevel( HYPRE_Solver solver , int level );

  /* Returns the pointer to residual norms matrix (blockSize x 1)*/
utilities_FortranMatrix*
HYPRE_LOBPCGResidualNorms( HYPRE_Solver solver );

  /* Returns the pointer to residual norms history matrix (blockSize x maxIter)*/
utilities_FortranMatrix*
HYPRE_LOBPCGResidualNormsHistory( HYPRE_Solver solver );

  /* Returns the pointer to eigenvalue history matrix (blockSize x maxIter)*/
utilities_FortranMatrix*
HYPRE_LOBPCGEigenvaluesHistory( HYPRE_Solver solver );

  /* Returns the number of iterations performed by LOBPCG */
int
HYPRE_LOBPCGIterations( HYPRE_Solver solver );

  /* The implementation of the above */

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
hypre_LOBPCGSetPrecondUsageMode( void* data, int mode );

int 
hypre_LOBPCGSetPrintLevel( void *pcg_vdata, int level );

int 
hypre_LOBPCGSolve( void *pcg_vdata, hypre_MultiVectorPtr, 
		   hypre_MultiVectorPtr, double* );

utilities_FortranMatrix*
hypre_LOBPCGResidualNorms( void *pcg_vdata );

utilities_FortranMatrix*
hypre_LOBPCGResidualNormsHistory( void *pcg_vdata );

utilities_FortranMatrix*
hypre_LOBPCGEigenvaluesHistory( void *pcg_vdata );

int
hypre_LOBPCGIterations( void* pcg_vdata );

  /* applies the preconditioner T to a vector x: y = Tx */
void
hypre_LOBPCGPreconditioner( void *vdata, void* x, void* y );

  /* applies the operator A to a vector x: y = Ax */
void
hypre_LOBPCGOperatorA( void *pcg_vdata, void* x, void* y );

  /* applies the operator B to a vector x: y = Bx */
void
hypre_LOBPCGOperatorB( void *pcg_vdata, void* x, void* y );

  /* applies the preconditioner T to a multivector x: y = Tx */
void
hypre_LOBPCGMultiPreconditioner( void *data, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y );

  /* applies the operator A to a multivector x: y = Ax */
void
hypre_LOBPCGMultiOperatorA( void *data, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y );

  /* applies the operator B to a multivector x: y = Bx */
void
hypre_LOBPCGMultiOperatorB( void *data, hypre_MultiVectorPtr x, hypre_MultiVectorPtr y );

  /* solves the generalized eigenvalue problem mtxA x = lambda mtxB x using DSYGV */
int
lobpcg_solveGEVP( 
utilities_FortranMatrix* mtxA, 
utilities_FortranMatrix* mtxB,
utilities_FortranMatrix* lambda
);

  /* prototypes for DSYGV and DPOTRF routines from LAPACK */

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

void hypre_F90_NAME_BLAS( dpotrf, DPOTRF )
( char* uplo, int* n, double* aval, int* lda, int* ierr );

#endif

#ifdef __cplusplus
}
#endif

#endif /* HYPRE_LOBPCG_SOLVER */

