/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "fortran_matrix.h"
#include "multivector.h"

#ifndef LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS
#define LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS

#define PROBLEM_SIZE_TOO_SMALL			       	1
#define WRONG_BLOCK_SIZE			       	2
#define WRONG_CONSTRAINTS                               3
#define REQUESTED_ACCURACY_NOT_ACHIEVED			-1

typedef struct {

  double	absolute;
  double	relative;

} lobpcg_Tolerance;

typedef struct
{

  lobpcg_Tolerance	       	tolerance;
  int		       	       	maxIterations;
  int		      	       	verbosityLevel;
  int		       		precondUsageMode;

  int	       	       		iterationNumber;

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

#ifdef __cplusplus
extern "C" {
#endif

int
lobpcg_initialize( lobpcg_Data* data );

int
lobpcg_clean( lobpcg_Data* data );

int
lobpcg_solve( hypre_MultiVectorPtr blockVectorX,
	      void* operatorAData,
	      void (*operatorA)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr ),
	      void* operatorBData,
	      void (*operatorB)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr ),
	      void* operatorTData,
	      void (*operatorT)( void*, hypre_MultiVectorPtr, hypre_MultiVectorPtr ),
	      hypre_MultiVectorPtr blockVectorY,
	      lobpcg_Tolerance tolerance,
	      int maxIterations,
	      int verbosityLevel,
	      int* iterationNumber,
	      utilities_FortranMatrix* lambda,
	      utilities_FortranMatrix* lambdaHistory,
	      utilities_FortranMatrix* residualNorms,
	      utilities_FortranMatrix* residualNormsHistory 
	);

void
lobpcg_MultiVectorByMultiVector(
hypre_MultiVectorPtr x,
hypre_MultiVectorPtr y,
utilities_FortranMatrix* xy
);

void
lobpcg_MultiVectorByMatrix(
hypre_MultiVectorPtr x,
utilities_FortranMatrix* r,
hypre_MultiVectorPtr y
);

int
lobpcg_MultiVectorImplicitQR( 
hypre_MultiVectorPtr x,  hypre_MultiVectorPtr y, 
utilities_FortranMatrix* r,
hypre_MultiVectorPtr z
);

void
lobpcg_sqrtVector( int n, int* mask, double* v );

int
lobpcg_checkResiduals( 
utilities_FortranMatrix* resNorms,
utilities_FortranMatrix* lambda,
lobpcg_Tolerance tol,
int* activeMask
);

int
lobpcg_solveGEVP( 
utilities_FortranMatrix* mtxA, 
utilities_FortranMatrix* mtxB,
utilities_FortranMatrix* eigVal
);

int
lobpcg_chol( utilities_FortranMatrix* a );

void
lobpcg_errorMessage( int, char* );

#ifdef __cplusplus
}
#endif

#endif /* LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS */

