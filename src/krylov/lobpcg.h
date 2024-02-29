/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "multivector.h"
#include "_hypre_utilities.h"
#include "fortran_matrix.h"
#include "HYPRE_MatvecFunctions.h"
#include "HYPRE_krylov.h"

#ifdef HYPRE_MIXED_PRECISION
#include "krylov_mup_func.h"
#endif

#ifndef LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS
#define LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS

#ifdef __cplusplus
extern "C" {
#endif

#define PROBLEM_SIZE_TOO_SMALL           1
#define WRONG_BLOCK_SIZE                 2
#define WRONG_CONSTRAINTS                3
#define REQUESTED_ACCURACY_NOT_ACHIEVED -1

typedef struct
{

   HYPRE_Real   absolute;
   HYPRE_Real   relative;

} lobpcg_Tolerance;

typedef struct
{
   HYPRE_Int    (*Precond)(void*, void*, void*, void*);
   HYPRE_Int    (*PrecondSetup)(void*, void*, void*, void*);

} hypre_LOBPCGPrecond;

typedef struct
{
   lobpcg_Tolerance              tolerance;
   HYPRE_Int                           maxIterations;
   HYPRE_Int                           verbosityLevel;
   HYPRE_Int                           precondUsageMode;
   HYPRE_Int                           iterationNumber;
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

typedef struct
{

   /* these pointers should point to 2 functions providing standard lapack  functionality */
   HYPRE_Int   (*dpotrf) (const char *uplo, HYPRE_Int *n, HYPRE_Real *a, HYPRE_Int *
                          lda, HYPRE_Int *info);
   HYPRE_Int   (*dsygv) (HYPRE_Int *itype, char *jobz, char *uplo, HYPRE_Int *
                         n, HYPRE_Real *a, HYPRE_Int *lda, HYPRE_Real *b, HYPRE_Int *ldb,
                         HYPRE_Real *w, HYPRE_Real *work, HYPRE_Int *lwork, HYPRE_Int *info);

} lobpcg_BLASLAPACKFunctions;

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
              HYPRE_Real * lambda_values,

              /* eigenvalues history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matrix s
              tored
              in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see next
              argument; If you don't need eigenvalues history, provide NULL in this entry */
              HYPRE_Real * lambdaHistory_values,

              /* global height of the matrix (stored in fotran-style)  specified by previous argument */
              HYPRE_BigInt lambdaHistory_gh,

              /* residual norms; argument should point to array of <blocksize> doubles */
              HYPRE_Real * residualNorms_values,

              /* residual norms history; a pointer to the entries of the  <blocksize>-by-(<maxIterations>+1) matri
              x
              stored in  fortran-style. (i.e. column-wise) The matrix may be  a submatrix of a larger matrix, see
              next
              argument If you don't need residual norms history, provide NULL in this entry */
              HYPRE_Real * residualNormsHistory_values,

              /* global height of the matrix (stored in fotran-style)  specified by previous argument */
              HYPRE_BigInt residualNormsHistory_gh

            );

HYPRE_Int
lobpcg_initialize( lobpcg_Data* data );
HYPRE_Int
lobpcg_clean( lobpcg_Data* data );
HYPRE_Int
hypre_LOBPCGDestroy( void *pcg_vdata );
HYPRE_Int
hypre_LOBPCGSetup( void *pcg_vdata, void *A, void *b, void *x );
HYPRE_Int
hypre_LOBPCGSetupB( void *pcg_vdata, void *B, void *x );
HYPRE_Int
hypre_LOBPCGSetupT( void *pcg_vdata, void *T, void *x );
HYPRE_Int
hypre_LOBPCGSetTol( void* pcg_vdata, HYPRE_Real tol );
HYPRE_Int
hypre_LOBPCGSetRTol( void* pcg_vdata, HYPRE_Real tol );
HYPRE_Int
hypre_LOBPCGSetMaxIter( void* pcg_vdata, HYPRE_Int max_iter  );
HYPRE_Int
hypre_LOBPCGSetPrecondUsageMode( void* pcg_vdata, HYPRE_Int mode  );
HYPRE_Int
hypre_LOBPCGGetPrecond( void         *pcg_vdata,
                        HYPRE_Solver *precond_data_ptr );
HYPRE_Int
hypre_LOBPCGSetPrecond( void  *pcg_vdata,
                        HYPRE_Int  (*precond)(void*, void*, void*, void*),
                        HYPRE_Int  (*precond_setup)(void*, void*, void*, void*),
                        void  *precond_data );
HYPRE_Int
hypre_LOBPCGSetPrintLevel( void *pcg_vdata, HYPRE_Int level );
void
hypre_LOBPCGPreconditioner( void *vdata, void* x, void* y );
void
hypre_LOBPCGOperatorA( void *pcg_vdata, void* x, void* y );
void
hypre_LOBPCGOperatorB( void *pcg_vdata, void* x, void* y );
void
hypre_LOBPCGMultiPreconditioner( void *data, void * x, void*  y );
void
hypre_LOBPCGMultiOperatorA( void *data, void * x, void*  y );
void
hypre_LOBPCGMultiOperatorB( void *data, void * x, void*  y );
HYPRE_Int
hypre_LOBPCGSolve( void *vdata,
                   mv_MultiVectorPtr con,
                   mv_MultiVectorPtr vec,
                   HYPRE_Real* val );
utilities_FortranMatrix*
hypre_LOBPCGResidualNorms( void *vdata );
utilities_FortranMatrix*
hypre_LOBPCGResidualNormsHistory( void *vdata );
utilities_FortranMatrix*
hypre_LOBPCGEigenvaluesHistory( void *vdata );
HYPRE_Int
hypre_LOBPCGIterations( void* vdata );
void hypre_LOBPCGMultiOperatorB(void *data,
                                void *x,
                                void *y);

void lobpcg_MultiVectorByMultiVector(mv_MultiVectorPtr        x,
                                     mv_MultiVectorPtr        y,
                                     utilities_FortranMatrix *xy);

#ifdef __cplusplus
}
#endif

#endif /* LOCALLY_OPTIMAL_BLOCK_PRECONDITIONED_CONJUGATE_GRADIENTS */
