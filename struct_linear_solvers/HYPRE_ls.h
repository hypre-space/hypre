/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Header file for HYPRE_ls library
 *
 *****************************************************************************/

#ifndef HYPRE_LS_HEADER
#define HYPRE_LS_HEADER

#include <HYPRE_config.h>

#include "HYPRE_utilities.h"
#include "HYPRE_mv.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

struct hypre_StructSolver_struct;
typedef struct hypre_StructSolver_struct *HYPRE_StructSolverBase;

#ifndef HYPRE_USE_PTHREADS
#define hypre_MAX_THREADS 1
#ifndef HYPRE_NO_PTHREAD_MANGLING
#define HYPRE_NO_PTHREAD_MANGLING
#endif
#endif

typedef HYPRE_StructSolverBase HYPRE_StructSolverArray[hypre_MAX_THREADS];

#ifdef HYPRE_NO_PTHREAD_MANGLING
typedef HYPRE_StructSolverBase HYPRE_StructSolver;
#else
typedef HYPRE_StructSolverArray HYPRE_StructSolver;
#endif

/* This is temporary to help keep our pthreads version up-to-date */
typedef int (*hypre_PtrToStructSolverFcn)(HYPRE_StructSolver,
                                          HYPRE_StructMatrix,
                                          HYPRE_StructVector,
                                          HYPRE_StructVector);

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/


#ifndef HYPRE_NO_PTHREAD_MANGLING

#define HYPRE_StructHybridCreate HYPRE_StructHybridCreatePush
#define HYPRE_StructHybridDestroy HYPRE_StructHybridDestroyPush
#define HYPRE_StructHybridSetup HYPRE_StructHybridSetupPush
#define HYPRE_StructHybridSolve HYPRE_StructHybridSolvePush
#define HYPRE_StructHybridSetTol HYPRE_StructHybridSetTolPush
#define HYPRE_StructHybridSetConvergenceTol HYPRE_StructHybridSetConvergenceTolPush
#define HYPRE_StructHybridSetDSCGMaxIter HYPRE_StructHybridSetDSCGMaxIterPush
#define HYPRE_StructHybridSetPCGMaxIter HYPRE_StructHybridSetPCGMaxIterPush
#define HYPRE_StructHybridSetTwoNorm HYPRE_StructHybridSetTwoNormPush
#define HYPRE_StructHybridSetRelChange HYPRE_StructHybridSetRelChangePush
#define HYPRE_StructHybridSetPrecond HYPRE_StructHybridSetPrecondPush
#define HYPRE_StructHybridSetLogging HYPRE_StructHybridSetLoggingPush
#define HYPRE_StructHybridGetNumIterations HYPRE_StructHybridGetNumIterationsPush
#define HYPRE_StructHybridGetDSCGNumIterations HYPRE_StructHybridGetDSCGNumIterationsPush
#define HYPRE_StructHybridGetPCGNumIterations HYPRE_StructHybridGetPCGNumIterationsPush
#define HYPRE_StructHybridGetFinalRelativeResidualNorm HYPRE_StructHybridGetFinalRelativeResidualNormPush
#define HYPRE_StructJacobiCreate HYPRE_StructJacobiCreatePush
#define HYPRE_StructJacobiDestroy HYPRE_StructJacobiDestroyPush
#define HYPRE_StructJacobiSetup HYPRE_StructJacobiSetupPush
#define HYPRE_StructJacobiSolve HYPRE_StructJacobiSolvePush
#define HYPRE_StructJacobiSetTol HYPRE_StructJacobiSetTolPush
#define HYPRE_StructJacobiSetMaxIter HYPRE_StructJacobiSetMaxIterPush
#define HYPRE_StructJacobiSetZeroGuess HYPRE_StructJacobiSetZeroGuessPush
#define HYPRE_StructJacobiSetNonZeroGuess HYPRE_StructJacobiSetNonZeroGuessPush
#define HYPRE_StructJacobiGetNumIterations HYPRE_StructJacobiGetNumIterationsPush
#define HYPRE_StructJacobiGetFinalRelativeResidualNorm HYPRE_StructJacobiGetFinalRelativeResidualNormPush
#define HYPRE_StructPCGCreate HYPRE_StructPCGCreatePush
#define HYPRE_StructPCGDestroy HYPRE_StructPCGDestroyPush
#define HYPRE_StructPCGSetup HYPRE_StructPCGSetupPush
#define HYPRE_StructPCGSolve HYPRE_StructPCGSolvePush
#define HYPRE_StructPCGSetTol HYPRE_StructPCGSetTolPush
#define HYPRE_StructPCGSetMaxIter HYPRE_StructPCGSetMaxIterPush
#define HYPRE_StructPCGSetTwoNorm HYPRE_StructPCGSetTwoNormPush
#define HYPRE_StructPCGSetRelChange HYPRE_StructPCGSetRelChangePush
#define HYPRE_StructPCGSetPrecond HYPRE_StructPCGSetPrecondPush
#define HYPRE_StructPCGSetLogging HYPRE_StructPCGSetLoggingPush
#define HYPRE_StructPCGGetNumIterations HYPRE_StructPCGGetNumIterationsPush
#define HYPRE_StructPCGGetFinalRelativeResidualNorm HYPRE_StructPCGGetFinalRelativeResidualNormPush
#define HYPRE_StructDiagScaleSetup HYPRE_StructDiagScaleSetupPush
#define HYPRE_StructDiagScale HYPRE_StructDiagScalePush
#define HYPRE_StructPFMGCreate HYPRE_StructPFMGCreatePush
#define HYPRE_StructPFMGDestroy HYPRE_StructPFMGDestroyPush
#define HYPRE_StructPFMGSetup HYPRE_StructPFMGSetupPush
#define HYPRE_StructPFMGSolve HYPRE_StructPFMGSolvePush
#define HYPRE_StructPFMGSetTol HYPRE_StructPFMGSetTolPush
#define HYPRE_StructPFMGSetMaxIter HYPRE_StructPFMGSetMaxIterPush
#define HYPRE_StructPFMGSetRelChange HYPRE_StructPFMGSetRelChangePush
#define HYPRE_StructPFMGSetZeroGuess HYPRE_StructPFMGSetZeroGuessPush
#define HYPRE_StructPFMGSetNonZeroGuess HYPRE_StructPFMGSetNonZeroGuessPush
#define HYPRE_StructPFMGSetRelaxType HYPRE_StructPFMGSetRelaxTypePush
#define HYPRE_StructPFMGSetNumPreRelax HYPRE_StructPFMGSetNumPreRelaxPush
#define HYPRE_StructPFMGSetNumPostRelax HYPRE_StructPFMGSetNumPostRelaxPush
#define HYPRE_StructPFMGSetSkipRelax HYPRE_StructPFMGSetSkipRelaxPush
#define HYPRE_StructPFMGSetDxyz HYPRE_StructPFMGSetDxyzPush
#define HYPRE_StructPFMGSetLogging HYPRE_StructPFMGSetLoggingPush
#define HYPRE_StructPFMGGetNumIterations HYPRE_StructPFMGGetNumIterationsPush
#define HYPRE_StructPFMGGetFinalRelativeResidualNorm HYPRE_StructPFMGGetFinalRelativeResidualNormPush
#define HYPRE_StructSMGCreate HYPRE_StructSMGCreatePush
#define HYPRE_StructSMGDestroy HYPRE_StructSMGDestroyPush
#define HYPRE_StructSMGSetup HYPRE_StructSMGSetupPush
#define HYPRE_StructSMGSolve HYPRE_StructSMGSolvePush
#define HYPRE_StructSMGSetMemoryUse HYPRE_StructSMGSetMemoryUsePush
#define HYPRE_StructSMGSetTol HYPRE_StructSMGSetTolPush
#define HYPRE_StructSMGSetMaxIter HYPRE_StructSMGSetMaxIterPush
#define HYPRE_StructSMGSetRelChange HYPRE_StructSMGSetRelChangePush
#define HYPRE_StructSMGSetZeroGuess HYPRE_StructSMGSetZeroGuessPush
#define HYPRE_StructSMGSetNonZeroGuess HYPRE_StructSMGSetNonZeroGuessPush
#define HYPRE_StructSMGSetNumPreRelax HYPRE_StructSMGSetNumPreRelaxPush
#define HYPRE_StructSMGSetNumPostRelax HYPRE_StructSMGSetNumPostRelaxPush
#define HYPRE_StructSMGSetLogging HYPRE_StructSMGSetLoggingPush
#define HYPRE_StructSMGGetNumIterations HYPRE_StructSMGGetNumIterationsPush
#define HYPRE_StructSMGGetFinalRelativeResidualNorm HYPRE_StructSMGGetFinalRelativeResidualNormPush
#define HYPRE_StructSparseMSGCreate HYPRE_StructSparseMSGCreatePush
#define HYPRE_StructSparseMSGDestroy HYPRE_StructSparseMSGDestroyPush
#define HYPRE_StructSparseMSGSetup HYPRE_StructSparseMSGSetupPush
#define HYPRE_StructSparseMSGSolve HYPRE_StructSparseMSGSolvePush
#define HYPRE_StructSparseMSGSetTol HYPRE_StructSparseMSGSetTolPush
#define HYPRE_StructSparseMSGSetMaxIter HYPRE_StructSparseMSGSetMaxIterPush
#define HYPRE_StructSparseMSGSetJump HYPRE_StructSparseMSGSetJumpPush
#define HYPRE_StructSparseMSGSetRelChange HYPRE_StructSparseMSGSetRelChangePush
#define HYPRE_StructSparseMSGSetZeroGuess HYPRE_StructSparseMSGSetZeroGuessPush
#define HYPRE_StructSparseMSGSetNonZeroGuess HYPRE_StructSparseMSGSetNonZeroGuessPush
#define HYPRE_StructSparseMSGSetRelaxType HYPRE_StructSparseMSGSetRelaxTypePush
#define HYPRE_StructSparseMSGSetNumPreRelax HYPRE_StructSparseMSGSetNumPreRelaxPush
#define HYPRE_StructSparseMSGSetNumPostRelax HYPRE_StructSparseMSGSetNumPostRelaxPush
#define HYPRE_StructSparseMSGSetNumFineRelax HYPRE_StructSparseMSGSetNumFineRelaxPush
#define HYPRE_StructSparseMSGSetLogging HYPRE_StructSparseMSGSetLoggingPush
#define HYPRE_StructSparseMSGGetNumIterations HYPRE_StructSparseMSGGetNumIterationsPush
#define HYPRE_StructSparseMSGGetFinalRelativeResidualNorm HYPRE_StructSparseMSGGetFinalRelativeResidualNormPush

#endif


/* HYPRE_struct_hybrid.c */
int HYPRE_StructHybridCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructHybridDestroy( HYPRE_StructSolver solver );
int HYPRE_StructHybridSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructHybridSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructHybridSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructHybridSetConvergenceTol( HYPRE_StructSolver solver , double cf_tol );
int HYPRE_StructHybridSetDSCGMaxIter( HYPRE_StructSolver solver , int dscg_max_its );
int HYPRE_StructHybridSetPCGMaxIter( HYPRE_StructSolver solver , int pcg_max_its );
int HYPRE_StructHybridSetTwoNorm( HYPRE_StructSolver solver , int two_norm );
int HYPRE_StructHybridSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructHybridSetPrecond( HYPRE_StructSolver solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
int HYPRE_StructHybridSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructHybridGetNumIterations( HYPRE_StructSolver solver , int *num_its );
int HYPRE_StructHybridGetDSCGNumIterations( HYPRE_StructSolver solver , int *dscg_num_its );
int HYPRE_StructHybridGetPCGNumIterations( HYPRE_StructSolver solver , int *pcg_num_its );
int HYPRE_StructHybridGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_jacobi.c */
int HYPRE_StructJacobiCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructJacobiDestroy( HYPRE_StructSolver solver );
int HYPRE_StructJacobiSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructJacobiSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructJacobiSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructJacobiSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructJacobiSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructJacobiSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructJacobiGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructJacobiGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_pcg.c */
int HYPRE_StructPCGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructPCGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructPCGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPCGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPCGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructPCGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructPCGSetTwoNorm( HYPRE_StructSolver solver , int two_norm );
int HYPRE_StructPCGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructPCGSetPrecond( HYPRE_StructSolver solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver );
int HYPRE_StructPCGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructPCGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructPCGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );
int HYPRE_StructDiagScaleSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x );
int HYPRE_StructDiagScale( HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx );

/* HYPRE_struct_pfmg.c */
int HYPRE_StructPFMGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructPFMGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructPFMGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPFMGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructPFMGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructPFMGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructPFMGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructPFMGSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructPFMGSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructPFMGSetRelaxType( HYPRE_StructSolver solver , int relax_type );
int HYPRE_StructPFMGSetNumPreRelax( HYPRE_StructSolver solver , int num_pre_relax );
int HYPRE_StructPFMGSetNumPostRelax( HYPRE_StructSolver solver , int num_post_relax );
int HYPRE_StructPFMGSetSkipRelax( HYPRE_StructSolver solver , int skip_relax );
int HYPRE_StructPFMGSetDxyz( HYPRE_StructSolver solver , double *dxyz );
int HYPRE_StructPFMGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructPFMGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructPFMGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_smg.c */
int HYPRE_StructSMGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructSMGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructSMGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSMGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSMGSetMemoryUse( HYPRE_StructSolver solver , int memory_use );
int HYPRE_StructSMGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructSMGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructSMGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructSMGSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSMGSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSMGSetNumPreRelax( HYPRE_StructSolver solver , int num_pre_relax );
int HYPRE_StructSMGSetNumPostRelax( HYPRE_StructSolver solver , int num_post_relax );
int HYPRE_StructSMGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructSMGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructSMGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

/* HYPRE_struct_sparse_msg.c */
int HYPRE_StructSparseMSGCreate( MPI_Comm comm , HYPRE_StructSolver *solver );
int HYPRE_StructSparseMSGDestroy( HYPRE_StructSolver solver );
int HYPRE_StructSparseMSGSetup( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSparseMSGSolve( HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x );
int HYPRE_StructSparseMSGSetTol( HYPRE_StructSolver solver , double tol );
int HYPRE_StructSparseMSGSetMaxIter( HYPRE_StructSolver solver , int max_iter );
int HYPRE_StructSparseMSGSetJump( HYPRE_StructSolver solver , int jump );
int HYPRE_StructSparseMSGSetRelChange( HYPRE_StructSolver solver , int rel_change );
int HYPRE_StructSparseMSGSetZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSparseMSGSetNonZeroGuess( HYPRE_StructSolver solver );
int HYPRE_StructSparseMSGSetRelaxType( HYPRE_StructSolver solver , int relax_type );
int HYPRE_StructSparseMSGSetNumPreRelax( HYPRE_StructSolver solver , int num_pre_relax );
int HYPRE_StructSparseMSGSetNumPostRelax( HYPRE_StructSolver solver , int num_post_relax );
int HYPRE_StructSparseMSGSetNumFineRelax( HYPRE_StructSolver solver , int num_fine_relax );
int HYPRE_StructSparseMSGSetLogging( HYPRE_StructSolver solver , int logging );
int HYPRE_StructSparseMSGGetNumIterations( HYPRE_StructSolver solver , int *num_iterations );
int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm( HYPRE_StructSolver solver , double *norm );

#ifdef __cplusplus
}
#endif

#endif

