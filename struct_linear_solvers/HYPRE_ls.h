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

#include "HYPRE_utilities.h"
#include "HYPRE_mv.h"

#ifdef __cplusplus
extern "C" {
#endif


/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {int opaque;} *HYPRE_StructSolverBase;

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

#define HYPRE_StructHybridInitialize HYPRE_StructHybridInitializePush
#define HYPRE_StructHybridFinalize HYPRE_StructHybridFinalizePush
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
#define HYPRE_StructJacobiInitialize HYPRE_StructJacobiInitializePush
#define HYPRE_StructJacobiFinalize HYPRE_StructJacobiFinalizePush
#define HYPRE_StructJacobiSetup HYPRE_StructJacobiSetupPush
#define HYPRE_StructJacobiSolve HYPRE_StructJacobiSolvePush
#define HYPRE_StructJacobiSetTol HYPRE_StructJacobiSetTolPush
#define HYPRE_StructJacobiSetMaxIter HYPRE_StructJacobiSetMaxIterPush
#define HYPRE_StructJacobiSetZeroGuess HYPRE_StructJacobiSetZeroGuessPush
#define HYPRE_StructJacobiSetNonZeroGuess HYPRE_StructJacobiSetNonZeroGuessPush
#define HYPRE_StructJacobiGetNumIterations HYPRE_StructJacobiGetNumIterationsPush
#define HYPRE_StructJacobiGetFinalRelativeResidualNorm HYPRE_StructJacobiGetFinalRelativeResidualNormPush
#define HYPRE_StructPCGInitialize HYPRE_StructPCGInitializePush
#define HYPRE_StructPCGFinalize HYPRE_StructPCGFinalizePush
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
#define HYPRE_StructPFMGInitialize HYPRE_StructPFMGInitializePush
#define HYPRE_StructPFMGFinalize HYPRE_StructPFMGFinalizePush
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
#define HYPRE_StructSMGInitialize HYPRE_StructSMGInitializePush
#define HYPRE_StructSMGFinalize HYPRE_StructSMGFinalizePush
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
#define HYPRE_StructSparseMSGInitialize HYPRE_StructSparseMSGInitializePush
#define HYPRE_StructSparseMSGFinalize HYPRE_StructSparseMSGFinalizePush
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
#define HYPRE_StructSparseMSGSetLogging HYPRE_StructSparseMSGSetLoggingPush
#define HYPRE_StructSparseMSGGetNumIterations HYPRE_StructSparseMSGGetNumIterationsPush
#define HYPRE_StructSparseMSGGetFinalRelativeResidualNorm HYPRE_StructSparseMSGGetFinalRelativeResidualNormPush

#endif

# define	P(s) s

/* HYPRE_struct_hybrid.c */
int HYPRE_StructHybridInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructHybridFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructHybridSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructHybridSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructHybridSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructHybridSetConvergenceTol P((HYPRE_StructSolver solver , double cf_tol ));
int HYPRE_StructHybridSetDSCGMaxIter P((HYPRE_StructSolver solver , int dscg_max_its ));
int HYPRE_StructHybridSetPCGMaxIter P((HYPRE_StructSolver solver , int pcg_max_its ));
int HYPRE_StructHybridSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructHybridSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructHybridSetPrecond P((HYPRE_StructSolver solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver ));
int HYPRE_StructHybridSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructHybridGetNumIterations P((HYPRE_StructSolver solver , int *num_its ));
int HYPRE_StructHybridGetDSCGNumIterations P((HYPRE_StructSolver solver , int *dscg_num_its ));
int HYPRE_StructHybridGetPCGNumIterations P((HYPRE_StructSolver solver , int *pcg_num_its ));
int HYPRE_StructHybridGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_jacobi.c */
int HYPRE_StructJacobiInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructJacobiFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructJacobiSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructJacobiSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructJacobiSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructJacobiSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructJacobiSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructJacobiSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructJacobiGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructJacobiGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_pcg.c */
int HYPRE_StructPCGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructPCGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructPCGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructPCGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructPCGSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructPCGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructPCGSetPrecond P((HYPRE_StructSolver solver , hypre_PtrToStructSolverFcn precond , hypre_PtrToStructSolverFcn precond_setup , HYPRE_StructSolver precond_solver ));
int HYPRE_StructPCGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructPCGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructPCGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));
int HYPRE_StructDiagScaleSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x ));
int HYPRE_StructDiagScale P((HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx ));

/* HYPRE_struct_pfmg.c */
int HYPRE_StructPFMGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructPFMGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructPFMGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPFMGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPFMGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructPFMGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructPFMGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructPFMGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructPFMGSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructPFMGSetRelaxType P((HYPRE_StructSolver solver , int relax_type ));
int HYPRE_StructPFMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructPFMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructPFMGSetSkipRelax P((HYPRE_StructSolver solver , int skip_relax ));
int HYPRE_StructPFMGSetDxyz P((HYPRE_StructSolver solver , double *dxyz ));
int HYPRE_StructPFMGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructPFMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructPFMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_smg.c */
int HYPRE_StructSMGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructSMGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSMGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSMGSetMemoryUse P((HYPRE_StructSolver solver , int memory_use ));
int HYPRE_StructSMGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructSMGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructSMGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructSMGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructSMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructSMGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructSMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructSMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

/* HYPRE_struct_sparse_msg.c */
int HYPRE_StructSparseMSGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructSparseMSGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructSparseMSGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSparseMSGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructSparseMSGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructSparseMSGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructSparseMSGSetJump P((HYPRE_StructSolver solver , int jump ));
int HYPRE_StructSparseMSGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructSparseMSGSetZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSparseMSGSetNonZeroGuess P((HYPRE_StructSolver solver ));
int HYPRE_StructSparseMSGSetRelaxType P((HYPRE_StructSolver solver , int relax_type ));
int HYPRE_StructSparseMSGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructSparseMSGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructSparseMSGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructSparseMSGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructSparseMSGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

#undef P

#ifdef __cplusplus
}
#endif

#endif

