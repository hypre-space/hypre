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

#ifndef HYPRE_USE_PTHREADS
#define NO_PTHREAD_MANGLING
#endif

/*--------------------------------------------------------------------------
 * Structures
 *--------------------------------------------------------------------------*/

typedef struct {int opaque;} *HYPRE_StructSolver;

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* HYPRE_struct_pcg.c */
int HYPRE_StructPCGInitialize P((MPI_Comm comm , HYPRE_StructSolver *solver ));
int HYPRE_StructPCGFinalize P((HYPRE_StructSolver solver ));
int HYPRE_StructPCGSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSolve P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
int HYPRE_StructPCGSetTol P((HYPRE_StructSolver solver , double tol ));
int HYPRE_StructPCGSetMaxIter P((HYPRE_StructSolver solver , int max_iter ));
int HYPRE_StructPCGSetTwoNorm P((HYPRE_StructSolver solver , int two_norm ));
int HYPRE_StructPCGSetRelChange P((HYPRE_StructSolver solver , int rel_change ));
int HYPRE_StructPCGSetPrecond P((HYPRE_StructSolver solver , int (*precond )(), int (*precond_setup )(), void *precond_data ));
int HYPRE_StructPCGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructPCGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructPCGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));
int HYPRE_StructDiagScaleSetup P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x ));
int HYPRE_StructDiagScale P((HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx ));

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
int HYPRE_StructSMGSetNumPreRelax P((HYPRE_StructSolver solver , int num_pre_relax ));
int HYPRE_StructSMGSetNumPostRelax P((HYPRE_StructSolver solver , int num_post_relax ));
int HYPRE_StructSMGSetLogging P((HYPRE_StructSolver solver , int logging ));
int HYPRE_StructSMGGetNumIterations P((HYPRE_StructSolver solver , int *num_iterations ));
int HYPRE_StructSMGGetFinalRelativeResidualNorm P((HYPRE_StructSolver solver , double *norm ));

#undef P
#ifdef __STDC__
# define	P(s) s
#else
# define P(s) ()
#endif


/* thread_wrappers.c */
void HYPRE_StructPCGInitializeVoidPtr P((void *argptr ));
int HYPRE_StructPCGInitializePush P((MPI_Comm comm , HYPRE_StructSolver *solver ));
void HYPRE_StructPCGFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructPCGFinalizePush P((HYPRE_StructSolver solver ));
void HYPRE_StructPCGSetupVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetupPush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
void HYPRE_StructPCGSolveVoidPtr P((void *argptr ));
int HYPRE_StructPCGSolvePush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
void HYPRE_StructPCGSetTolVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetTolPush P((HYPRE_StructSolver solver , double tol ));
void HYPRE_StructPCGSetMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetMaxIterPush P((HYPRE_StructSolver solver , int max_iter ));
void HYPRE_StructPCGSetTwoNormVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetTwoNormPush P((HYPRE_StructSolver solver , int two_norm ));
void HYPRE_StructPCGSetRelChangeVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetRelChangePush P((HYPRE_StructSolver solver , int rel_change ));
void HYPRE_StructPCGSetPrecondVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetPrecondPush P((HYPRE_StructSolver solver , int , int , void *precond_data ));
void HYPRE_StructPCGSetLoggingVoidPtr P((void *argptr ));
int HYPRE_StructPCGSetLoggingPush P((HYPRE_StructSolver solver , int logging ));
void HYPRE_StructPCGGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructPCGGetNumIterationsPush P((HYPRE_StructSolver solver , int *num_iterations ));
void HYPRE_StructPCGGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructPCGGetFinalRelativeResidualNormPush P((HYPRE_StructSolver solver , double *norm ));
void HYPRE_StructDiagScaleSetupVoidPtr P((void *argptr ));
int HYPRE_StructDiagScaleSetupPush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector y , HYPRE_StructVector x ));
void HYPRE_StructDiagScaleVoidPtr P((void *argptr ));
int HYPRE_StructDiagScalePush P((HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx ));
void HYPRE_StructSMGInitializeVoidPtr P((void *argptr ));
int HYPRE_StructSMGInitializePush P((MPI_Comm comm , HYPRE_StructSolver *solver ));
void HYPRE_StructSMGFinalizeVoidPtr P((void *argptr ));
int HYPRE_StructSMGFinalizePush P((HYPRE_StructSolver solver ));
void HYPRE_StructSMGSetupVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetupPush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
void HYPRE_StructSMGSolveVoidPtr P((void *argptr ));
int HYPRE_StructSMGSolvePush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
void HYPRE_StructSMGSetMemoryUseVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetMemoryUsePush P((HYPRE_StructSolver solver , int memory_use ));
void HYPRE_StructSMGSetTolVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetTolPush P((HYPRE_StructSolver solver , double tol ));
void HYPRE_StructSMGSetMaxIterVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetMaxIterPush P((HYPRE_StructSolver solver , int max_iter ));
void HYPRE_StructSMGSetRelChangeVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetRelChangePush P((HYPRE_StructSolver solver , int rel_change ));
void HYPRE_StructSMGSetZeroGuessVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetZeroGuessPush P((HYPRE_StructSolver solver ));
void HYPRE_StructSMGSetNumPreRelaxVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetNumPreRelaxPush P((HYPRE_StructSolver solver , int num_pre_relax ));
void HYPRE_StructSMGSetNumPostRelaxVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetNumPostRelaxPush P((HYPRE_StructSolver solver , int num_post_relax ));
void HYPRE_StructSMGSetLoggingVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetLoggingPush P((HYPRE_StructSolver solver , int logging ));
void HYPRE_StructSMGGetNumIterationsVoidPtr P((void *argptr ));
int HYPRE_StructSMGGetNumIterationsPush P((HYPRE_StructSolver solver , int *num_iterations ));
void HYPRE_StructSMGGetFinalRelativeResidualNormVoidPtr P((void *argptr ));
int HYPRE_StructSMGGetFinalRelativeResidualNormPush P((HYPRE_StructSolver solver , double *norm ));

#undef P

#ifdef __cplusplus
}
#endif

#endif
