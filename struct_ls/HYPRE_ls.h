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

/*--------------------------------------------------------------------------
 * Prototypes
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NO_PTHREAD_MANGLING

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
#define HYPRE_StructSMGInitialize HYPRE_StructSMGInitializePush
#define HYPRE_StructSMGFinalize HYPRE_StructSMGFinalizePush
#define HYPRE_StructSMGSetup HYPRE_StructSMGSetupPush
#define HYPRE_StructSMGSolve HYPRE_StructSMGSolvePush
#define HYPRE_StructSMGSetMemoryUse HYPRE_StructSMGSetMemoryUsePush
#define HYPRE_StructSMGSetTol HYPRE_StructSMGSetTolPush
#define HYPRE_StructSMGSetMaxIter HYPRE_StructSMGSetMaxIterPush
#define HYPRE_StructSMGSetRelChange HYPRE_StructSMGSetRelChangePush
#define HYPRE_StructSMGSetZeroGuess HYPRE_StructSMGSetZeroGuessPush
#define HYPRE_StructSMGSetNumPreRelax HYPRE_StructSMGSetNumPreRelaxPush
#define HYPRE_StructSMGSetNumPostRelax HYPRE_StructSMGSetNumPostRelaxPush
#define HYPRE_StructSMGSetLogging HYPRE_StructSMGSetLoggingPush
#define HYPRE_StructSMGGetNumIterations HYPRE_StructSMGGetNumIterationsPush
#define HYPRE_StructSMGGetFinalRelativeResidualNorm HYPRE_StructSMGGetFinalRelativeResidualNormPush

#endif

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

#ifdef __cplusplus
}
#endif

#endif

