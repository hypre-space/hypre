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
# define        P(s) s
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

#ifdef HYPRE_USE_PTHREADS
void HYPRE_StructPCGInitializeVoidPtr P((void *argptr ));
HYPRE_StructSolver HYPRE_StructPCGInitializePush P((MPI_Comm comm ));
void HYPRE_StructPCGSolveVoidPtr P((void *argptr ));
int HYPRE_StructPCGSolvePush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
void HYPRE_StructDiagScaleVoidPtr P((void *argptr ));
int HYPRE_StructDiagScalePush P((HYPRE_StructSolver solver , HYPRE_StructMatrix HA , HYPRE_StructVector Hy , HYPRE_StructVector Hx ));

#ifndef NO_PTHREAD_MANGLING
#define HYPRE_StructPCGInitialize HYPRE_StructPCGInitializePush
#define HYPRE_StructPCGSolve HYPRE_StructPCGSolvePush
#define HYPRE_StructDiagScale HYPRE_StructDiagScalePush
#endif
#endif

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

#ifdef HYPRE_USE_PTHREADS
void HYPRE_StructSMGSetupVoidPtr P((void *argptr ));
int HYPRE_StructSMGSetupPush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));
void HYPRE_StructSMGSolveVoidPtr P((void *argptr ));
int HYPRE_StructSMGSolvePush P((HYPRE_StructSolver solver , HYPRE_StructMatrix A , HYPRE_StructVector b , HYPRE_StructVector x ));

#ifndef NO_PTHREAD_MANGLING
#define HYPRE_StructSMGSetup HYPRE_StructSMGSetupPush
#define HYPRE_StructSMGSolve HYPRE_StructSMGSolvePush
#endif
#endif

#undef P

#ifdef __cplusplus
}
#endif

#endif
