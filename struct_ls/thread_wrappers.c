
#ifdef HYPRE_USE_PTHREADS
#include "HYPRE_ls.h"
#include "utilities.h"


/*----------------------------------------------------------------
 * HYPRE_StructHybridCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridCreateArgs;

void
HYPRE_StructHybridCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridCreateArgs *localargs =
      (HYPRE_StructHybridCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridCreate(
         localargs -> comm,
         &(*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructHybridCreatePush(
   MPI_Comm comm,
   HYPRE_StructSolverArray *solver )
{
   HYPRE_StructHybridCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridDestroyArgs;

void
HYPRE_StructHybridDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridDestroyArgs *localargs =
      (HYPRE_StructHybridDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridDestroy(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructHybridDestroyPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructHybridDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetupArgs;

void
HYPRE_StructHybridSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetupArgs *localargs =
      (HYPRE_StructHybridSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructHybridSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructHybridSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSolve thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSolveArgs;

void
HYPRE_StructHybridSolveVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSolveArgs *localargs =
      (HYPRE_StructHybridSolveArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSolve(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructHybridSolvePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructHybridSolveArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSolveVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetTolArgs;

void
HYPRE_StructHybridSetTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetTolArgs *localargs =
      (HYPRE_StructHybridSetTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetTol(
         (*(localargs -> solver))[threadid],
         localargs -> tol );
}

int 
HYPRE_StructHybridSetTolPush(
   HYPRE_StructSolverArray solver,
   double tol )
{
   HYPRE_StructHybridSetTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.tol = tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetConvergenceTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double cf_tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetConvergenceTolArgs;

void
HYPRE_StructHybridSetConvergenceTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetConvergenceTolArgs *localargs =
      (HYPRE_StructHybridSetConvergenceTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetConvergenceTol(
         (*(localargs -> solver))[threadid],
         localargs -> cf_tol );
}

int 
HYPRE_StructHybridSetConvergenceTolPush(
   HYPRE_StructSolverArray solver,
   double cf_tol )
{
   HYPRE_StructHybridSetConvergenceTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.cf_tol = cf_tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetConvergenceTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetDSCGMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int dscg_max_its;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetDSCGMaxIterArgs;

void
HYPRE_StructHybridSetDSCGMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetDSCGMaxIterArgs *localargs =
      (HYPRE_StructHybridSetDSCGMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetDSCGMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> dscg_max_its );
}

int 
HYPRE_StructHybridSetDSCGMaxIterPush(
   HYPRE_StructSolverArray solver,
   int dscg_max_its )
{
   HYPRE_StructHybridSetDSCGMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.dscg_max_its = dscg_max_its;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetDSCGMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetPCGMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int pcg_max_its;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetPCGMaxIterArgs;

void
HYPRE_StructHybridSetPCGMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetPCGMaxIterArgs *localargs =
      (HYPRE_StructHybridSetPCGMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetPCGMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> pcg_max_its );
}

int 
HYPRE_StructHybridSetPCGMaxIterPush(
   HYPRE_StructSolverArray solver,
   int pcg_max_its )
{
   HYPRE_StructHybridSetPCGMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.pcg_max_its = pcg_max_its;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetPCGMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetTwoNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int two_norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetTwoNormArgs;

void
HYPRE_StructHybridSetTwoNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetTwoNormArgs *localargs =
      (HYPRE_StructHybridSetTwoNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetTwoNorm(
         (*(localargs -> solver))[threadid],
         localargs -> two_norm );
}

int 
HYPRE_StructHybridSetTwoNormPush(
   HYPRE_StructSolverArray solver,
   int two_norm )
{
   HYPRE_StructHybridSetTwoNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.two_norm = two_norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetTwoNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetRelChange thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int rel_change;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetRelChangeArgs;

void
HYPRE_StructHybridSetRelChangeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetRelChangeArgs *localargs =
      (HYPRE_StructHybridSetRelChangeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetRelChange(
         (*(localargs -> solver))[threadid],
         localargs -> rel_change );
}

int 
HYPRE_StructHybridSetRelChangePush(
   HYPRE_StructSolverArray solver,
   int rel_change )
{
   HYPRE_StructHybridSetRelChangeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.rel_change = rel_change;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetRelChangeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetPrecond thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   hypre_PtrToStructSolverFcn precond;
   hypre_PtrToStructSolverFcn precond_setup;
   HYPRE_StructSolverArray *precond_solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetPrecondArgs;

void
HYPRE_StructHybridSetPrecondVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetPrecondArgs *localargs =
      (HYPRE_StructHybridSetPrecondArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetPrecond(
         (*(localargs -> solver))[threadid],
         localargs -> precond,
         localargs -> precond_setup,
         (*(localargs -> precond_solver))[threadid] );
}

int 
HYPRE_StructHybridSetPrecondPush(
   HYPRE_StructSolverArray solver,
   hypre_PtrToStructSolverFcn precond,
   hypre_PtrToStructSolverFcn precond_setup,
   HYPRE_StructSolverArray precond_solver )
{
   HYPRE_StructHybridSetPrecondArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.precond = precond;
   pushargs.precond_setup = precond_setup;
   pushargs.precond_solver = (HYPRE_StructSolverArray *)precond_solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetPrecondVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridSetLogging thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int logging;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridSetLoggingArgs;

void
HYPRE_StructHybridSetLoggingVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridSetLoggingArgs *localargs =
      (HYPRE_StructHybridSetLoggingArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridSetLogging(
         (*(localargs -> solver))[threadid],
         localargs -> logging );
}

int 
HYPRE_StructHybridSetLoggingPush(
   HYPRE_StructSolverArray solver,
   int logging )
{
   HYPRE_StructHybridSetLoggingArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.logging = logging;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridSetLoggingVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridGetNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *num_its;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridGetNumIterationsArgs;

void
HYPRE_StructHybridGetNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridGetNumIterationsArgs *localargs =
      (HYPRE_StructHybridGetNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridGetNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> num_its );
}

int 
HYPRE_StructHybridGetNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *num_its )
{
   HYPRE_StructHybridGetNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_its = num_its;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridGetNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridGetDSCGNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *dscg_num_its;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridGetDSCGNumIterationsArgs;

void
HYPRE_StructHybridGetDSCGNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridGetDSCGNumIterationsArgs *localargs =
      (HYPRE_StructHybridGetDSCGNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridGetDSCGNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> dscg_num_its );
}

int 
HYPRE_StructHybridGetDSCGNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *dscg_num_its )
{
   HYPRE_StructHybridGetDSCGNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.dscg_num_its = dscg_num_its;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridGetDSCGNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridGetPCGNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *pcg_num_its;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridGetPCGNumIterationsArgs;

void
HYPRE_StructHybridGetPCGNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridGetPCGNumIterationsArgs *localargs =
      (HYPRE_StructHybridGetPCGNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridGetPCGNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> pcg_num_its );
}

int 
HYPRE_StructHybridGetPCGNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *pcg_num_its )
{
   HYPRE_StructHybridGetPCGNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.pcg_num_its = pcg_num_its;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridGetPCGNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructHybridGetFinalRelativeResidualNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructHybridGetFinalRelativeResidualNormArgs;

void
HYPRE_StructHybridGetFinalRelativeResidualNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructHybridGetFinalRelativeResidualNormArgs *localargs =
      (HYPRE_StructHybridGetFinalRelativeResidualNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructHybridGetFinalRelativeResidualNorm(
         (*(localargs -> solver))[threadid],
         localargs -> norm );
}

int 
HYPRE_StructHybridGetFinalRelativeResidualNormPush(
   HYPRE_StructSolverArray solver,
   double *norm )
{
   HYPRE_StructHybridGetFinalRelativeResidualNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.norm = norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructHybridGetFinalRelativeResidualNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiCreateArgs;

void
HYPRE_StructJacobiCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiCreateArgs *localargs =
      (HYPRE_StructJacobiCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiCreate(
         localargs -> comm,
         &(*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructJacobiCreatePush(
   MPI_Comm comm,
   HYPRE_StructSolverArray *solver )
{
   HYPRE_StructJacobiCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiDestroyArgs;

void
HYPRE_StructJacobiDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiDestroyArgs *localargs =
      (HYPRE_StructJacobiDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiDestroy(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructJacobiDestroyPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructJacobiDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiSetupArgs;

void
HYPRE_StructJacobiSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiSetupArgs *localargs =
      (HYPRE_StructJacobiSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructJacobiSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructJacobiSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiSolve thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiSolveArgs;

void
HYPRE_StructJacobiSolveVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiSolveArgs *localargs =
      (HYPRE_StructJacobiSolveArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiSolve(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructJacobiSolvePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructJacobiSolveArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiSolveVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiSetTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiSetTolArgs;

void
HYPRE_StructJacobiSetTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiSetTolArgs *localargs =
      (HYPRE_StructJacobiSetTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiSetTol(
         (*(localargs -> solver))[threadid],
         localargs -> tol );
}

int 
HYPRE_StructJacobiSetTolPush(
   HYPRE_StructSolverArray solver,
   double tol )
{
   HYPRE_StructJacobiSetTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.tol = tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiSetTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiSetMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int max_iter;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiSetMaxIterArgs;

void
HYPRE_StructJacobiSetMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiSetMaxIterArgs *localargs =
      (HYPRE_StructJacobiSetMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiSetMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> max_iter );
}

int 
HYPRE_StructJacobiSetMaxIterPush(
   HYPRE_StructSolverArray solver,
   int max_iter )
{
   HYPRE_StructJacobiSetMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.max_iter = max_iter;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiSetMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiSetZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiSetZeroGuessArgs;

void
HYPRE_StructJacobiSetZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiSetZeroGuessArgs *localargs =
      (HYPRE_StructJacobiSetZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiSetZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructJacobiSetZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructJacobiSetZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiSetZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiSetNonZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiSetNonZeroGuessArgs;

void
HYPRE_StructJacobiSetNonZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiSetNonZeroGuessArgs *localargs =
      (HYPRE_StructJacobiSetNonZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiSetNonZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructJacobiSetNonZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructJacobiSetNonZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiSetNonZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiGetNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *num_iterations;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiGetNumIterationsArgs;

void
HYPRE_StructJacobiGetNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiGetNumIterationsArgs *localargs =
      (HYPRE_StructJacobiGetNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiGetNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> num_iterations );
}

int 
HYPRE_StructJacobiGetNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *num_iterations )
{
   HYPRE_StructJacobiGetNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_iterations = num_iterations;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiGetNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructJacobiGetFinalRelativeResidualNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructJacobiGetFinalRelativeResidualNormArgs;

void
HYPRE_StructJacobiGetFinalRelativeResidualNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructJacobiGetFinalRelativeResidualNormArgs *localargs =
      (HYPRE_StructJacobiGetFinalRelativeResidualNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructJacobiGetFinalRelativeResidualNorm(
         (*(localargs -> solver))[threadid],
         localargs -> norm );
}

int 
HYPRE_StructJacobiGetFinalRelativeResidualNormPush(
   HYPRE_StructSolverArray solver,
   double *norm )
{
   HYPRE_StructJacobiGetFinalRelativeResidualNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.norm = norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructJacobiGetFinalRelativeResidualNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGCreateArgs;

void
HYPRE_StructPCGCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGCreateArgs *localargs =
      (HYPRE_StructPCGCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGCreate(
         localargs -> comm,
         &(*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructPCGCreatePush(
   MPI_Comm comm,
   HYPRE_StructSolverArray *solver )
{
   HYPRE_StructPCGCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGDestroyArgs;

void
HYPRE_StructPCGDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGDestroyArgs *localargs =
      (HYPRE_StructPCGDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGDestroy(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructPCGDestroyPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructPCGDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetupArgs;

void
HYPRE_StructPCGSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetupArgs *localargs =
      (HYPRE_StructPCGSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructPCGSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructPCGSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSolve thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSolveArgs;

void
HYPRE_StructPCGSolveVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSolveArgs *localargs =
      (HYPRE_StructPCGSolveArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSolve(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructPCGSolvePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructPCGSolveArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSolveVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetTolArgs;

void
HYPRE_StructPCGSetTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetTolArgs *localargs =
      (HYPRE_StructPCGSetTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetTol(
         (*(localargs -> solver))[threadid],
         localargs -> tol );
}

int 
HYPRE_StructPCGSetTolPush(
   HYPRE_StructSolverArray solver,
   double tol )
{
   HYPRE_StructPCGSetTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.tol = tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int max_iter;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetMaxIterArgs;

void
HYPRE_StructPCGSetMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetMaxIterArgs *localargs =
      (HYPRE_StructPCGSetMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> max_iter );
}

int 
HYPRE_StructPCGSetMaxIterPush(
   HYPRE_StructSolverArray solver,
   int max_iter )
{
   HYPRE_StructPCGSetMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.max_iter = max_iter;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetTwoNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int two_norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetTwoNormArgs;

void
HYPRE_StructPCGSetTwoNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetTwoNormArgs *localargs =
      (HYPRE_StructPCGSetTwoNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetTwoNorm(
         (*(localargs -> solver))[threadid],
         localargs -> two_norm );
}

int 
HYPRE_StructPCGSetTwoNormPush(
   HYPRE_StructSolverArray solver,
   int two_norm )
{
   HYPRE_StructPCGSetTwoNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.two_norm = two_norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetTwoNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetRelChange thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int rel_change;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetRelChangeArgs;

void
HYPRE_StructPCGSetRelChangeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetRelChangeArgs *localargs =
      (HYPRE_StructPCGSetRelChangeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetRelChange(
         (*(localargs -> solver))[threadid],
         localargs -> rel_change );
}

int 
HYPRE_StructPCGSetRelChangePush(
   HYPRE_StructSolverArray solver,
   int rel_change )
{
   HYPRE_StructPCGSetRelChangeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.rel_change = rel_change;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetRelChangeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetPrecond thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   hypre_PtrToStructSolverFcn precond;
   hypre_PtrToStructSolverFcn precond_setup;
   HYPRE_StructSolverArray *precond_solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetPrecondArgs;

void
HYPRE_StructPCGSetPrecondVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetPrecondArgs *localargs =
      (HYPRE_StructPCGSetPrecondArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetPrecond(
         (*(localargs -> solver))[threadid],
         localargs -> precond,
         localargs -> precond_setup,
         (*(localargs -> precond_solver))[threadid] );
}

int 
HYPRE_StructPCGSetPrecondPush(
   HYPRE_StructSolverArray solver,
   hypre_PtrToStructSolverFcn precond,
   hypre_PtrToStructSolverFcn precond_setup,
   HYPRE_StructSolverArray precond_solver )
{
   HYPRE_StructPCGSetPrecondArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.precond = precond;
   pushargs.precond_setup = precond_setup;
   pushargs.precond_solver = (HYPRE_StructSolverArray *)precond_solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetPrecondVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGSetLogging thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int logging;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGSetLoggingArgs;

void
HYPRE_StructPCGSetLoggingVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGSetLoggingArgs *localargs =
      (HYPRE_StructPCGSetLoggingArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGSetLogging(
         (*(localargs -> solver))[threadid],
         localargs -> logging );
}

int 
HYPRE_StructPCGSetLoggingPush(
   HYPRE_StructSolverArray solver,
   int logging )
{
   HYPRE_StructPCGSetLoggingArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.logging = logging;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGSetLoggingVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGGetNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *num_iterations;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGGetNumIterationsArgs;

void
HYPRE_StructPCGGetNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGGetNumIterationsArgs *localargs =
      (HYPRE_StructPCGGetNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGGetNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> num_iterations );
}

int 
HYPRE_StructPCGGetNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *num_iterations )
{
   HYPRE_StructPCGGetNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_iterations = num_iterations;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGGetNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPCGGetFinalRelativeResidualNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPCGGetFinalRelativeResidualNormArgs;

void
HYPRE_StructPCGGetFinalRelativeResidualNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPCGGetFinalRelativeResidualNormArgs *localargs =
      (HYPRE_StructPCGGetFinalRelativeResidualNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPCGGetFinalRelativeResidualNorm(
         (*(localargs -> solver))[threadid],
         localargs -> norm );
}

int 
HYPRE_StructPCGGetFinalRelativeResidualNormPush(
   HYPRE_StructSolverArray solver,
   double *norm )
{
   HYPRE_StructPCGGetFinalRelativeResidualNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.norm = norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPCGGetFinalRelativeResidualNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructDiagScaleSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *y;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructDiagScaleSetupArgs;

void
HYPRE_StructDiagScaleSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructDiagScaleSetupArgs *localargs =
      (HYPRE_StructDiagScaleSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructDiagScaleSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> y))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructDiagScaleSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray y,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructDiagScaleSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.y = (HYPRE_StructVectorArray *)y;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructDiagScaleSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructDiagScale thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *HA;
   HYPRE_StructVectorArray *Hy;
   HYPRE_StructVectorArray *Hx;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructDiagScaleArgs;

void
HYPRE_StructDiagScaleVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructDiagScaleArgs *localargs =
      (HYPRE_StructDiagScaleArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructDiagScale(
         (*(localargs -> solver))[threadid],
         (*(localargs -> HA))[threadid],
         (*(localargs -> Hy))[threadid],
         (*(localargs -> Hx))[threadid] );
}

int 
HYPRE_StructDiagScalePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray HA,
   HYPRE_StructVectorArray Hy,
   HYPRE_StructVectorArray Hx )
{
   HYPRE_StructDiagScaleArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.HA = (HYPRE_StructMatrixArray *)HA;
   pushargs.Hy = (HYPRE_StructVectorArray *)Hy;
   pushargs.Hx = (HYPRE_StructVectorArray *)Hx;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructDiagScaleVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGCreateArgs;

void
HYPRE_StructPFMGCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGCreateArgs *localargs =
      (HYPRE_StructPFMGCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGCreate(
         localargs -> comm,
         &(*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructPFMGCreatePush(
   MPI_Comm comm,
   HYPRE_StructSolverArray *solver )
{
   HYPRE_StructPFMGCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGDestroyArgs;

void
HYPRE_StructPFMGDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGDestroyArgs *localargs =
      (HYPRE_StructPFMGDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGDestroy(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructPFMGDestroyPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructPFMGDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetupArgs;

void
HYPRE_StructPFMGSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetupArgs *localargs =
      (HYPRE_StructPFMGSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructPFMGSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructPFMGSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSolve thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSolveArgs;

void
HYPRE_StructPFMGSolveVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSolveArgs *localargs =
      (HYPRE_StructPFMGSolveArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSolve(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructPFMGSolvePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructPFMGSolveArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSolveVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetTolArgs;

void
HYPRE_StructPFMGSetTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetTolArgs *localargs =
      (HYPRE_StructPFMGSetTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetTol(
         (*(localargs -> solver))[threadid],
         localargs -> tol );
}

int 
HYPRE_StructPFMGSetTolPush(
   HYPRE_StructSolverArray solver,
   double tol )
{
   HYPRE_StructPFMGSetTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.tol = tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int max_iter;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetMaxIterArgs;

void
HYPRE_StructPFMGSetMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetMaxIterArgs *localargs =
      (HYPRE_StructPFMGSetMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> max_iter );
}

int 
HYPRE_StructPFMGSetMaxIterPush(
   HYPRE_StructSolverArray solver,
   int max_iter )
{
   HYPRE_StructPFMGSetMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.max_iter = max_iter;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int rel_change;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetRelChangeArgs;

void
HYPRE_StructPFMGSetRelChangeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetRelChangeArgs *localargs =
      (HYPRE_StructPFMGSetRelChangeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetRelChange(
         (*(localargs -> solver))[threadid],
         localargs -> rel_change );
}

int 
HYPRE_StructPFMGSetRelChangePush(
   HYPRE_StructSolverArray solver,
   int rel_change )
{
   HYPRE_StructPFMGSetRelChangeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.rel_change = rel_change;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetRelChangeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetZeroGuessArgs;

void
HYPRE_StructPFMGSetZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetZeroGuessArgs *localargs =
      (HYPRE_StructPFMGSetZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructPFMGSetZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructPFMGSetZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetNonZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetNonZeroGuessArgs;

void
HYPRE_StructPFMGSetNonZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetNonZeroGuessArgs *localargs =
      (HYPRE_StructPFMGSetNonZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetNonZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructPFMGSetNonZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructPFMGSetNonZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetNonZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetRelaxType thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int relax_type;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetRelaxTypeArgs;

void
HYPRE_StructPFMGSetRelaxTypeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetRelaxTypeArgs *localargs =
      (HYPRE_StructPFMGSetRelaxTypeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetRelaxType(
         (*(localargs -> solver))[threadid],
         localargs -> relax_type );
}

int 
HYPRE_StructPFMGSetRelaxTypePush(
   HYPRE_StructSolverArray solver,
   int relax_type )
{
   HYPRE_StructPFMGSetRelaxTypeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.relax_type = relax_type;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetRelaxTypeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int num_pre_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetNumPreRelaxArgs;

void
HYPRE_StructPFMGSetNumPreRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetNumPreRelaxArgs *localargs =
      (HYPRE_StructPFMGSetNumPreRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetNumPreRelax(
         (*(localargs -> solver))[threadid],
         localargs -> num_pre_relax );
}

int 
HYPRE_StructPFMGSetNumPreRelaxPush(
   HYPRE_StructSolverArray solver,
   int num_pre_relax )
{
   HYPRE_StructPFMGSetNumPreRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_pre_relax = num_pre_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetNumPreRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int num_post_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetNumPostRelaxArgs;

void
HYPRE_StructPFMGSetNumPostRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetNumPostRelaxArgs *localargs =
      (HYPRE_StructPFMGSetNumPostRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetNumPostRelax(
         (*(localargs -> solver))[threadid],
         localargs -> num_post_relax );
}

int 
HYPRE_StructPFMGSetNumPostRelaxPush(
   HYPRE_StructSolverArray solver,
   int num_post_relax )
{
   HYPRE_StructPFMGSetNumPostRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_post_relax = num_post_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetNumPostRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetSkipRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int skip_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetSkipRelaxArgs;

void
HYPRE_StructPFMGSetSkipRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetSkipRelaxArgs *localargs =
      (HYPRE_StructPFMGSetSkipRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetSkipRelax(
         (*(localargs -> solver))[threadid],
         localargs -> skip_relax );
}

int 
HYPRE_StructPFMGSetSkipRelaxPush(
   HYPRE_StructSolverArray solver,
   int skip_relax )
{
   HYPRE_StructPFMGSetSkipRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.skip_relax = skip_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetSkipRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *dxyz;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetDxyzArgs;

void
HYPRE_StructPFMGSetDxyzVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetDxyzArgs *localargs =
      (HYPRE_StructPFMGSetDxyzArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetDxyz(
         (*(localargs -> solver))[threadid],
         localargs -> dxyz );
}

int 
HYPRE_StructPFMGSetDxyzPush(
   HYPRE_StructSolverArray solver,
   double *dxyz )
{
   HYPRE_StructPFMGSetDxyzArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.dxyz = dxyz;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetDxyzVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int logging;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGSetLoggingArgs;

void
HYPRE_StructPFMGSetLoggingVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGSetLoggingArgs *localargs =
      (HYPRE_StructPFMGSetLoggingArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGSetLogging(
         (*(localargs -> solver))[threadid],
         localargs -> logging );
}

int 
HYPRE_StructPFMGSetLoggingPush(
   HYPRE_StructSolverArray solver,
   int logging )
{
   HYPRE_StructPFMGSetLoggingArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.logging = logging;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGSetLoggingVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGGetNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *num_iterations;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGGetNumIterationsArgs;

void
HYPRE_StructPFMGGetNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGGetNumIterationsArgs *localargs =
      (HYPRE_StructPFMGGetNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGGetNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> num_iterations );
}

int 
HYPRE_StructPFMGGetNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *num_iterations )
{
   HYPRE_StructPFMGGetNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_iterations = num_iterations;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGGetNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructPFMGGetFinalRelativeResidualNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructPFMGGetFinalRelativeResidualNormArgs;

void
HYPRE_StructPFMGGetFinalRelativeResidualNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructPFMGGetFinalRelativeResidualNormArgs *localargs =
      (HYPRE_StructPFMGGetFinalRelativeResidualNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructPFMGGetFinalRelativeResidualNorm(
         (*(localargs -> solver))[threadid],
         localargs -> norm );
}

int 
HYPRE_StructPFMGGetFinalRelativeResidualNormPush(
   HYPRE_StructSolverArray solver,
   double *norm )
{
   HYPRE_StructPFMGGetFinalRelativeResidualNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.norm = norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructPFMGGetFinalRelativeResidualNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGCreateArgs;

void
HYPRE_StructSMGCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGCreateArgs *localargs =
      (HYPRE_StructSMGCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGCreate(
         localargs -> comm,
         &(*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSMGCreatePush(
   MPI_Comm comm,
   HYPRE_StructSolverArray *solver )
{
   HYPRE_StructSMGCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGDestroyArgs;

void
HYPRE_StructSMGDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGDestroyArgs *localargs =
      (HYPRE_StructSMGDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGDestroy(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSMGDestroyPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructSMGDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetupArgs;

void
HYPRE_StructSMGSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetupArgs *localargs =
      (HYPRE_StructSMGSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructSMGSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructSMGSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSolve thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSolveArgs;

void
HYPRE_StructSMGSolveVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSolveArgs *localargs =
      (HYPRE_StructSMGSolveArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSolve(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructSMGSolvePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructSMGSolveArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSolveVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int memory_use;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetMemoryUseArgs;

void
HYPRE_StructSMGSetMemoryUseVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetMemoryUseArgs *localargs =
      (HYPRE_StructSMGSetMemoryUseArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetMemoryUse(
         (*(localargs -> solver))[threadid],
         localargs -> memory_use );
}

int 
HYPRE_StructSMGSetMemoryUsePush(
   HYPRE_StructSolverArray solver,
   int memory_use )
{
   HYPRE_StructSMGSetMemoryUseArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.memory_use = memory_use;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetMemoryUseVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetTolArgs;

void
HYPRE_StructSMGSetTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetTolArgs *localargs =
      (HYPRE_StructSMGSetTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetTol(
         (*(localargs -> solver))[threadid],
         localargs -> tol );
}

int 
HYPRE_StructSMGSetTolPush(
   HYPRE_StructSolverArray solver,
   double tol )
{
   HYPRE_StructSMGSetTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.tol = tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int max_iter;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetMaxIterArgs;

void
HYPRE_StructSMGSetMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetMaxIterArgs *localargs =
      (HYPRE_StructSMGSetMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> max_iter );
}

int 
HYPRE_StructSMGSetMaxIterPush(
   HYPRE_StructSolverArray solver,
   int max_iter )
{
   HYPRE_StructSMGSetMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.max_iter = max_iter;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int rel_change;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetRelChangeArgs;

void
HYPRE_StructSMGSetRelChangeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetRelChangeArgs *localargs =
      (HYPRE_StructSMGSetRelChangeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetRelChange(
         (*(localargs -> solver))[threadid],
         localargs -> rel_change );
}

int 
HYPRE_StructSMGSetRelChangePush(
   HYPRE_StructSolverArray solver,
   int rel_change )
{
   HYPRE_StructSMGSetRelChangeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.rel_change = rel_change;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetRelChangeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetZeroGuessArgs;

void
HYPRE_StructSMGSetZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetZeroGuessArgs *localargs =
      (HYPRE_StructSMGSetZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSMGSetZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructSMGSetZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetNonZeroGuessArgs;

void
HYPRE_StructSMGSetNonZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetNonZeroGuessArgs *localargs =
      (HYPRE_StructSMGSetNonZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetNonZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSMGSetNonZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructSMGSetNonZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetNonZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int num_pre_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetNumPreRelaxArgs;

void
HYPRE_StructSMGSetNumPreRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetNumPreRelaxArgs *localargs =
      (HYPRE_StructSMGSetNumPreRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetNumPreRelax(
         (*(localargs -> solver))[threadid],
         localargs -> num_pre_relax );
}

int 
HYPRE_StructSMGSetNumPreRelaxPush(
   HYPRE_StructSolverArray solver,
   int num_pre_relax )
{
   HYPRE_StructSMGSetNumPreRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_pre_relax = num_pre_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetNumPreRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int num_post_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetNumPostRelaxArgs;

void
HYPRE_StructSMGSetNumPostRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetNumPostRelaxArgs *localargs =
      (HYPRE_StructSMGSetNumPostRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetNumPostRelax(
         (*(localargs -> solver))[threadid],
         localargs -> num_post_relax );
}

int 
HYPRE_StructSMGSetNumPostRelaxPush(
   HYPRE_StructSolverArray solver,
   int num_post_relax )
{
   HYPRE_StructSMGSetNumPostRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_post_relax = num_post_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetNumPostRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGSetLogging thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int logging;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGSetLoggingArgs;

void
HYPRE_StructSMGSetLoggingVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGSetLoggingArgs *localargs =
      (HYPRE_StructSMGSetLoggingArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGSetLogging(
         (*(localargs -> solver))[threadid],
         localargs -> logging );
}

int 
HYPRE_StructSMGSetLoggingPush(
   HYPRE_StructSolverArray solver,
   int logging )
{
   HYPRE_StructSMGSetLoggingArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.logging = logging;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGSetLoggingVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *num_iterations;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGGetNumIterationsArgs;

void
HYPRE_StructSMGGetNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGGetNumIterationsArgs *localargs =
      (HYPRE_StructSMGGetNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGGetNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> num_iterations );
}

int 
HYPRE_StructSMGGetNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *num_iterations )
{
   HYPRE_StructSMGGetNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_iterations = num_iterations;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGGetNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSMGGetFinalRelativeResidualNormArgs;

void
HYPRE_StructSMGGetFinalRelativeResidualNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSMGGetFinalRelativeResidualNormArgs *localargs =
      (HYPRE_StructSMGGetFinalRelativeResidualNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSMGGetFinalRelativeResidualNorm(
         (*(localargs -> solver))[threadid],
         localargs -> norm );
}

int 
HYPRE_StructSMGGetFinalRelativeResidualNormPush(
   HYPRE_StructSolverArray solver,
   double *norm )
{
   HYPRE_StructSMGGetFinalRelativeResidualNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.norm = norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSMGGetFinalRelativeResidualNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGCreate thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   MPI_Comm comm;
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGCreateArgs;

void
HYPRE_StructSparseMSGCreateVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGCreateArgs *localargs =
      (HYPRE_StructSparseMSGCreateArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGCreate(
         localargs -> comm,
         &(*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSparseMSGCreatePush(
   MPI_Comm comm,
   HYPRE_StructSolverArray *solver )
{
   HYPRE_StructSparseMSGCreateArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.comm = comm;
   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGCreateVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGDestroy thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGDestroyArgs;

void
HYPRE_StructSparseMSGDestroyVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGDestroyArgs *localargs =
      (HYPRE_StructSparseMSGDestroyArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGDestroy(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSparseMSGDestroyPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructSparseMSGDestroyArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGDestroyVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetup thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetupArgs;

void
HYPRE_StructSparseMSGSetupVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetupArgs *localargs =
      (HYPRE_StructSparseMSGSetupArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetup(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructSparseMSGSetupPush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructSparseMSGSetupArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetupVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSolve thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   HYPRE_StructMatrixArray *A;
   HYPRE_StructVectorArray *b;
   HYPRE_StructVectorArray *x;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSolveArgs;

void
HYPRE_StructSparseMSGSolveVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSolveArgs *localargs =
      (HYPRE_StructSparseMSGSolveArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSolve(
         (*(localargs -> solver))[threadid],
         (*(localargs -> A))[threadid],
         (*(localargs -> b))[threadid],
         (*(localargs -> x))[threadid] );
}

int 
HYPRE_StructSparseMSGSolvePush(
   HYPRE_StructSolverArray solver,
   HYPRE_StructMatrixArray A,
   HYPRE_StructVectorArray b,
   HYPRE_StructVectorArray x )
{
   HYPRE_StructSparseMSGSolveArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.A = (HYPRE_StructMatrixArray *)A;
   pushargs.b = (HYPRE_StructVectorArray *)b;
   pushargs.x = (HYPRE_StructVectorArray *)x;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSolveVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetTol thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double tol;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetTolArgs;

void
HYPRE_StructSparseMSGSetTolVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetTolArgs *localargs =
      (HYPRE_StructSparseMSGSetTolArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetTol(
         (*(localargs -> solver))[threadid],
         localargs -> tol );
}

int 
HYPRE_StructSparseMSGSetTolPush(
   HYPRE_StructSolverArray solver,
   double tol )
{
   HYPRE_StructSparseMSGSetTolArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.tol = tol;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetTolVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetMaxIter thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int max_iter;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetMaxIterArgs;

void
HYPRE_StructSparseMSGSetMaxIterVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetMaxIterArgs *localargs =
      (HYPRE_StructSparseMSGSetMaxIterArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetMaxIter(
         (*(localargs -> solver))[threadid],
         localargs -> max_iter );
}

int 
HYPRE_StructSparseMSGSetMaxIterPush(
   HYPRE_StructSolverArray solver,
   int max_iter )
{
   HYPRE_StructSparseMSGSetMaxIterArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.max_iter = max_iter;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetMaxIterVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJump thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int jump;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetJumpArgs;

void
HYPRE_StructSparseMSGSetJumpVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetJumpArgs *localargs =
      (HYPRE_StructSparseMSGSetJumpArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetJump(
         (*(localargs -> solver))[threadid],
         localargs -> jump );
}

int 
HYPRE_StructSparseMSGSetJumpPush(
   HYPRE_StructSolverArray solver,
   int jump )
{
   HYPRE_StructSparseMSGSetJumpArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.jump = jump;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetJumpVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelChange thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int rel_change;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetRelChangeArgs;

void
HYPRE_StructSparseMSGSetRelChangeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetRelChangeArgs *localargs =
      (HYPRE_StructSparseMSGSetRelChangeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetRelChange(
         (*(localargs -> solver))[threadid],
         localargs -> rel_change );
}

int 
HYPRE_StructSparseMSGSetRelChangePush(
   HYPRE_StructSolverArray solver,
   int rel_change )
{
   HYPRE_StructSparseMSGSetRelChangeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.rel_change = rel_change;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetRelChangeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetZeroGuessArgs;

void
HYPRE_StructSparseMSGSetZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetZeroGuessArgs *localargs =
      (HYPRE_StructSparseMSGSetZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSparseMSGSetZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructSparseMSGSetZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNonZeroGuess thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetNonZeroGuessArgs;

void
HYPRE_StructSparseMSGSetNonZeroGuessVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetNonZeroGuessArgs *localargs =
      (HYPRE_StructSparseMSGSetNonZeroGuessArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetNonZeroGuess(
         (*(localargs -> solver))[threadid] );
}

int 
HYPRE_StructSparseMSGSetNonZeroGuessPush(
   HYPRE_StructSolverArray solver )
{
   HYPRE_StructSparseMSGSetNonZeroGuessArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetNonZeroGuessVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelaxType thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int relax_type;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetRelaxTypeArgs;

void
HYPRE_StructSparseMSGSetRelaxTypeVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetRelaxTypeArgs *localargs =
      (HYPRE_StructSparseMSGSetRelaxTypeArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetRelaxType(
         (*(localargs -> solver))[threadid],
         localargs -> relax_type );
}

int 
HYPRE_StructSparseMSGSetRelaxTypePush(
   HYPRE_StructSolverArray solver,
   int relax_type )
{
   HYPRE_StructSparseMSGSetRelaxTypeArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.relax_type = relax_type;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetRelaxTypeVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPreRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int num_pre_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetNumPreRelaxArgs;

void
HYPRE_StructSparseMSGSetNumPreRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetNumPreRelaxArgs *localargs =
      (HYPRE_StructSparseMSGSetNumPreRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetNumPreRelax(
         (*(localargs -> solver))[threadid],
         localargs -> num_pre_relax );
}

int 
HYPRE_StructSparseMSGSetNumPreRelaxPush(
   HYPRE_StructSolverArray solver,
   int num_pre_relax )
{
   HYPRE_StructSparseMSGSetNumPreRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_pre_relax = num_pre_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetNumPreRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPostRelax thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int num_post_relax;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetNumPostRelaxArgs;

void
HYPRE_StructSparseMSGSetNumPostRelaxVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetNumPostRelaxArgs *localargs =
      (HYPRE_StructSparseMSGSetNumPostRelaxArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetNumPostRelax(
         (*(localargs -> solver))[threadid],
         localargs -> num_post_relax );
}

int 
HYPRE_StructSparseMSGSetNumPostRelaxPush(
   HYPRE_StructSolverArray solver,
   int num_post_relax )
{
   HYPRE_StructSparseMSGSetNumPostRelaxArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_post_relax = num_post_relax;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetNumPostRelaxVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGSetLogging thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int logging;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGSetLoggingArgs;

void
HYPRE_StructSparseMSGSetLoggingVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGSetLoggingArgs *localargs =
      (HYPRE_StructSparseMSGSetLoggingArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGSetLogging(
         (*(localargs -> solver))[threadid],
         localargs -> logging );
}

int 
HYPRE_StructSparseMSGSetLoggingPush(
   HYPRE_StructSolverArray solver,
   int logging )
{
   HYPRE_StructSparseMSGSetLoggingArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.logging = logging;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGSetLoggingVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGGetNumIterations thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   int *num_iterations;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGGetNumIterationsArgs;

void
HYPRE_StructSparseMSGGetNumIterationsVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGGetNumIterationsArgs *localargs =
      (HYPRE_StructSparseMSGGetNumIterationsArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGGetNumIterations(
         (*(localargs -> solver))[threadid],
         localargs -> num_iterations );
}

int 
HYPRE_StructSparseMSGGetNumIterationsPush(
   HYPRE_StructSolverArray solver,
   int *num_iterations )
{
   HYPRE_StructSparseMSGGetNumIterationsArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.num_iterations = num_iterations;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGGetNumIterationsVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

/*----------------------------------------------------------------
 * HYPRE_StructSparseMSGGetFinalRelativeResidualNorm thread wrappers
 *----------------------------------------------------------------*/

typedef struct {
   HYPRE_StructSolverArray *solver;
   double *norm;
   int  returnvalue[hypre_MAX_THREADS];
} HYPRE_StructSparseMSGGetFinalRelativeResidualNormArgs;

void
HYPRE_StructSparseMSGGetFinalRelativeResidualNormVoidPtr( void *argptr )
{
   int threadid = hypre_GetThreadID();

   HYPRE_StructSparseMSGGetFinalRelativeResidualNormArgs *localargs =
      (HYPRE_StructSparseMSGGetFinalRelativeResidualNormArgs *) argptr;

   (localargs -> returnvalue[threadid]) =
      HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(
         (*(localargs -> solver))[threadid],
         localargs -> norm );
}

int 
HYPRE_StructSparseMSGGetFinalRelativeResidualNormPush(
   HYPRE_StructSolverArray solver,
   double *norm )
{
   HYPRE_StructSparseMSGGetFinalRelativeResidualNormArgs pushargs;
   int i;
   int  returnvalue;

   pushargs.solver = (HYPRE_StructSolverArray *)solver;
   pushargs.norm = norm;
   for (i = 0; i < hypre_NumThreads; i++)
      hypre_work_put( HYPRE_StructSparseMSGGetFinalRelativeResidualNormVoidPtr, (void *)&pushargs );

   hypre_work_wait();

   returnvalue = pushargs.returnvalue[0];

   return returnvalue;
}

#else

/* this is used only to eliminate compiler warnings */
int hypre_empty;

#endif

