/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructPCG interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgcreate, HYPRE_SSTRUCTPCGCREATE)
                                                     (long int *comm,
                                                      long int *solver,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGCreate( (MPI_Comm)             *comm,
                                          (HYPRE_SStructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgdestroy, HYPRE_SSTRUCTPCGDESTROY)
                                                     (long int *solver,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetup, HYPRE_SSTRUCTPCGDESTROY)
                                                     (long int *solver,
                                                      long int *A,
                                                      long int *b,
                                                      long int *x,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetup( (HYPRE_SStructSolver) *solver,
                                         (HYPRE_SStructMatrix) *A,
                                         (HYPRE_SStructVector) *b,
                                         (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsolve, HYPRE_SSTRUCTPCGSOLVE)
                                                     (long int *solver,
                                                      long int *A,
                                                      long int *b,
                                                      long int *x,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSolve( (HYPRE_SStructSolver) *solver,
                                         (HYPRE_SStructMatrix) *A,
                                         (HYPRE_SStructVector) *b,
                                         (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsettol, HYPRE_SSTRUCTPCGSETTOL)
                                                     (long int *solver,
                                                      double   *tol,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetTol( (HYPRE_SStructSolver) *solver,
                                          (double)              *tol ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetmaxiter, HYPRE_SSTRUCTPCGSETMAXITER)
                                                     (long int *solver,
                                                      int      *max_iter,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetMaxIter( (HYPRE_SStructSolver) *solver,
                                              (int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsettwonorm, HYPRE_SSTRUCTPCGSETTWONORM)
                                                     (long int *solver,
                                                      int      *two_norm,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetTwoNorm( (HYPRE_SStructSolver) *solver,
                                              (int)                 *two_norm ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetrelchange, HYPRE_SSTRUCTPCGSETRELCHANGE)
                                                     (long int *solver,
                                                      int      *rel_change,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetRelChange( (HYPRE_SStructSolver) *solver,
                                                (int)                 *rel_change ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetprecond, HYPRE_SSTRUCTPCGSETPRECOND)
                                                     (long int *solver,
                                                      int      *precond_id,
                                                      long int *precond_solver,
                                                      int      *ierr)
/*------------------------------------------
 *    precond_id flags mean:
 *    2 - setup a split-solver preconditioner
 *    3 - setup a syspfmg preconditioner
 *    8 - setup a DiagScale preconditioner
 *    9 - no preconditioner setup
 *----------------------------------------*/

{
   if(*precond_id == 2)
      {
       *ierr = (int)
               (HYPRE_SStructPCGSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSplitSolve,
                                             HYPRE_SStructSplitSetup,
                                            (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (int)
               (HYPRE_SStructPCGSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSysPFMGSolve,
                                             HYPRE_SStructSysPFMGSetup,
                                            (HYPRE_SStructSolver *)   precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (int)
               (HYPRE_SStructPCGSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructDiagScale,
                                             HYPRE_SStructDiagScaleSetup,
                                            (HYPRE_SStructSolver *)   precond_solver));
      }
   else if(*precond_id == 9)
      {
       *ierr = 0;
      }

   else
      {
       *ierr = -1;
      }

}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetlogging, HYPRE_SSTRUCTPCGSETLOGGING)
                                                     (long int *solver,
                                                      int      *logging,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetLogging( (HYPRE_SStructSolver) *solver,
                                              (int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcgsetprintlevel, HYPRE_SSTRUCTPCGSETPRINTLEVEL)
                                                     (long int *solver,
                                                      int      *level,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                 (int)                 *level ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetnumiteration, HYPRE_SSTRUCTPCGGETNUMITERATION)
                                                     (long int *solver,
                                                      int      *num_iterations,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                    (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetfinalrelativ, HYPRE_SSTRUCTPCGGETFINALRELATIV)
                                                     (long int *solver,
                                                      double   *norm,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPCGGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpcggetresidual, HYPRE_SSTRUCTPCGGETRESIDUAL)
                                                     (long int *solver,
                                                      long int *residual,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructPCGGetResidual( (HYPRE_SStructSolver) *solver,
                                               (void *)              *residual ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructdiagscalesetup, HYPRE_SSTRUCTDIAGSCALESETUP)
                                                     (long int *solver,
                                                      long int *A,
                                                      long int *y,
                                                      long int *x,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructDiagScaleSetup( (HYPRE_SStructSolver) *solver,
                                               (HYPRE_SStructMatrix) *A,
                                               (HYPRE_SStructVector) *y,
                                               (HYPRE_SStructVector) *x    ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructdiagscale, HYPRE_SSTRUCTDIAGSCALE)
                                                     (long int *solver,
                                                      long int *A,
                                                      long int *y,
                                                      long int *x,
                                                      int      *ierr)
{
   *ierr = (int) (HYPRE_SStructDiagScale( (HYPRE_SStructSolver) *solver,
                                          (HYPRE_SStructMatrix) *A,
                                          (HYPRE_SStructVector) *y,
                                          (HYPRE_SStructVector) *x    ) );
}
