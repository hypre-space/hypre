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
 * HYPRE_SStructGMRES interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmrescreate, HYPRE_SSTRUCTGMRESCREATE)
                                                       (long int  *comm,
                                                        long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESCreate( (MPI_Comm)              *comm,
                                            (HYPRE_SStructSolver *)  solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresdestroy, HYPRE_SSTRUCTGMRESDESTROY)
                                                       (long int  *solver,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetup, HYPRE_SSTRUCTGMRESSETUP)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressolve, HYPRE_SSTRUCTGMRESSOLVE)
                                                       (long int  *solver,
                                                        long int  *A,
                                                        long int  *b,
                                                        long int  *x,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetkdim, HYPRE_SSTRUCTGMRESSETKDIM)
                                                       (long int  *solver,
                                                        int       *k_dim,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetKDim( (HYPRE_SStructSolver) *solver,
                                             (int)                 *k_dim ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressettol, HYPRE_SSTRUCTGMRESSETTOL)
                                                       (long int  *solver,
                                                        double    *tol,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetminiter, HYPRE_SSTRUCTGMRESSETMINITER)
                                                       (long int  *solver,
                                                        int       *min_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetMinIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetmaxiter, HYPRE_SSTRUCTGMRESSETMAXITER)
                                                       (long int  *solver,
                                                        int       *max_iter,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetstopcrit, HYPRE_SSTRUCTGMRESSETSTOPCRIT)
                                                       (long int  *solver,
                                                        int       *stop_crit,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetStopCrit( (HYPRE_SStructSolver) *solver,
                                                  (int)                *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprecond, HYPRE_SSTRUCTGMRESSETPRECOND)
                                                     (long int *solver,
                                                      int      *precond_id,
                                                      long int *precond_solver,
                                                      int      *ierr)
/*------------------------------------------
 *    precond_id flags mean:
 *    0 - setup a smg preconditioner
 *    1 - setup a pfmg preconditioner
 *    2 - setup a split-solver preconditioner
 *    3 - setup a syspfmg preconditioner
 *    4 - setup a ParCSRPilut preconditioner
 *    5 - setup a ParCSRParaSails preconditioner
 *    6 - setup a BoomerAMG preconditioner
 *    7 - setup a ParCSRDiagScale preconditioner
 *    8 - setup a DiagScale preconditioner
 *    9 - no preconditioner setup
 *----------------------------------------*/

{
   if(*precond_id == 0)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_StructSMGSolve,
                                               HYPRE_StructSMGSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 1)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_StructPFMGSolve,
                                               HYPRE_StructPFMGSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 2)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSplitSolve,
                                               HYPRE_SStructSplitSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructSysPFMGSolve,
                                               HYPRE_SStructSysPFMGSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 4)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_ParCSRPilutSolve,
                                               HYPRE_ParCSRPilutSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 5)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_ParCSRParaSailsSolve,
                                               HYPRE_ParCSRParaSailsSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 6)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_BoomerAMGSolve,
                                               HYPRE_BoomerAMGSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 7)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_ParCSRDiagScale,
                                               HYPRE_ParCSRDiagScaleSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (int)
               (HYPRE_SStructGMRESSetPrecond( (HYPRE_SStructSolver)    *solver,
                                               HYPRE_SStructDiagScale,
                                               HYPRE_SStructDiagScaleSetup,
                                              (HYPRE_SStructSolver)    *precond_solver));
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
 * HYPRE_SStructGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetlogging, HYPRE_SSTRUCTGMRESSETLOGGING)
                                                       (long int  *solver,
                                                        int       *logging,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetLogging( (HYPRE_SStructSolver) *solver,
                                                (int)                 *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmressetprintlevel, HYPRE_SSTRUCTGMRESSETPRINTLEVEL)
                                                       (long int  *solver,
                                                        int       *level,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetnumiterati, HYPRE_SSTRUCTGMRESGETNUMITERATI)
                                                       (long int  *solver,
                                                        int       *num_iterations,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetfinalrelat, HYPRE_SSTRUCTGMRESGETFINALRELAT)
                                                       (long int  *solver,
                                                        double    *norm,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructGMRESGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgmresgetresidual, HYPRE_SSTRUCTGMRESGETRESIDUAL)
                                                       (long int  *solver,
                                                        long int  *residual,
                                                        int       *ierr)
{
   *ierr = (int) (HYPRE_SStructGMRESGetResidual( (HYPRE_SStructSolver) *solver,
                                                 (void *)              *residual ) );
}
