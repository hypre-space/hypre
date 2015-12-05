/*BHEADER**********************************************************************
 * (c) 2000   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision: 2.2 $
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructBiCGSTAB interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabcreate, HYPRE_SSTRUCTBICGSTABCREATE)
                                                          (int      *comm,
                                                           long int *solver,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABCreate( (MPI_Comm)              *comm,
                                               (HYPRE_SStructSolver *)  solver )) ;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabdestroy, HYPRE_SSTRUCTBICGSTABDESTROY)
                                                          (long int *solver,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetup, HYPRE_SSTRUCTBICGSTABSETUP)
                                                          (long int *solver,
                                                           long int *A,
                                                           long int *b,
                                                           long int *x,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetup( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_SStructMatrix) *A,
                                              (HYPRE_SStructVector) *b,
                                              (HYPRE_SStructVector) *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsolve, HYPRE_SSTRUCTBICGSTABSOLVE)
                                                          (long int *solver,
                                                           long int *A,
                                                           long int *b,
                                                           long int *x,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSolve( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_SStructMatrix) *A,
                                              (HYPRE_SStructVector) *b,
                                              (HYPRE_SStructVector) *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsettol, HYPRE_SSTRUCTBICGSTABSETTOL)
                                                          (long int *solver,
                                                           double   *tol,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetTol( (HYPRE_SStructSolver) *solver,
                                               (double)              *tol ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetminiter, HYPRE_SSTRUCTBICGSTABSETMINITER)
                                                          (long int *solver,
                                                           int      *min_iter,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetMinIter( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *min_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetmaxiter, HYPRE_SSTRUCTBICGSTABSETMAXITER)
                                                          (long int *solver,
                                                           int      *max_iter,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *max_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetstopcri, HYPRE_SSTRUCTBICGSTABSETSTOPCRI)
                                                          (long int *solver,
                                                           int      *stop_crit,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetStopCrit( (HYPRE_SStructSolver) *solver,
                                                    (int)                 *stop_crit ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprecond, HYPRE_SSTRUCTBICGSTABSETPRECOND)
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
               (HYPRE_SStructBiCGSTABSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSplitSolve,
                                             HYPRE_SStructSplitSetup,
                                            (HYPRE_SStructSolver *)    precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (int)
               (HYPRE_SStructBiCGSTABSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSysPFMGSolve,
                                             HYPRE_SStructSysPFMGSetup,
                                            (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (int)
               (HYPRE_SStructBiCGSTABSetPrecond( (HYPRE_SStructSolver)    *solver,
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
 * HYPRE_SStructBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetlogging, HYPRE_SSTRUCTBICGSTABSETLOGGING)
                                                          (long int *solver,
                                                           int      *logging,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetLogging( (HYPRE_SStructSolver) *solver,
                                                   (int)                 *logging ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprintle, HYPRE_SSTRUCTBICGSTABSETPRINTLE)
                                                          (long int *solver,
                                                           int      *print_level,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                      (int)                 *print_level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetnumiter, HYPRE_SSTRUCTBICGSTABGETNUMITER)
                                                          (long int *solver,
                                                           int      *num_iterations,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                         (int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetfinalre, HYPRE_SSTRUCTBICGSTABGETFINALRE)
                                                          (long int *solver,
                                                           double   *norm,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                     (double *)             norm ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetresidua, HYPRE_SSTRUCTBICGSTABGETRESIDUA)
                                                          (long int *solver,
                                                           long int *residual,
                                                           int      *ierr)
{
   *ierr = (int) (HYPRE_SStructBiCGSTABGetResidual( (HYPRE_SStructSolver) *solver,
                                                    (void *)              *residual));
}
