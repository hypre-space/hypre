/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * HYPRE_ParCSRPCG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgcreate)( int      *comm,
                                            long int *solver,
                                            int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRPCGCreate( (MPI_Comm)       *comm,
                                              (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgdestroy)( long int *solver,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsetup)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetup( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsolve)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSolve( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettol)( long int *solver,
                                        double   *tol,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetTol( (HYPRE_Solver) *solver,
                                          (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetmaxiter)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetMaxIter( (HYPRE_Solver) *solver,
                                              (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettwonorm)( long int *solver,
                                            int      *two_norm,
                                            int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetTwoNorm( (HYPRE_Solver) *solver,
                                              (int)          *two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetrelchange)( long int *solver,
                                              int      *rel_change,
                                              int      *ierr        )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetRelChange( (HYPRE_Solver) *solver,
                                                (int)          *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprecond)( long int *solver,
                                            int      *precond_id,
                                            long int *precond_solver,
                                            int      *ierr            )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 2 - set up an amg preconditioner
    * 7 - set up a pilut preconditioner
    * 8 - set up a ds preconditioner
    * 9 - do not set up a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 2)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_BoomerAMGSolve,
                               HYPRE_BoomerAMGSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParCSRPilutSolve,
                               HYPRE_ParCSRPilutSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParCSRDiagScale,
                               HYPRE_ParCSRDiagScaleSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetprecond)( long int *solver,
                                            long int *precond_solver_ptr,
                                            int      *ierr                )
{
    *ierr = (int)
            ( HYPRE_ParCSRPCGGetPrecond( (HYPRE_Solver)   *solver,
                                         (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetlogging)( long int *solver,
                                            int      *logging,
                                            int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRPCGSetLogging( (HYPRE_Solver) *solver,
                                              (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetnumiterations)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRPCGGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetfinalrelative)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscalesetup)( long int *solver,
                                             long int *A,
                                             long int *y,
                                             long int *x,
                                             int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRDiagScaleSetup( (HYPRE_Solver)       *solver,
                                               (HYPRE_ParCSRMatrix) *A,
                                               (HYPRE_ParVector)    *y,
                                               (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscale)( long int *solver,
                                        long int *HA,
                                        long int *Hy,
                                        long int *Hx,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRDiagScale( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *HA,
                                          (HYPRE_ParVector)    *Hy,
                                          (HYPRE_ParVector)    *Hx      ) );
}

