/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrinitialize)( int      *comm,
                                             long int *solver,
                                             int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRCGNRInitialize( (MPI_Comm)       *comm,
                                               (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRFinalize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrfinalize)( long int *solver,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRFinalize( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsetup)( long int *solver,
                                        long int *A,
                                        long int *b,
                                        long int *x,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetup( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b,
                                          (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsolve)( long int *solver,
                                        long int *A,
                                        long int *b,
                                        long int *x,
                                        int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSolve( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b,
                                          (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsettol)( long int *solver,
                                         double   *tol,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetTol( (HYPRE_Solver) *solver,
                                           (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetmaxiter)( long int *solver,
                                             int      *max_iter,
                                             int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetMaxIter( (HYPRE_Solver) *solver,
                                               (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetprecond)( long int *solver,
                                             int      *precond_id,
                                             long int *precond_data,
                                             int      *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 8 - set up a ds preconditioner
    * 9 - do not set up a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 8)
   {
      *ierr = (int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
                                                  HYPRE_ParCSRDiagScale,
                                                  HYPRE_ParCSRDiagScale,
                                                  HYPRE_ParCSRDiagScaleSetup,
                                                  (void *)        precond_data  ) );
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
 * HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetlogging)( long int *solver,
                                             int      *logging,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetLogging( (HYPRE_Solver) *solver,
                                               (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetnumiter)( long int *solver,
                                             int      *num_iterations,
                                             int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRGetNumIterations( (HYPRE_Solver) *solver,
                                                     (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetFinalRelative
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetfinalrel)( long int *solver,
                                              double   *norm,
                                              int      *ierr    )
{
   *ierr = (int)
           ( HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                           (double *)      norm     ) );
}
