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
 * HYPRE_ParCSRGMRESInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresinitialize)( int      *comm,
                                              long int *solver,
                                              int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRGMRESInitialize( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESFinalize
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmresfinalize)( long int *solver,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESFinalize( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmressetup)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmressolve)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetkdim)( long int *solver,
                                           int      *kdim,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressettol)( long int *solver,
                                          double   *tol,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetmaxiter)( long int *solver,
                                              int      *max_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprecond)( long int *solver,
                                              int      *precond_id,
                                              long int *precond_data,
                                              int      *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 2 - set up an amg preconditioner
    * 8 - set up a ds preconditioner
    * 9 - do not set up a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 2)
   {

   *ierr = (int) ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_ParAMGSolve,
                                                HYPRE_ParAMGSetup,
                                                (void *)        precond_data ) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (int)
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             (void *)        precond_data ) );
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
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetlogging)( long int *solver,
                                              int      *logging,
                                              int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetnumiter)( long int *solver,
                                              int      *num_iterations,
                                              int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetfinalrel)( long int *solver,
                                               double   *norm,
                                               int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
