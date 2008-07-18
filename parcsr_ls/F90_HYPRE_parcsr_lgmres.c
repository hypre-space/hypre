/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_ParCSRLGMRES Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmrescreate, HYPRE_PARCSRLGMRESCREATE)( int      *comm,
                                          long int *solver,
                                          int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRLGMRESCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrlgmresdestroy, HYPRE_PARCSRLGMRESDESTROY)( long int *solver,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrlgmressetup, HYPRE_PARCSRLGMRESSETUP)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrlgmressolve, HYPRE_PARCSRLGMRESSOLVE)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetkdim, HYPRE_PARCSRLGMRESSETKDIM)( long int *solver,
                                           int      *kdim,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressettol, HYPRE_PARCSRLGMRESSETTOL)( long int *solver,
                                          double   *tol,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetabsolutetol, HYPRE_PARCSRLGMRESSETABSOLUTETOL)( long int *solver,
                                          double   *tol,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetAbsoluteTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetminiter, HYPRE_PARCSRLGMRESSETMINITER)( long int *solver,
                                              int      *min_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetMinIter( (HYPRE_Solver) *solver,
                                                (int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetmaxiter, HYPRE_PARCSRLGMRESSETMAXITER)( long int *solver,
                                              int      *max_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (int)          *max_iter ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetprecond, HYPRE_PARCSRLGMRESSETPRECOND)( long int *solver,
                                              int      *precond_id,
                                              long int *precond_solver,
                                              int      *ierr          )
{
   /*------------------------------------------------------------
    * The precond_id flags mean :
    *  0 - no preconditioner
    *  1 - set up a ds preconditioner
    *  2 - set up an amg preconditioner
    *  3 - set up a pilut preconditioner
    *  4 - set up a parasails preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (int)
              ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (int) ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (int)
              ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (int)
              ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRParaSailsSolve,
                                             HYPRE_ParCSRParaSailsSetup,
                                             (void *)       *precond_solver ) );
   }
     else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetprecond, HYPRE_PARCSRLGMRESGETPRECOND)( long int *solver,
                                              long int *precond_solver_ptr,
                                              int      *ierr                )
{
    *ierr = (int)
            ( HYPRE_ParCSRLGMRESGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetlogging, HYPRE_PARCSRLGMRESSETLOGGING)( long int *solver,
                                              int      *logging,
                                              int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetprintlevel, HYPRE_PARCSRLGMRESSETPRINTLEVEL)
                                            ( long int *solver,
                                              int      *print_level,
                                              int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetnumiteratio, HYPRE_PARCSRLGMRESGETNUMITERATIO)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetfinalrelati, HYPRE_PARCSRLGMRESGETFINALRELATI)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
