/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.2 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_ParCSRFlexGMRES Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmrescreate, HYPRE_PARCSRFLEXGMRESCREATE)( int      *comm,
                                          long int *solver,
                                          int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrflexgmresdestroy, HYPRE_PARCSRFLEXGMRESDESTROY)( long int *solver,
                                            int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrflexgmressetup, HYPRE_PARCSRFLEXGMRESSETUP)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrflexgmressolve, HYPRE_PARCSRFLEXGMRESSOLVE)( long int *solver,
                                         long int *A,
                                         long int *b,
                                         long int *x,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetkdim, HYPRE_PARCSRFLEXGMRESSETKDIM)( long int *solver,
                                           int      *kdim,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressettol, HYPRE_PARCSRFLEXGMRESSETTOL)( long int *solver,
                                          double   *tol,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetabsolutetol, HYPRE_PARCSRFLEXGMRESSETABSOLUTETOL)( long int *solver,
                                          double   *tol,
                                          int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetAbsoluteTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetminiter, HYPRE_PARCSRFLEXGMRESSETMINITER)( long int *solver,
                                              int      *min_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetMinIter( (HYPRE_Solver) *solver,
                                                (int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetmaxiter, HYPRE_PARCSRFLEXGMRESSETMAXITER)( long int *solver,
                                              int      *max_iter,
                                              int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetprecond, HYPRE_PARCSRFLEXGMRESSETPRECOND)( long int *solver,
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
              ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (int)
              ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (int)
              ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
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
 * HYPRE_ParCSRFlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetprecond, HYPRE_PARCSRFLEXGMRESGETPRECOND)( long int *solver,
                                              long int *precond_solver_ptr,
                                              int      *ierr                )
{
    *ierr = (int)
            ( HYPRE_ParCSRFlexGMRESGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetlogging, HYPRE_PARCSRFLEXGMRESSETLOGGING)( long int *solver,
                                              int      *logging,
                                              int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetprintlevel, HYPRE_PARCSRFLEXGMRESSETPRINTLEVEL)
                                            ( long int *solver,
                                              int      *print_level,
                                              int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetnumiteratio, HYPRE_PARCSRFLEXGMRESGETNUMITERATIO)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetfinalrelati, HYPRE_PARCSRFLEXGMRESGETFINALRELATI)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
