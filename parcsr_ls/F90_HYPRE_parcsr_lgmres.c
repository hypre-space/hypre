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
hypre_F90_IFACE(hypre_parcsrlgmrescreate, HYPRE_PARCSRLGMRESCREATE)( HYPRE_Int      *comm,
                                          hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrlgmresdestroy, HYPRE_PARCSRLGMRESDESTROY)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrlgmressetup, HYPRE_PARCSRLGMRESSETUP)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrlgmressolve, HYPRE_PARCSRLGMRESSOLVE)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetkdim, HYPRE_PARCSRLGMRESSETKDIM)( hypre_F90_Obj *solver,
                                           HYPRE_Int      *kdim,
                                           HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (HYPRE_Int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressettol, HYPRE_PARCSRLGMRESSETTOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetabsolutetol, HYPRE_PARCSRLGMRESSETABSOLUTETOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetAbsoluteTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetminiter, HYPRE_PARCSRLGMRESSETMINITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *min_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetMinIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetmaxiter, HYPRE_PARCSRLGMRESSETMAXITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *max_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *max_iter ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetprecond, HYPRE_PARCSRLGMRESSETPRECOND)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *precond_id,
                                              hypre_F90_Obj *precond_solver,
                                              HYPRE_Int      *ierr          )
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
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRLGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (HYPRE_Int)
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
hypre_F90_IFACE(hypre_parcsrlgmresgetprecond, HYPRE_PARCSRLGMRESGETPRECOND)( hypre_F90_Obj *solver,
                                              hypre_F90_Obj *precond_solver_ptr,
                                              HYPRE_Int      *ierr                )
{
    *ierr = (HYPRE_Int)
            ( HYPRE_ParCSRLGMRESGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetlogging, HYPRE_PARCSRLGMRESSETLOGGING)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *logging,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmressetprintlevel, HYPRE_PARCSRLGMRESSETPRINTLEVEL)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *print_level,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetnumiteratio, HYPRE_PARCSRLGMRESGETNUMITERATIO)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrlgmresgetfinalrelati, HYPRE_PARCSRLGMRESGETFINALRELATI)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
