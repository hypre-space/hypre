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
 * HYPRE_ParCSRFlexGMRES Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmrescreate, HYPRE_PARCSRFLEXGMRESCREATE)( HYPRE_Int      *comm,
                                          hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrflexgmresdestroy, HYPRE_PARCSRFLEXGMRESDESTROY)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrflexgmressetup, HYPRE_PARCSRFLEXGMRESSETUP)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrflexgmressolve, HYPRE_PARCSRFLEXGMRESSOLVE)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetkdim, HYPRE_PARCSRFLEXGMRESSETKDIM)( hypre_F90_Obj *solver,
                                           HYPRE_Int      *kdim,
                                           HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (HYPRE_Int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressettol, HYPRE_PARCSRFLEXGMRESSETTOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetabsolutetol, HYPRE_PARCSRFLEXGMRESSETABSOLUTETOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetAbsoluteTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetminiter, HYPRE_PARCSRFLEXGMRESSETMINITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *min_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetMinIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetmaxiter, HYPRE_PARCSRFLEXGMRESSETMAXITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *max_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetprecond, HYPRE_PARCSRFLEXGMRESSETPRECOND)( hypre_F90_Obj *solver,
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
              ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRFlexGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (HYPRE_Int)
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
hypre_F90_IFACE(hypre_parcsrflexgmresgetprecond, HYPRE_PARCSRFLEXGMRESGETPRECOND)( hypre_F90_Obj *solver,
                                              hypre_F90_Obj *precond_solver_ptr,
                                              HYPRE_Int      *ierr                )
{
    *ierr = (HYPRE_Int)
            ( HYPRE_ParCSRFlexGMRESGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetlogging, HYPRE_PARCSRFLEXGMRESSETLOGGING)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *logging,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmressetprintlevel, HYPRE_PARCSRFLEXGMRESSETPRINTLEVEL)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *print_level,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetnumiteratio, HYPRE_PARCSRFLEXGMRESGETNUMITERATIO)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrflexgmresgetfinalrelati, HYPRE_PARCSRFLEXGMRESGETFINALRELATI)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
