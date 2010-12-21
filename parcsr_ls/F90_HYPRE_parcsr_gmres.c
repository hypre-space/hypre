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
 * HYPRE_ParCSRGMRES Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmrescreate, HYPRE_PARCSRGMRESCREATE)( hypre_F90_Comm *comm,
                                          hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmresdestroy, HYPRE_PARCSRGMRESDESTROY)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmressetup, HYPRE_PARCSRGMRESSETUP)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrgmressolve, HYPRE_PARCSRGMRESSOLVE)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetkdim, HYPRE_PARCSRGMRESSETKDIM)( hypre_F90_Obj *solver,
                                           HYPRE_Int      *kdim,
                                           HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetKDim( (HYPRE_Solver) *solver,
                                             (HYPRE_Int)          *kdim    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressettol, HYPRE_PARCSRGMRESSETTOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetabsolutetol, HYPRE_PARCSRGMRESSETABSOLUTETOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetAbsoluteTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetminiter, HYPRE_PARCSRGMRESSETMINITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *min_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetMinIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetmaxiter, HYPRE_PARCSRGMRESSETMAXITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *max_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetMaxIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetstopcrit, HYPRE_PARCSRGMRESSETSTOPCRIT)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *stop_crit,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetStopCrit( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprecond, HYPRE_PARCSRGMRESSETPRECOND)( hypre_F90_Obj *solver,
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
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRGMRESSetPrecond( (HYPRE_Solver) *solver,
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
 * HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetprecond, HYPRE_PARCSRGMRESGETPRECOND)( hypre_F90_Obj *solver,
                                              hypre_F90_Obj *precond_solver_ptr,
                                              HYPRE_Int      *ierr                )
{
    *ierr = (HYPRE_Int)
            ( HYPRE_ParCSRGMRESGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetlogging, HYPRE_PARCSRGMRESSETLOGGING)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *logging,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetLogging( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmressetprintlevel, HYPRE_PARCSRGMRESSETPRINTLEVEL)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *print_level,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESSetPrintLevel( (HYPRE_Solver) *solver,
                                                   (HYPRE_Int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetnumiteratio, HYPRE_PARCSRGMRESGETNUMITERATIO)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrgmresgetfinalrelati, HYPRE_PARCSRGMRESGETFINALRELATI)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
