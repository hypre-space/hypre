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
 * HYPRE_ParCSRBiCGSTAB Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabcreate, HYPRE_PARCSRBICGSTABCREATE)( HYPRE_Int      *comm,
                                          hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABCreate( (MPI_Comm)      *comm,
                                                (HYPRE_Solver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrbicgstabdestroy, HYPRE_PARCSRBICGSTABDESTROY)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrbicgstabsetup, HYPRE_PARCSRBICGSTABSETUP)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetup( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrbicgstabsolve, HYPRE_PARCSRBICGSTABSOLVE)( hypre_F90_Obj *solver,
                                         hypre_F90_Obj *A,
                                         hypre_F90_Obj *b,
                                         hypre_F90_Obj *x,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSolve( (HYPRE_Solver)       *solver,
                                           (HYPRE_ParCSRMatrix) *A,
                                           (HYPRE_ParVector)    *b,
                                           (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsettol, HYPRE_PARCSRBICGSTABSETTOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetTol( (HYPRE_Solver) *solver,
                                            (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetatol, HYPRE_PARCSRBICGSTABSETATOL)( hypre_F90_Obj *solver,
                                          double   *tol,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetAbsoluteTol( (HYPRE_Solver) *solver,
                                                       (double)       *tol     ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetminiter, HYPRE_PARCSRBICGSTABSETMINITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *min_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetMinIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetmaxiter, HYPRE_PARCSRBICGSTABSETMAXITER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *max_iter,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetMaxIter( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSeStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetstopcrit, HYPRE_PARCSRBICGSTABSETSTOP)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *stop_crit,
                                              HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetStopCrit( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetprecond, HYPRE_PARCSRBICGSTABSETPRECOND)( hypre_F90_Obj *solver,
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
              ( HYPRE_ParCSRBiCGSTABSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRDiagScale,
                                             HYPRE_ParCSRDiagScaleSetup,
                                             NULL                        ) );
   }
   else if (*precond_id == 2)
   {

   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetPrecond( (HYPRE_Solver) *solver,
                                                HYPRE_BoomerAMGSolve,
                                                HYPRE_BoomerAMGSetup,
                                                (void *)       *precond_solver ) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond( (HYPRE_Solver) *solver,
                                             HYPRE_ParCSRPilutSolve,
                                             HYPRE_ParCSRPilutSetup,
                                             (void *)       *precond_solver ) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (HYPRE_Int)
              ( HYPRE_ParCSRBiCGSTABSetPrecond( (HYPRE_Solver) *solver,
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
 * HYPRE_ParCSRBiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabgetprecond, HYPRE_PARCSRBICGSTABGETPRECOND)( hypre_F90_Obj *solver,
                                              hypre_F90_Obj *precond_solver_ptr,
                                              HYPRE_Int      *ierr                )
{
    *ierr = (HYPRE_Int)
            ( HYPRE_ParCSRBiCGSTABGetPrecond( (HYPRE_Solver)   *solver,
                                           (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetlogging, HYPRE_PARCSRBICGSTABSETLOGGING)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *logging,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetLogging( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabsetprintlev, HYPRE_PARCSRBICGSTABSETPRINTLEV)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *print_level,
                                              HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABSetPrintLevel( (HYPRE_Solver) *solver,
                                                      (HYPRE_Int)          *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetNumIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabgetnumiter, HYPRE_PARCSRBICGSTABGETNUMITER)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrbicgstabgetfinalrel, HYPRE_PARCSRBICGSTABGETFINALREL)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}
