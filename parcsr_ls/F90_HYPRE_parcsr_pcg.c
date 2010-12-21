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
 * HYPRE_ParCSRPCG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgcreate, HYPRE_PARCSRPCGCREATE)( hypre_F90_Comm *comm,
                                            hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGCreate( (MPI_Comm)       *comm,
                                              (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgdestroy, HYPRE_PARCSRPCGDESTROY)( hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsetup, HYPRE_PARCSRPCGSETUP)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetup( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrpcgsolve, HYPRE_PARCSRPCGSOLVE)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSolve( (HYPRE_Solver)       *solver,
                                         (HYPRE_ParCSRMatrix) *A,
                                         (HYPRE_ParVector)    *b,
                                         (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettol, HYPRE_PARCSRPCGSETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetTol( (HYPRE_Solver) *solver,
                                          (double)       *tol     ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetAbsoluteTol
 *-------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetatol, HYPRE_PARCSRPCGSETATOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetAbsoluteTol( (HYPRE_Solver) *solver,
                                                  (double)       *tol     ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetmaxiter, HYPRE_PARCSRPCGSETMAXITER)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetMaxIter( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetstopcrit, HYPRE_PARCSRPCGSETSTOPCRIT)
                                          ( hypre_F90_Obj *solver,
                                            HYPRE_Int      *stop_crit,
                                            HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetStopCrit( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsettwonorm, HYPRE_PARCSRPCGSETTWONORM)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *two_norm,
                                            HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetTwoNorm( (HYPRE_Solver) *solver,
                                              (HYPRE_Int)          *two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetrelchange, HYPRE_PARCSRPCGSETRELCHANGE)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *rel_change,
                                              HYPRE_Int      *ierr        )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetRelChange( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprecond, HYPRE_PARCSRPCGSETPRECOND)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *precond_id,
                                            hypre_F90_Obj *precond_solver,
                                            HYPRE_Int      *ierr            )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - do not set up a preconditioner
    * 1 - set up a ds preconditioner
    * 2 - set up an amg preconditioner
    * 3 - set up a pilut preconditioner
    * 4 - set up a ParaSails preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParCSRDiagScale,
                               HYPRE_ParCSRDiagScaleSetup,
                               NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_BoomerAMGSolve,
                               HYPRE_BoomerAMGSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 3)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParCSRPilutSolve,
                               HYPRE_ParCSRPilutSetup,
                               (void *)       *precond_solver) );
   }
   else if (*precond_id == 4)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetPrecond(
                               (HYPRE_Solver) *solver,
                               HYPRE_ParaSailsSolve,
                               HYPRE_ParaSailsSetup,
                               (void *)       *precond_solver) );
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
hypre_F90_IFACE(hypre_parcsrpcggetprecond, HYPRE_PARCSRPCGGETPRECOND)( hypre_F90_Obj *solver,
                                            hypre_F90_Obj *precond_solver_ptr,
                                            HYPRE_Int      *ierr                )
{
    *ierr = (HYPRE_Int)
            ( HYPRE_ParCSRPCGGetPrecond( (HYPRE_Solver)   *solver,
                                         (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetprintlevel, HYPRE_PARCSRPCGSETPRINTLEVEL)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *level,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetPrintLevel( (HYPRE_Solver) *solver,
                                                 (HYPRE_Int)       *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcgsetlogging, HYPRE_PARCSRPCGSETLOGGING)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *level,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGSetLogging( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)       *level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetnumiterations, HYPRE_PARCSRPCGGETNUMITERATIONS)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGGetNumIterations(
                            (HYPRE_Solver) *solver,
                            (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrpcggetfinalrelative, HYPRE_PARCSRPCGGETFINALRELATIVE)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRPCGGetFinalRelativeResidualNorm(
                            (HYPRE_Solver) *solver,
                            (double *)      norm    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscalesetup, HYPRE_PARCSRDIAGSCALESETUP)( hypre_F90_Obj *solver,
                                             hypre_F90_Obj *A,
                                             hypre_F90_Obj *y,
                                             hypre_F90_Obj *x,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRDiagScaleSetup( (HYPRE_Solver)       *solver,
                                               (HYPRE_ParCSRMatrix) *A,
                                               (HYPRE_ParVector)    *y,
                                               (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrdiagscale, HYPRE_PARCSRDIAGSCALE)( hypre_F90_Obj *solver,
                                        hypre_F90_Obj *HA,
                                        hypre_F90_Obj *Hy,
                                        hypre_F90_Obj *Hx,
                                        HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRDiagScale( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *HA,
                                          (HYPRE_ParVector)    *Hy,
                                          (HYPRE_ParVector)    *Hx      ) );
}
