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
 * HYPRE_ParCSRCGNR Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrcreate, HYPRE_PARCSRCGNRCREATE)( hypre_F90_Comm *comm,
                                             hypre_F90_Obj *solver,
                                             HYPRE_Int      *ierr    )

{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRCreate( (MPI_Comm)       *comm,
                                               (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrdestroy, HYPRE_PARCSRCGNRDESTROY)( hypre_F90_Obj *solver,
                                           HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsetup, HYPRE_PARCSRCGNRSETUP)( hypre_F90_Obj *solver,
                                        hypre_F90_Obj *A,
                                        hypre_F90_Obj *b,
                                        hypre_F90_Obj *x,
                                        HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetup( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b,
                                          (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsolve, HYPRE_PARCSRCGNRSOLVE)( hypre_F90_Obj *solver,
                                        hypre_F90_Obj *A,
                                        hypre_F90_Obj *b,
                                        hypre_F90_Obj *x,
                                        HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSolve( (HYPRE_Solver)       *solver,
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b,
                                          (HYPRE_ParVector)    *x       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsettol, HYPRE_PARCSRCGNRSETTOL)( hypre_F90_Obj *solver,
                                         double   *tol,
                                         HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetTol( (HYPRE_Solver) *solver,
                                           (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetminiter, HYPRE_PARCSRCGNRSETMINITER)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *min_iter,
                                             HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetMinIter( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetmaxiter, HYPRE_PARCSRCGNRSETMAXITER)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *max_iter,
                                             HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetMaxIter( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetstopcrit, HYPRE_PARCSRCGNRSETSTOPCRIT)
                                           ( hypre_F90_Obj *solver,
                                             HYPRE_Int      *stop_crit,
                                             HYPRE_Int      *ierr      )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetStopCrit( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetprecond, HYPRE_PARCSRCGNRSETPRECOND)( hypre_F90_Obj *solver,
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
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = 0;
   }
   else if (*precond_id == 1)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
                                                  HYPRE_ParCSRDiagScale,
                                                  HYPRE_ParCSRDiagScale,
                                                  HYPRE_ParCSRDiagScaleSetup,
                                                  NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
                                                  HYPRE_BoomerAMGSolve,
                                                  HYPRE_BoomerAMGSolve,
                                                  HYPRE_BoomerAMGSetup,
                                                  (void *)       *precond_solver ) );
   }
   if (*precond_id == 3)
   {
      *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
                                                  HYPRE_ParCSRPilutSolve,
                                                  HYPRE_ParCSRPilutSolve,
                                                  HYPRE_ParCSRPilutSetup,
                                                  (void *)       *precond_solver ) );
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetprecond, HYPRE_PARCSRCGNRGETPRECOND)( hypre_F90_Obj *solver,
                                             hypre_F90_Obj *precond_solver_ptr,
                                             HYPRE_Int      *ierr                 )
{
    *ierr = (HYPRE_Int)
            ( HYPRE_ParCSRCGNRGetPrecond( (HYPRE_Solver)   *solver,
                                          (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetlogging, HYPRE_PARCSRCGNRSETLOGGING)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *logging,
                                             HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRSetLogging( (HYPRE_Solver) *solver,
                                               (HYPRE_Int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetNumIteration
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetnumiteration, HYPRE_PARCSRCGNRGETNUMITERATION)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr            )
{
   *ierr = (HYPRE_Int) ( HYPRE_ParCSRCGNRGetNumIterations( (HYPRE_Solver) *solver,
                                                     (HYPRE_Int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetfinalrelativ, HYPRE_PARCSRCGNRGETFINALRELATIV)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
           ( HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                           (double *)      norm     ) );
}
