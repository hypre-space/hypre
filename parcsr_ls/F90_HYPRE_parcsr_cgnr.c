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
hypre_F90_IFACE(hypre_parcsrcgnrcreate, HYPRE_PARCSRCGNRCREATE)( int      *comm,
                                             long int *solver,
                                             int      *ierr    )

{
   *ierr = (int) ( HYPRE_ParCSRCGNRCreate( (MPI_Comm)       *comm,
                                               (HYPRE_Solver *)  solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrdestroy, HYPRE_PARCSRCGNRDESTROY)( long int *solver,
                                           int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRDestroy( (HYPRE_Solver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parcsrcgnrsetup, HYPRE_PARCSRCGNRSETUP)( long int *solver,
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
hypre_F90_IFACE(hypre_parcsrcgnrsolve, HYPRE_PARCSRCGNRSOLVE)( long int *solver,
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
hypre_F90_IFACE(hypre_parcsrcgnrsettol, HYPRE_PARCSRCGNRSETTOL)( long int *solver,
                                         double   *tol,
                                         int      *ierr    )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetTol( (HYPRE_Solver) *solver,
                                           (double)       *tol     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetminiter, HYPRE_PARCSRCGNRSETMINITER)( long int *solver,
                                             int      *min_iter,
                                             int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetMinIter( (HYPRE_Solver) *solver,
                                               (int)          *min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetmaxiter, HYPRE_PARCSRCGNRSETMAXITER)( long int *solver,
                                             int      *max_iter,
                                             int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetMaxIter( (HYPRE_Solver) *solver,
                                               (int)          *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetstopcrit, HYPRE_PARCSRCGNRSETSTOPCRIT)
                                           ( long int *solver,
                                             int      *stop_crit,
                                             int      *ierr      )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetStopCrit( (HYPRE_Solver) *solver,
                                               (int)          *stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetprecond, HYPRE_PARCSRCGNRSETPRECOND)( long int *solver,
                                             int      *precond_id,
                                             long int *precond_solver,
                                             int      *ierr            )
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
      *ierr = (int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
                                                  HYPRE_ParCSRDiagScale,
                                                  HYPRE_ParCSRDiagScale,
                                                  HYPRE_ParCSRDiagScaleSetup,
                                                  NULL                        ) );
   }
   else if (*precond_id == 2)
   {
      *ierr = (int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
                                                  HYPRE_BoomerAMGSolve,
                                                  HYPRE_BoomerAMGSolve,
                                                  HYPRE_BoomerAMGSetup,
                                                  (void *)       *precond_solver ) );
   }
   if (*precond_id == 3)
   {
      *ierr = (int) ( HYPRE_ParCSRCGNRSetPrecond( (HYPRE_Solver) *solver,
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
hypre_F90_IFACE(hypre_parcsrcgnrgetprecond, HYPRE_PARCSRCGNRGETPRECOND)( long int *solver,
                                             long int *precond_solver_ptr,
                                             int      *ierr                 )
{
    *ierr = (int)
            ( HYPRE_ParCSRCGNRGetPrecond( (HYPRE_Solver)   *solver,
                                          (HYPRE_Solver *)  precond_solver_ptr ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrsetlogging, HYPRE_PARCSRCGNRSETLOGGING)( long int *solver,
                                             int      *logging,
                                             int      *ierr     )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRSetLogging( (HYPRE_Solver) *solver,
                                               (int)          *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetNumIteration
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetnumiteration, HYPRE_PARCSRCGNRGETNUMITERATION)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr            )
{
   *ierr = (int) ( HYPRE_ParCSRCGNRGetNumIterations( (HYPRE_Solver) *solver,
                                                     (int *)         num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parcsrcgnrgetfinalrelativ, HYPRE_PARCSRCGNRGETFINALRELATIV)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr    )
{
   *ierr = (int)
           ( HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( (HYPRE_Solver) *solver,
                                                           (double *)      norm     ) );
}
