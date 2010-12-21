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
 * HYPRE_SStructBiCGSTAB interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabcreate, HYPRE_SSTRUCTBICGSTABCREATE)
                                                          (hypre_F90_Comm *comm,
                                                           hypre_F90_Obj *solver,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABCreate( (MPI_Comm)              *comm,
                                               (HYPRE_SStructSolver *)  solver )) ;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabdestroy, HYPRE_SSTRUCTBICGSTABDESTROY)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetup, HYPRE_SSTRUCTBICGSTABSETUP)
                                                          (hypre_F90_Obj *solver,
                                                           hypre_F90_Obj *A,
                                                           hypre_F90_Obj *b,
                                                           hypre_F90_Obj *x,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetup( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_SStructMatrix) *A,
                                              (HYPRE_SStructVector) *b,
                                              (HYPRE_SStructVector) *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsolve, HYPRE_SSTRUCTBICGSTABSOLVE)
                                                          (hypre_F90_Obj *solver,
                                                           hypre_F90_Obj *A,
                                                           hypre_F90_Obj *b,
                                                           hypre_F90_Obj *x,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSolve( (HYPRE_SStructSolver) *solver,
                                              (HYPRE_SStructMatrix) *A,
                                              (HYPRE_SStructVector) *b,
                                              (HYPRE_SStructVector) *x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsettol, HYPRE_SSTRUCTBICGSTABSETTOL)
                                                          (hypre_F90_Obj *solver,
                                                           double   *tol,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetTol( (HYPRE_SStructSolver) *solver,
                                               (double)              *tol ));
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetAnsoluteTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetabsolutetol, HYPRE_SSTRUCTBICGSTABSETABSOLUTETOL)
                                                          (hypre_F90_Obj *solver,
                                                           double   *tol,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetAbsoluteTol( (HYPRE_SStructSolver) *solver,
                                               (double)              *tol ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetminiter, HYPRE_SSTRUCTBICGSTABSETMINITER)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *min_iter,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetMinIter( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *min_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetmaxiter, HYPRE_SSTRUCTBICGSTABSETMAXITER)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *max_iter,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *max_iter ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetstopcri, HYPRE_SSTRUCTBICGSTABSETSTOPCRI)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *stop_crit,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetStopCrit( (HYPRE_SStructSolver) *solver,
                                                    (HYPRE_Int)                 *stop_crit ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprecond, HYPRE_SSTRUCTBICGSTABSETPRECOND)
                                                     (hypre_F90_Obj *solver,
                                                      HYPRE_Int      *precond_id,
                                                      hypre_F90_Obj *precond_solver,
                                                      HYPRE_Int      *ierr)
/*------------------------------------------
 *    precond_id flags mean:
 *    2 - setup a split-solver preconditioner
 *    3 - setup a syspfmg preconditioner
 *    8 - setup a DiagScale preconditioner
 *    9 - no preconditioner setup
 *----------------------------------------*/

{
   if(*precond_id == 2)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructBiCGSTABSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSplitSolve,
                                             HYPRE_SStructSplitSetup,
                                            (HYPRE_SStructSolver *)    precond_solver));
      }

   else if(*precond_id == 3)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructBiCGSTABSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructSysPFMGSolve,
                                             HYPRE_SStructSysPFMGSetup,
                                            (HYPRE_SStructSolver)    *precond_solver));
      }

   else if(*precond_id == 8)
      {
       *ierr = (HYPRE_Int)
               (HYPRE_SStructBiCGSTABSetPrecond( (HYPRE_SStructSolver)    *solver,
                                             HYPRE_SStructDiagScale,
                                             HYPRE_SStructDiagScaleSetup,
                                            (HYPRE_SStructSolver)    *precond_solver));
      }
   else if(*precond_id == 9)
      {
       *ierr = 0;
      }

   else
      {
       *ierr = -1;
      }

}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetlogging, HYPRE_SSTRUCTBICGSTABSETLOGGING)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *logging,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetLogging( (HYPRE_SStructSolver) *solver,
                                                   (HYPRE_Int)                 *logging ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabsetprintle, HYPRE_SSTRUCTBICGSTABSETPRINTLE)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *print_level,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                      (HYPRE_Int)                 *print_level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetnumiter, HYPRE_SSTRUCTBICGSTABGETNUMITER)
                                                          (hypre_F90_Obj *solver,
                                                           HYPRE_Int      *num_iterations,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                         (HYPRE_Int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetfinalre, HYPRE_SSTRUCTBICGSTABGETFINALRE)
                                                          (hypre_F90_Obj *solver,
                                                           double   *norm,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                     (double *)             norm ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructbicgstabgetresidua, HYPRE_SSTRUCTBICGSTABGETRESIDUA)
                                                          (hypre_F90_Obj *solver,
                                                           hypre_F90_Obj *residual,
                                                           HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructBiCGSTABGetResidual( (HYPRE_SStructSolver) *solver,
                                                    (void *)              *residual));
}
