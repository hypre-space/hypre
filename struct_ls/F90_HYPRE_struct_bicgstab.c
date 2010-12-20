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


#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabcreate, HYPRE_STRUCTBICGSTABCREATE)( HYPRE_Int      *comm,
                                            hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr   )

{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructBiCGSTABCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbicgstabdestroy, HYPRE_STRUCTBICGSTABDESTROY)( hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructBiCGSTABDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbicgstabsetup, HYPRE_STRUCTBICGSTABSETUP)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructBiCGSTABSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structbicgstabsolve, HYPRE_STRUCTBICGSTABSOLVE)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructBiCGSTABSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsettol, HYPRE_STRUCTBICGSTABSETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructBiCGSTABSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetmaxiter, HYPRE_STRUCTBICGSTABSETMAXITER)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructBiCGSTABSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetprecond, HYPRE_STRUCTBICGSTABSETPRECOND)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *precond_id,
                                            hypre_F90_Obj *precond_solver,
                                            HYPRE_Int      *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructBiCGSTABSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructBiCGSTABSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructBiCGSTABSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructDiagScale,
                                      HYPRE_StructDiagScaleSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 9)
   {
      *ierr = 0;
   }
   else
   {
      *ierr = -1;
   }
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetlogging, HYPRE_STRUCTBICGSTABSETLOGGING)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *logging,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructBiCGSTABSetLogging( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabsetprintlev, HYPRE_STRUCTBICGSTABSETPRINTLEV)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *print_level,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructBiCGSTABSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetnumitera, HYPRE_STRUCTBICGSTABGETNUMITERA)
                                                ( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructBiCGSTABGetNumIterations( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetresidual, HYPRE_STRUCTBICGSTABGETRESIDUAL)
                                                ( hypre_F90_Obj *solver,
                                                  void *residual,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructBiCGSTABGetResidual( (HYPRE_StructSolver) *solver,
                                              (void *)          residual ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structbicgstabgetfinalrel, HYPRE_STRUCTBICGSTABGETFINALREL)
                                                ( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm( (HYPRE_StructSolver) *solver,
                                                    (double *)           norm ) );
}
