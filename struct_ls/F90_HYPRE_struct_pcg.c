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
 * HYPRE_StructPCGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgcreate, HYPRE_STRUCTPCGCREATE)( hypre_F90_Comm *comm,
                                            hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr   )

{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpcgdestroy, HYPRE_STRUCTPCGDESTROY)( hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPCGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpcgsetup, HYPRE_STRUCTPCGSETUP)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPCGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpcgsolve, HYPRE_STRUCTPCGSOLVE)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPCGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsettol, HYPRE_STRUCTPCGSETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPCGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetmaxiter, HYPRE_STRUCTPCGSETMAXITER)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsettwonorm, HYPRE_STRUCTPCGSETTWONORM)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *two_norm,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGSetTwoNorm( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetrelchange, HYPRE_STRUCTPCGSETRELCHANGE)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *rel_change,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGSetRelChange( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetprecond, HYPRE_STRUCTPCGSETPRECOND)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *precond_id,
                                            hypre_F90_Obj *precond_solver,
                                            HYPRE_Int      *ierr           )
{

   /*------------------------------------------------------------
    * The precond_id flags mean :
    * 0 - setup a smg preconditioner
    * 1 - setup a pfmg preconditioner
    * 7 - setup a jacobi preconditioner
    * 8 - setup a ds preconditioner
    * 9 - dont setup a preconditioner
    *------------------------------------------------------------*/

   if (*precond_id == 0)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructPCGSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructPCGSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 7)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructPCGSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructJacobiSolve,
                                      HYPRE_StructJacobiSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructPCGSetPrecond( (HYPRE_StructSolver) *solver,
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
 * HYPRE_StructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetlogging, HYPRE_STRUCTPCGSETLOGGING)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *logging,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGSetLogging( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcgsetprintlevel, HYPRE_STRUCTPCGSETPRINTLEVEL)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *print_level,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcggetnumiterations, HYPRE_STRUCTPCGGETNUMITERATIONS)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpcggetfinalrelative, HYPRE_STRUCTPCGGETFINALRELATIVE)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPCGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScaleSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structdiagscalesetup, HYPRE_STRUCTDIAGSCALESETUP)( hypre_F90_Obj *solver,
                                             hypre_F90_Obj *A,
                                             hypre_F90_Obj *y,
                                             hypre_F90_Obj *x,
                                             HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructDiagScaleSetup( (HYPRE_StructSolver) *solver,
                                               (HYPRE_StructMatrix) *A,
                                               (HYPRE_StructVector) *y,
                                               (HYPRE_StructVector) *x     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructDiagScale
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structdiagscale, HYPRE_STRUCTDIAGSCALE)( hypre_F90_Obj *solver,
                                        hypre_F90_Obj *HA,
                                        hypre_F90_Obj *Hy,
                                        hypre_F90_Obj *Hx,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructDiagScale( (HYPRE_StructSolver) *solver,
                                          (HYPRE_StructMatrix) *HA,
                                          (HYPRE_StructVector) *Hy,
                                          (HYPRE_StructVector) *Hx     ) );
}
