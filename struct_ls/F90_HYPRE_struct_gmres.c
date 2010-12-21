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
 * HYPRE_StructGMRESCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmrescreate, HYPRE_STRUCTGMRESCREATE)( hypre_F90_Comm *comm,
                                            hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr   )

{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructGMRESCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structgmresdestroy, HYPRE_STRUCTGMRESDESTROY)( hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGMRESDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structgmressetup, HYPRE_STRUCTGMRESSETUP)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGMRESSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structgmressolve, HYPRE_STRUCTGMRESSOLVE)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGMRESSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetKDim
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetkdim, HYPRE_STRUCTGMRESSETKDIM)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int       *k_dim,
                                                        HYPRE_Int       *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_StructGMRESSetKDim( (HYPRE_StructSolver) *solver,
                                             (HYPRE_Int)                 *k_dim ));
}
/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressettol, HYPRE_STRUCTGMRESSETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructGMRESSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetmaxiter, HYPRE_STRUCTGMRESSETMAXITER)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructGMRESSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetPrecond
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetprecond, HYPRE_STRUCTGMRESSETPRECOND)( hypre_F90_Obj *solver,
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
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSMGSolve,
                                      HYPRE_StructSMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 1)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructPFMGSolve,
                                      HYPRE_StructPFMGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 5)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructSparseMSGSolve,
                                      HYPRE_StructSparseMSGSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 6)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
                                      HYPRE_StructJacobiSolve,
                                      HYPRE_StructJacobiSetup,
                                      (HYPRE_StructSolver) *precond_solver) );
   }
   else if (*precond_id == 8)
   {
      *ierr = (HYPRE_Int)
         ( HYPRE_StructGMRESSetPrecond( (HYPRE_StructSolver) *solver,
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
 * HYPRE_StructGMRESSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetlogging, HYPRE_STRUCTGMRESSETLOGGING)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *logging,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructGMRESSetLogging( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmressetprintlevel, HYPRE_STRUCTGMRESSETPRINTLEVEL)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *print_level,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructGMRESSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmresgetnumiteratio, HYPRE_STRUCTGMRESGETNUMITERATIO)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructGMRESGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgmresgetfinalrelati, HYPRE_STRUCTGMRESGETFINALRELATI)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructGMRESGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
