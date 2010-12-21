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
 * HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgcreate, HYPRE_STRUCTSPARSEMSGCREATE)( hypre_F90_Comm *comm,
                                            hypre_F90_Obj *solver,
                                            HYPRE_Int      *ierr   )

{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsparsemsgdestroy, HYPRE_STRUCTSPARSEMSGDESTROY)( hypre_F90_Obj *solver,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSparseMSGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsparsemsgsetup, HYPRE_STRUCTSPARSEMSGSETUP)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSparseMSGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsparsemsgsolve, HYPRE_STRUCTSPARSEMSGSOLVE)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSparseMSGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsettol, HYPRE_STRUCTSPARSEMSGSETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSparseMSGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetmaxiter, HYPRE_STRUCTSPARSEMSGSETMAXITER)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetjump, HYPRE_STRUCTSPARSEMSGSETJUMP)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *jump,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetJump( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *jump ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelchan, HYPRE_STRUCTSPARSEMSGSETRELCHAN)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *rel_change,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetRelChange( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetzerogue, HYPRE_STRUCTSPARSEMSGSETZEROGUE)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnonzero, HYPRE_STRUCTSPARSEMSGSETNONZERO)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelaxty, HYPRE_STRUCTSPARSEMSGSETRELAXTY)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *relax_type,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetRelaxType( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structsparsemsgsetjacobiweigh, HYPRE_STRUCTSPARSEMSGSETJACOBIWEIGH)
                                                  (hypre_F90_Obj *solver,
                                                   double   *weight,
                                                   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_StructSparseMSGSetJacobiWeight( (HYPRE_StructSolver) *solver,
                                                        (double)             *weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumprer, HYPRE_STRUCTSPARSEMSGSETNUMPRER)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *num_pre_relax,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *num_pre_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumpost, HYPRE_STRUCTSPARSEMSGSETNUMPOST)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *num_post_relax,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *num_post_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumfine, HYPRE_STRUCTSPARSEMSGSETNUMFINE)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *num_fine_relax,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetNumFineRelax( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *num_fine_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetlogging, HYPRE_STRUCTSPARSEMSGSETLOGGING)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *logging,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetLogging( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetprintle, HYPRE_STRUCTSPARSEMSGSETPRINTLE)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *print_level,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetnumiter, HYPRE_STRUCTSPARSEMSGGETNUMITER)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetfinalre, HYPRE_STRUCTSPARSEMSGGETFINALRE)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
