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
 * HYPRE_SStructSplit solver interface
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"


/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitcreate, HYPRE_SSTRUCTSPLITCREATE)
                                                       (hypre_F90_Comm *comm,
                                                        hypre_F90_Obj *solver_ptr,
                                                        HYPRE_Int     *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitCreate( (MPI_Comm)             *comm,
                                            (HYPRE_SStructSolver *) solver_ptr ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitdestroy, HYPRE_SSTRUCTSPLITDESTROY)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitDestroy( (HYPRE_SStructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetup, HYPRE_SSTRUCTSPLITSETUP)
                                                       (hypre_F90_Obj *solver,
                                                        hypre_F90_Obj *A,
                                                        hypre_F90_Obj *b,
                                                        hypre_F90_Obj *x,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSetup( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsolve, HYPRE_SSTRUCTSPLITSOLVE)
                                                       (hypre_F90_Obj *solver,
                                                        hypre_F90_Obj *A,
                                                        hypre_F90_Obj *b,
                                                        hypre_F90_Obj *x,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSolve( (HYPRE_SStructSolver) *solver,
                                           (HYPRE_SStructMatrix) *A,
                                           (HYPRE_SStructVector) *b,
                                           (HYPRE_SStructVector) *x ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsettol, HYPRE_SSTRUCTSPLITSETTOL)
                                                       (hypre_F90_Obj *solver,
                                                        double   *tol,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSetTol( (HYPRE_SStructSolver) *solver,
                                            (double)              *tol ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetmaxiter, HYPRE_SSTRUCTSPLITSETMAXITER)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int      *max_iter,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                (HYPRE_Int)                 *max_iter ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetzeroguess, HYPRE_SSTRUCTSPLITSETZEROGUESS)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSetZeroGuess( (HYPRE_SStructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetnonzerogue, HYPRE_SSTRUCTSPLITSETNONZEROGUE)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitSetStructSolver
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitsetstructsolv, HYPRE_SSTRUCTSPLITSETSTRUCTSOLV)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int      *ssolver,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitSetStructSolver( (HYPRE_SStructSolver) *solver,
                                                     (HYPRE_Int)                 *ssolver ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitgetnumiterati, HYPRE_SSTRUCTSPLITGETNUMITERATI)
                                                       (hypre_F90_Obj *solver,
                                                        HYPRE_Int      *num_iterations,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                      (HYPRE_Int *)                num_iterations ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSplitGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsplitgetfinalrelat, HYPRE_SSTRUCTSPLITGETFINALRELAT)
                                                       (hypre_F90_Obj *solver,
                                                        double   *norm,
                                                        HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSplitGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                  (double *)             norm ) );
}
