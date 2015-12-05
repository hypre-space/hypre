/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgcreate, HYPRE_STRUCTSPARSEMSGCREATE)( int      *comm,
                                            long int *solver,
                                            int      *ierr   )

{
   *ierr = (int)
      ( HYPRE_StructSparseMSGCreate( (MPI_Comm)             *comm,
                               (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsparsemsgdestroy, HYPRE_STRUCTSPARSEMSGDESTROY)( long int *solver,
                                          int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSparseMSGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsparsemsgsetup, HYPRE_STRUCTSPARSEMSGSETUP)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSparseMSGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsparsemsgsolve, HYPRE_STRUCTSPARSEMSGSOLVE)( long int *solver,
                                       long int *A,
                                       long int *b,
                                       long int *x,
                                       int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSparseMSGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsettol, HYPRE_STRUCTSPARSEMSGSETTOL)( long int *solver,
                                        double   *tol,
                                        int      *ierr   )
{
   *ierr = (int) ( HYPRE_StructSparseMSGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetmaxiter, HYPRE_STRUCTSPARSEMSGSETMAXITER)( long int *solver,
                                            int      *max_iter,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (int)                *max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJump
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetjump, HYPRE_STRUCTSPARSEMSGSETJUMP)( long int *solver,
                                            int      *jump,
                                            int      *ierr     )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetJump( (HYPRE_StructSolver) *solver,
                                   (int)                *jump ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelchan, HYPRE_STRUCTSPARSEMSGSETRELCHAN)( long int *solver,
                                              int      *rel_change,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetRelChange( (HYPRE_StructSolver) *solver,
                                     (int)                *rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetzerogue, HYPRE_STRUCTSPARSEMSGSETZEROGUE)( long int *solver,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnonzero, HYPRE_STRUCTSPARSEMSGSETNONZERO)( long int *solver,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetrelaxty, HYPRE_STRUCTSPARSEMSGSETRELAXTY)( long int *solver,
                                              int      *relax_type,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetRelaxType( (HYPRE_StructSolver) *solver,
                                     (int)                *relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structsparsemsgsetjacobiweigh, HYPRE_STRUCTSPARSEMSGSETJACOBIWEIGH)
                                                  (long int *solver,
                                                   double   *weight,
                                                   int      *ierr)
{
   *ierr = (int) (HYPRE_StructSparseMSGSetJacobiWeight( (HYPRE_StructSolver) *solver,
                                                        (double)             *weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumprer, HYPRE_STRUCTSPARSEMSGSETNUMPRER)( long int *solver,
                                              int      *num_pre_relax,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                     (int)                *num_pre_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumpost, HYPRE_STRUCTSPARSEMSGSETNUMPOST)( long int *solver,
                                              int      *num_post_relax,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                     (int)                *num_post_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetNumFineRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetnumfine, HYPRE_STRUCTSPARSEMSGSETNUMFINE)( long int *solver,
                                              int      *num_fine_relax,
                                              int      *ierr       )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetNumFineRelax( (HYPRE_StructSolver) *solver,
                                     (int)                *num_fine_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetlogging, HYPRE_STRUCTSPARSEMSGSETLOGGING)( long int *solver,
                                            int      *logging,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetLogging( (HYPRE_StructSolver) *solver,
                                   (int)                *logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsgsetprintle, HYPRE_STRUCTSPARSEMSGSETPRINTLE)( long int *solver,
                                            int      *print_level,
                                            int      *ierr    )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                   (int)                *print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetnumiter, HYPRE_STRUCTSPARSEMSGGETNUMITER)( long int *solver,
                                                  int      *num_iterations,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSparseMSGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsparsemsggetfinalre, HYPRE_STRUCTSPARSEMSGGETFINALRE)( long int *solver,
                                                  double   *norm,
                                                  int      *ierr   )
{
   *ierr = (int)
      ( HYPRE_StructSparseMSGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
