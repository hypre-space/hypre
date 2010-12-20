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
 * HYPRE_StructSMG Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgcreate, HYPRE_STRUCTSMGCREATE)( HYPRE_Int      *comm,
                                        hypre_F90_Obj *solver,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSMGCreate( (MPI_Comm)             *comm,
                                          (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgdestroy, HYPRE_STRUCTSMGDESTROY)( hypre_F90_Obj *solver,
                                         HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSMGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsetup, HYPRE_STRUCTSMGSETUP)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSMGSetup( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structsmgsolve, HYPRE_STRUCTSMGSOLVE)( hypre_F90_Obj *solver,
                                       hypre_F90_Obj *A,
                                       hypre_F90_Obj *b,
                                       hypre_F90_Obj *x,
                                       HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSMGSolve( (HYPRE_StructSolver) *solver,
                                         (HYPRE_StructMatrix) *A,
                                         (HYPRE_StructVector) *b,
                                         (HYPRE_StructVector) *x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMemoryUse, HYPRE_StructSMGGetMemoryUse
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmemoryuse, HYPRE_STRUCTSMGSETMEMORYUSE)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *memory_use,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetMemoryUse( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *memory_use ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmemoryuse, HYPRE_STRUCTSMGGETMEMORYUSE)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *memory_use,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetMemoryUse( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int *)               memory_use ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetTol, HYPRE_StructSMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsettol, HYPRE_STRUCTSMGSETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSMGSetTol( (HYPRE_StructSolver) *solver,
                                          (double)             *tol ) );
}

void
hypre_F90_IFACE(hypre_structsmggettol, HYPRE_STRUCTSMGGETTOL)( hypre_F90_Obj *solver,
                                        double   *tol,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructSMGGetTol( (HYPRE_StructSolver) *solver,
                                          (double *)            tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetMaxIter, HYPRE_StructSMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetmaxiter, HYPRE_STRUCTSMGSETMAXITER)( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *max_iter ) );
}

void
hypre_F90_IFACE(hypre_structsmggetmaxiter, HYPRE_STRUCTSMGGETMAXITER)
                                          ( hypre_F90_Obj *solver,
                                            HYPRE_Int      *max_iter,
                                            HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetMaxIter( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int *)               max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetRelChange, HYPRE_StructSMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetrelchange, HYPRE_STRUCTSMGSETRELCHANGE)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *rel_change,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetRelChange( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int)                *rel_change ) );
}

void
hypre_F90_IFACE(hypre_structsmggetrelchange, HYPRE_STRUCTSMGGETRELCHANGE)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *rel_change,
                                              HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetRelChange( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int *)               rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetZeroGuess, HYPRE_StructSMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetzeroguess, HYPRE_STRUCTSMGSETZEROGUESS)( hypre_F90_Obj *solver,
                                              HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}
 
void
hypre_F90_IFACE(hypre_structsmggetzeroguess, HYPRE_STRUCTSMGGETZEROGUESS)
                                            ( hypre_F90_Obj *solver,
                                              HYPRE_Int      *zeroguess,
                                              HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetZeroGuess( (HYPRE_StructSolver) *solver,
                                     (HYPRE_Int *)               zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structsmgsetnonzeroguess, HYPRE_STRUCTSMGSETNONZEROGUESS)( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPreRelax, HYPRE_StructSMGGetNumPreRelax
 *
 * Note that we require at least 1 pre-relax sweep. 
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumprerelax, HYPRE_STRUCTSMGSETNUMPRERELAX)( hypre_F90_Obj *solver,
                                                HYPRE_Int      *num_pre_relax,
                                                HYPRE_Int      *ierr         )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                       (HYPRE_Int)                *num_pre_relax) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumprerelax, HYPRE_STRUCTSMGGETNUMPRERELAX)
                                              ( hypre_F90_Obj *solver,
                                                HYPRE_Int      *num_pre_relax,
                                                HYPRE_Int      *ierr         )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetNumPreRelax( (HYPRE_StructSolver) *solver,
                                       (HYPRE_Int *)               num_pre_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetNumPostRelax, HYPRE_StructSMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetnumpostrelax, HYPRE_STRUCTSMGSETNUMPOSTRELAX)
                                               ( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *num_post_relax,
                                                 HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int)                *num_post_relax) );
}

void
hypre_F90_IFACE(hypre_structsmggetnumpostrelax, HYPRE_STRUCTSMGGETNUMPOSTRELAX)
                                               ( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *num_post_relax,
                                                 HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetNumPostRelax( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int *)               num_post_relax) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetLogging, HYPRE_StructSMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetlogging, HYPRE_STRUCTSMGSETLOGGING)
                                          ( hypre_F90_Obj *solver,
                                            HYPRE_Int      *logging,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetLogging( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int)                *logging) );
}

void
hypre_F90_IFACE(hypre_structsmggetlogging, HYPRE_STRUCTSMGGETLOGGING)
                                          ( hypre_F90_Obj *solver,
                                            HYPRE_Int      *logging,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetLogging( (HYPRE_StructSolver) *solver,
                                   (HYPRE_Int *)               logging) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGSetPrintLevel, HYPRE_StructSMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmgsetprintlevel, HYPRE_STRUCTSMGSETPRINTLEVEL)
                                          ( hypre_F90_Obj *solver,
                                            HYPRE_Int      *print_level,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int)                *print_level) );
}

void
hypre_F90_IFACE(hypre_structsmggetprintlevel, HYPRE_STRUCTSMGGETPRINTLEVEL)
                                          ( hypre_F90_Obj *solver,
                                            HYPRE_Int      *print_level,
                                            HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetPrintLevel( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int *)               print_level) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetnumiterations, HYPRE_STRUCTSMGGETNUMITERATIONS)
                                                ( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetNumIterations( (HYPRE_StructSolver) *solver,
                                         (HYPRE_Int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsmggetfinalrelative, HYPRE_STRUCTSMGGETFINALRELATIVE)
                                                ( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructSMGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm ) );
}
