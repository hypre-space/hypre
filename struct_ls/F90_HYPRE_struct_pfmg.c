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
 * HYPRE_StructPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgcreate, HYPRE_STRUCTPFMGCREATE)( hypre_F90_Comm *comm,
                                             hypre_F90_Obj *solver,
                                             HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGCreate( (MPI_Comm)             *comm,
                                (HYPRE_StructSolver *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgdestroy, HYPRE_STRUCTPFMGDESTROY)( hypre_F90_Obj *solver,
                                           HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPFMGDestroy( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetup
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsetup, HYPRE_STRUCTPFMGSETUP)( hypre_F90_Obj *solver,
                                        hypre_F90_Obj *A,
                                        hypre_F90_Obj *b,
                                        hypre_F90_Obj *x,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPFMGSetup( (HYPRE_StructSolver) *solver,
                                          (HYPRE_StructMatrix) *A,
                                          (HYPRE_StructVector) *b,
                                          (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSolve
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structpfmgsolve, HYPRE_STRUCTPFMGSOLVE)( hypre_F90_Obj *solver,
                                        hypre_F90_Obj *A,
                                        hypre_F90_Obj *b,
                                        hypre_F90_Obj *x,
                                        HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPFMGSolve( (HYPRE_StructSolver) *solver,
                                          (HYPRE_StructMatrix) *A,
                                          (HYPRE_StructVector) *b,
                                          (HYPRE_StructVector) *x      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetTol, HYPRE_StructPFMGGetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsettol, HYPRE_STRUCTPFMGSETTOL)( hypre_F90_Obj *solver,
                                         double   *tol,
                                         HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPFMGSetTol( (HYPRE_StructSolver) *solver,
                                           (double)             *tol    ) );
}

void
hypre_F90_IFACE(hypre_structpfmggettol, HYPRE_STRUCTPFMGGETTOL)( hypre_F90_Obj *solver,
                                         double   *tol,
                                         HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPFMGGetTol( (HYPRE_StructSolver) *solver,
                                           (double *)            tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxIter, HYPRE_StructPFMGGetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxiter, HYPRE_STRUCTPFMGSETMAXITER)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *max_iter,
                                             HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetMaxIter( (HYPRE_StructSolver) *solver,
                                    (HYPRE_Int)                *max_iter  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxiter, HYPRE_STRUCTPFMGGETMAXITER)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *max_iter,
                                             HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetMaxIter( (HYPRE_StructSolver) *solver,
                                    (HYPRE_Int *)               max_iter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetMaxLevels, HYPRE_StructPFMGGetMaxLevels
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetmaxlevels, HYPRE_STRUCTPFMGSETMAXLEVELS)
                                           ( hypre_F90_Obj *solver,
                                             HYPRE_Int      *max_levels,
                                             HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetMaxLevels( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int)                *max_levels  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetmaxlevels, HYPRE_STRUCTPFMGGETMAXLEVELS)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *max_levels,
                                             HYPRE_Int      *ierr     )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetMaxLevels( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int *)               max_levels  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelChange, HYPRE_StructPFMGGetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelchange, HYPRE_STRUCTPFMGSETRELCHANGE)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *rel_change,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetRelChange( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int)                *rel_change  ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelchange, HYPRE_STRUCTPFMGGETRELCHANGE)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *rel_change,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetRelChange( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int *)               rel_change  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetZeroGuess, HYPRE_StructPFMGGetZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetzeroguess, HYPRE_STRUCTPFMGSETZEROGUESS)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetZeroGuess( (HYPRE_StructSolver) *solver ) );
}
 
void
hypre_F90_IFACE(hypre_structpfmggetzeroguess, HYPRE_STRUCTPFMGGETZEROGUESS)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *zeroguess,
                                               HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetZeroGuess( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int *)               zeroguess ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structpfmgsetnonzeroguess, HYPRE_STRUCTPFMGSETNONZEROGUESS)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetNonZeroGuess( (HYPRE_StructSolver) *solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRelaxType, HYPRE_StructPFMGGetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetrelaxtype, HYPRE_STRUCTPFMGSETRELAXTYPE)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *relax_type,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetRelaxType( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int)                *relax_type ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetrelaxtype, HYPRE_STRUCTPFMGGETRELAXTYPE)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *relax_type,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetRelaxType( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int *)               relax_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
void
hypre_F90_IFACE(hypre_structpfmgsetjacobiweigh, HYPRE_STRUCTPFMGSETJACOBIWEIGH)
                                                  (hypre_F90_Obj *solver,
                                                   double   *weight,
                                                   HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_StructPFMGSetJacobiWeight( (HYPRE_StructSolver) *solver,
                                                   (double)             *weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetRAPType, HYPRE_StructPFMGSetRapType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetraptype, HYPRE_STRUCTPFMGSETRAPTYPE)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *rap_type,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetRAPType( (HYPRE_StructSolver) *solver,
                                      (HYPRE_Int)              *rap_type ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetraptype, HYPRE_STRUCTPFMGGETRAPTYPE)( hypre_F90_Obj *solver,
                                               HYPRE_Int      *rap_type,
                                               HYPRE_Int      *ierr       )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetRAPType( (HYPRE_StructSolver) *solver,
                                    (HYPRE_Int *)               rap_type ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPreRelax, HYPRE_StructPFMGGetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumprerelax, HYPRE_STRUCTPFMGSETNUMPRERELAX)( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *num_pre_relax,
                                                 HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetNumPreRelax( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int)                *num_pre_relax ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumprerelax, HYPRE_STRUCTPFMGGETNUMPRERELAX)
                                               ( hypre_F90_Obj *solver,
                                                 HYPRE_Int      *num_pre_relax,
                                                 HYPRE_Int      *ierr          )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetNumPreRelax( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int *)               num_pre_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetNumPostRelax, HYPRE_StructPFMGGetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetnumpostrelax, HYPRE_STRUCTPFMGSETNUMPOSTRELAX)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_post_relax,
                                                  HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetNumPostRelax( (HYPRE_StructSolver) *solver,
                                         (HYPRE_Int)                *num_post_relax ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetnumpostrelax, HYPRE_STRUCTPFMGGETNUMPOSTRELAX)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_post_relax,
                                                  HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetNumPostRelax( (HYPRE_StructSolver) *solver,
                                         (HYPRE_Int *)               num_post_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetSkipRelax, HYPRE_StructPFMGGetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetskiprelax, HYPRE_STRUCTPFMGSETSKIPRELAX)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *skip_relax,
                                                  HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetSkipRelax( (HYPRE_StructSolver) *solver,
                                         (HYPRE_Int)                *skip_relax ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetskiprelax, HYPRE_STRUCTPFMGGETSKIPRELAX)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *skip_relax,
                                                  HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetSkipRelax( (HYPRE_StructSolver) *solver,
                                       (HYPRE_Int *)              skip_relax ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetdxyz, HYPRE_STRUCTPFMGSETDXYZ)( hypre_F90_Obj *solver,
                                          double   *dxyz,
                                          HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int) ( HYPRE_StructPFMGSetDxyz( (HYPRE_StructSolver) *solver,
                                            (double *)           dxyz   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetLogging, HYPRE_StructPFMGGetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetlogging, HYPRE_STRUCTPFMGSETLOGGING)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *logging,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetLogging( (HYPRE_StructSolver) *solver,
                                    (HYPRE_Int)                *logging ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetlogging, HYPRE_STRUCTPFMGGETLOGGING)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *logging,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetLogging( (HYPRE_StructSolver) *solver,
                                    (HYPRE_Int *)               logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGSetPrintLevel, HYPRE_StructPFMGGetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmgsetprintlevel, HYPRE_STRUCTPFMGSETPRINTLEVEL)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *print_level,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGSetPrintLevel( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int)             *print_level ) );
}

void
hypre_F90_IFACE(hypre_structpfmggetprintlevel, HYPRE_STRUCTPFMGGETPRINTLEVEL)( hypre_F90_Obj *solver,
                                             HYPRE_Int      *print_level,
                                             HYPRE_Int      *ierr    )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetPrintLevel( (HYPRE_StructSolver) *solver,
                                        (HYPRE_Int *)              print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetnumiteration, HYPRE_STRUCTPFMGGETNUMITERATION)( hypre_F90_Obj *solver,
                                                  HYPRE_Int      *num_iterations,
                                                  HYPRE_Int      *ierr           )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetNumIterations(
         (HYPRE_StructSolver) *solver,
         (HYPRE_Int *)              num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structpfmggetfinalrelativ, HYPRE_STRUCTPFMGGETFINALRELATIV)( hypre_F90_Obj *solver,
                                                  double   *norm,
                                                  HYPRE_Int      *ierr   )
{
   *ierr = (HYPRE_Int)
      ( HYPRE_StructPFMGGetFinalRelativeResidualNorm(
         (HYPRE_StructSolver) *solver,
         (double *)           norm   ) );
}
