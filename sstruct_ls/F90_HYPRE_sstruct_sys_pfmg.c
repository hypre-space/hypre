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
 * HYPRE_SStructSysPFMG interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgcreate, HYPRE_SSTRUCTSYSPFMGCREATE)
                                                         (HYPRE_Int     *comm,
                                                          hypre_F90_Obj *solver,
                                                          HYPRE_Int     *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGCreate( (MPI_Comm)             *comm,
                                              (HYPRE_SStructSolver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgdestroy, HYPRE_SSTRUCTSYSPFMGDESTROY)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetup, HYPRE_SSTRUCTSYSPFMGSETUP)
                                                         (hypre_F90_Obj *solver,
                                                          hypre_F90_Obj *A,
                                                          hypre_F90_Obj *b,
                                                          hypre_F90_Obj *x,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetup( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_SStructMatrix) *A,
                                             (HYPRE_SStructVector) *b,
                                             (HYPRE_SStructVector) *x    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsolve, HYPRE_SSTRUCTSYSPFMGSOLVE)
                                                         (hypre_F90_Obj *solver,
                                                          hypre_F90_Obj *A,
                                                          hypre_F90_Obj *b,
                                                          hypre_F90_Obj *x,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSolve( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_SStructMatrix) *A,
                                             (HYPRE_SStructVector) *b,
                                             (HYPRE_SStructVector) *x    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsettol, HYPRE_SSTRUCTSYSPFMGSETTOL)
                                                         (hypre_F90_Obj *solver,
                                                          double   *tol,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetTol( (HYPRE_SStructSolver) *solver,
                                              (double)              *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetmaxiter, HYPRE_SSTRUCTSYSPFMGSETMAXITER)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *max_iter,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                  (HYPRE_Int)                 *max_iter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelchang, HYPRE_SSTRUCTSYSPFMGSETRELCHANG)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *rel_change,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetRelChange( (HYPRE_SStructSolver) *solver,
                                                    (HYPRE_Int)                 *rel_change  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetzerogues, HYPRE_SSTRUCTSYSPFMGSETZEROGUES)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnonzerog, HYPRE_SSTRUCTSYSPFMGSETNONZEROG)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelaxtyp, HYPRE_SSTRUCTSYSPFMGSETRELAXTYP)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *relax_type,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetRelaxType( (HYPRE_SStructSolver) *solver,
                                                    (HYPRE_Int)                 *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
void
hypre_F90_IFACE(hypre_sstructsyspfmgsetjacobiweigh, HYPRE_SSTRUCTSYSPFMGSETJACOBIWEIGH)
                                                         (hypre_F90_Obj *solver,
                                                          double   *weight,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetJacobiWeight( (HYPRE_SStructSolver) *solver,
                                                       (double)              *weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumprere, HYPRE_SSTRUCTSYSPFMGSETNUMPRERE)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *num_pre_relax,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetNumPreRelax( (HYPRE_SStructSolver) *solver,
                                                      (HYPRE_Int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumpostr, HYPRE_SSTRUCTSYSPFMGSETNUMPOSTR)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *num_post_relax,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetNumPostRelax( (HYPRE_SStructSolver) *solver,
                                                       (HYPRE_Int)                 *num_post_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetskiprela, HYPRE_SSTRUCTSYSPFMGSETSKIPRELA)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *skip_relax,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetSkipRelax( (HYPRE_SStructSolver) *solver,
                                                    (HYPRE_Int)                 *skip_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetdxyz, HYPRE_SSTRUCTSYSPFMGSETDXYZ)
                                                         (hypre_F90_Obj *solver,
                                                          double   *dxyz,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetDxyz( (HYPRE_SStructSolver) *solver,
                                               (double *)             dxyz   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetlogging, HYPRE_SSTRUCTSYSPFMGSETLOGGING)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *logging,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetLogging( (HYPRE_SStructSolver) *solver,
                                                  (HYPRE_Int)                 *logging ));
}

/*--------------------------------------------------------------------------
HYPRE_SStructSysPFMGSetPrintLevel
*--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetprintlev, HYPRE_SSTRUCTSYSPFMGSETPRINTLEV)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *print_level,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                     (HYPRE_Int)                 *print_level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetnumitera, HYPRE_SSTRUCTSYSPFMGGETNUMITERA)
                                                         (hypre_F90_Obj *solver,
                                                          HYPRE_Int      *num_iterations,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                        (HYPRE_Int *)                num_iterations ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetfinalrel, HYPRE_SSTRUCTSYSPFMGGETFINALREL)
                                                         (hypre_F90_Obj *solver,
                                                          double   *norm,
                                                          HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                    (double *)             norm   ));
}
