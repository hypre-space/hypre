/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
                                                         (long int *comm,
                                                          long int *solver,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGCreate( (MPI_Comm)             *comm,
                                              (HYPRE_SStructSolver *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgdestroy, HYPRE_SSTRUCTSYSPFMGDESTROY)
                                                         (long int *solver,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGDestroy( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetup, HYPRE_SSTRUCTSYSPFMGSETUP)
                                                         (long int *solver,
                                                          long int *A,
                                                          long int *b,
                                                          long int *x,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetup( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_SStructMatrix) *A,
                                             (HYPRE_SStructVector) *b,
                                             (HYPRE_SStructVector) *x    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsolve, HYPRE_SSTRUCTSYSPFMGSOLVE)
                                                         (long int *solver,
                                                          long int *A,
                                                          long int *b,
                                                          long int *x,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSolve( (HYPRE_SStructSolver) *solver,
                                             (HYPRE_SStructMatrix) *A,
                                             (HYPRE_SStructVector) *b,
                                             (HYPRE_SStructVector) *x    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetTol
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsettol, HYPRE_SSTRUCTSYSPFMGSETTOL)
                                                         (long int *solver,
                                                          double   *tol,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetTol( (HYPRE_SStructSolver) *solver,
                                              (double)              *tol    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetMaxIter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetmaxiter, HYPRE_SSTRUCTSYSPFMGSETMAXITER)
                                                         (long int *solver,
                                                          int      *max_iter,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetMaxIter( (HYPRE_SStructSolver) *solver,
                                                  (int)                 *max_iter  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelChange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelchang, HYPRE_SSTRUCTSYSPFMGSETRELCHANG)
                                                         (long int *solver,
                                                          int      *rel_change,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetRelChange( (HYPRE_SStructSolver) *solver,
                                                    (int)                 *rel_change  ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetzerogues, HYPRE_SSTRUCTSYSPFMGSETZEROGUES)
                                                         (long int *solver,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNonZeroGuess
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnonzerog, HYPRE_SSTRUCTSYSPFMGSETNONZEROG)
                                                         (long int *solver,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetNonZeroGuess( (HYPRE_SStructSolver) *solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetrelaxtyp, HYPRE_SSTRUCTSYSPFMGSETRELAXTYP)
                                                         (long int *solver,
                                                          int      *relax_type,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetRelaxType( (HYPRE_SStructSolver) *solver,
                                                    (int)                 *relax_type ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetJacobiWeight
 *--------------------------------------------------------------------------*/
                                                                                                                                                               
void
hypre_F90_IFACE(hypre_sstructsyspfmgsetjacobiweigh, HYPRE_SSTRUCTSYSPFMGSETJACOBIWEIGH)
                                                         (long int *solver,
                                                          double   *weight,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetJacobiWeight( (HYPRE_SStructSolver) *solver,
                                                       (double)              *weight ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPreRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumprere, HYPRE_SSTRUCTSYSPFMGSETNUMPRERE)
                                                         (long int *solver,
                                                          int      *num_pre_relax,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetNumPreRelax( (HYPRE_SStructSolver) *solver,
                                                      (int)                 *num_pre_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetNumPostRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetnumpostr, HYPRE_SSTRUCTSYSPFMGSETNUMPOSTR)
                                                         (long int *solver,
                                                          int      *num_post_relax,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetNumPostRelax( (HYPRE_SStructSolver) *solver,
                                                       (int)                 *num_post_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetSkipRelax
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetskiprela, HYPRE_SSTRUCTSYSPFMGSETSKIPRELA)
                                                         (long int *solver,
                                                          int      *skip_relax,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetSkipRelax( (HYPRE_SStructSolver) *solver,
                                                    (int)                 *skip_relax ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetDxyz
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetdxyz, HYPRE_SSTRUCTSYSPFMGSETDXYZ)
                                                         (long int *solver,
                                                          double   *dxyz,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetDxyz( (HYPRE_SStructSolver) *solver,
                                               (double *)             dxyz   ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGSetLogging
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetlogging, HYPRE_SSTRUCTSYSPFMGSETLOGGING)
                                                         (long int *solver,
                                                          int      *logging,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetLogging( (HYPRE_SStructSolver) *solver,
                                                  (int)                 *logging ));
}

/*--------------------------------------------------------------------------
HYPRE_SStructSysPFMGSetPrintLevel
*--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmgsetprintlev, HYPRE_SSTRUCTSYSPFMGSETPRINTLEV)
                                                         (long int *solver,
                                                          int      *print_level,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGSetPrintLevel( (HYPRE_SStructSolver) *solver,
                                                     (int)                 *print_level ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetNumIterations
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetnumitera, HYPRE_SSTRUCTSYSPFMGGETNUMITERA)
                                                         (long int *solver,
                                                          int      *num_iterations,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGGetNumIterations( (HYPRE_SStructSolver) *solver,
                                                        (int *)                num_iterations ));
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsyspfmggetfinalrel, HYPRE_SSTRUCTSYSPFMGGETFINALREL)
                                                         (long int *solver,
                                                          double   *norm,
                                                          int      *ierr)
{
   *ierr = (int) (HYPRE_SStructSysPFMGGetFinalRelativeResidualNorm( (HYPRE_SStructSolver) *solver,
                                                                    (double *)             norm   ));
}
