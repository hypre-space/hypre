/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team <hypre-users@llnl.gov>, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_BlockTridiag Fortran interface
 *
 *****************************************************************************/

#include "block_tridiag.h"
#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagcreate, HYPRE_BLOCKTRIDIAGCREATE)
               (long int *solver, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagCreate( (HYPRE_Solver *) solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagdestroy, HYPRE_BLOCKTRIDIAGDESTROY)
               (long int *solver, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagDestroy( (HYPRE_Solver) *solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetup, HYPRE_BLOCKTRIDIAGSETUP)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetup( (HYPRE_Solver)       *solver, 
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b, 
                                          (HYPRE_ParVector)    *x);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsolve, HYPRE_BLOCKTRIDIAGSOLVE)
               (long int *solver, long int *A, long int *b, long int *x, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSolve( (HYPRE_Solver)       *solver, 
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b, 
                                          (HYPRE_ParVector)    *x);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetindexset, HYPRE_BLOCKTRIDIAGSETINDEXSET)
               (long int *solver, int *n, int *inds, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetIndexSet( (HYPRE_Solver) *solver,
                                                (int)          *n, 
                                                (int *)         inds);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgstrengt, HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
               (long int *solver, double *thresh, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetAMGStrengthThreshold( (HYPRE_Solver) *solver,
                                                            (double)       *thresh);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgnumswee, HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
               (long int *solver, int *num_sweeps, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetAMGNumSweeps( (HYPRE_Solver) *solver,
                                                    (int)          *num_sweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgrelaxty, HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
               (long int *solver, int *relax_type, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetAMGRelaxType( (HYPRE_Solver) *solver,
                                                    (int)          *relax_type);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetprintlevel, HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
               (long int *solver, int *print_level, int *ierr)
{
   *ierr = (int) HYPRE_BlockTridiagSetPrintLevel( (HYPRE_Solver) *solver,
                                                  (int)          *print_level);
}
