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
 * HYPRE_BlockTridiag interface
 *
 *****************************************************************************/

#include "block_tridiag.h"

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagCreate(HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_BlockTridiagCreate( ) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagDestroy(HYPRE_Solver solver)
{
   return(hypre_BlockTridiagDestroy((void *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b, HYPRE_ParVector x)
{
   return(hypre_BlockTridiagSetup((void *) solver, (hypre_ParCSRMatrix *) A,
                              (hypre_ParVector *) b, (hypre_ParVector *) x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,   HYPRE_ParVector x)
{
   return(hypre_BlockTridiagSolve((void *) solver, (hypre_ParCSRMatrix *) A,
                               (hypre_ParVector *) b, (hypre_ParVector *) x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSetIndexSet(HYPRE_Solver solver,int n, int *inds)
{
   return(hypre_BlockTridiagSetIndexSet((void *) solver, n, inds));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSetAMGStrengthThreshold(HYPRE_Solver solver,double thresh)
{
   return(hypre_BlockTridiagSetAMGStrengthThreshold((void *) solver, thresh));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSetAMGNumSweeps(HYPRE_Solver solver, int num_sweeps)
{
   return(hypre_BlockTridiagSetAMGNumSweeps((void *) solver,num_sweeps));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSetAMGRelaxType(HYPRE_Solver solver, int relax_type)
{
   return(hypre_BlockTridiagSetAMGRelaxType( (void *) solver, relax_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

int HYPRE_BlockTridiagSetPrintLevel(HYPRE_Solver solver, int print_level)
{
   return(hypre_BlockTridiagSetPrintLevel( (void *) solver, print_level));
}

