/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_BlockTridiag interface
 *
 *****************************************************************************/

#include "block_tridiag.h"

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagCreate(HYPRE_Solver *solver)
{
   *solver = (HYPRE_Solver) hypre_BlockTridiagCreate( ) ;
   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagDestroy(HYPRE_Solver solver)
{
   return (hypre_BlockTridiagDestroy((void *) solver ));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSetup(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b, HYPRE_ParVector x)
{
   return (hypre_BlockTridiagSetup((void *) solver, (hypre_ParCSRMatrix *) A,
                                   (hypre_ParVector *) b, (hypre_ParVector *) x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSolve(HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b,   HYPRE_ParVector x)
{
   return (hypre_BlockTridiagSolve((void *) solver, (hypre_ParCSRMatrix *) A,
                                   (hypre_ParVector *) b, (hypre_ParVector *) x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSetIndexSet(HYPRE_Solver solver, HYPRE_Int n, HYPRE_Int *inds)
{
   return (hypre_BlockTridiagSetIndexSet((void *) solver, n, inds));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSetAMGStrengthThreshold(HYPRE_Solver solver, HYPRE_Real thresh)
{
   return (hypre_BlockTridiagSetAMGStrengthThreshold((void *) solver, thresh));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSetAMGNumSweeps(HYPRE_Solver solver, HYPRE_Int num_sweeps)
{
   return (hypre_BlockTridiagSetAMGNumSweeps((void *) solver, num_sweeps));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSetAMGRelaxType(HYPRE_Solver solver, HYPRE_Int relax_type)
{
   return (hypre_BlockTridiagSetAMGRelaxType( (void *) solver, relax_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_BlockTridiagSetPrintLevel(HYPRE_Solver solver, HYPRE_Int print_level)
{
   return (hypre_BlockTridiagSetPrintLevel( (void *) solver, print_level));
}

