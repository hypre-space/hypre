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
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagCreate( (HYPRE_Solver *) solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagdestroy, HYPRE_BLOCKTRIDIAGDESTROY)
               (hypre_F90_Obj *solver, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagDestroy( (HYPRE_Solver) *solver);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetup, HYPRE_BLOCKTRIDIAGSETUP)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSetup( (HYPRE_Solver)       *solver, 
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b, 
                                          (HYPRE_ParVector)    *x);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsolve, HYPRE_BLOCKTRIDIAGSOLVE)
               (hypre_F90_Obj *solver, hypre_F90_Obj *A, hypre_F90_Obj *b, hypre_F90_Obj *x, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSolve( (HYPRE_Solver)       *solver, 
                                          (HYPRE_ParCSRMatrix) *A,
                                          (HYPRE_ParVector)    *b, 
                                          (HYPRE_ParVector)    *x);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetindexset, HYPRE_BLOCKTRIDIAGSETINDEXSET)
               (hypre_F90_Obj *solver, HYPRE_Int *n, HYPRE_Int *inds, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSetIndexSet( (HYPRE_Solver) *solver,
                                                (HYPRE_Int)          *n, 
                                                (HYPRE_Int *)         inds);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgstrengt, HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
               (hypre_F90_Obj *solver, double *thresh, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSetAMGStrengthThreshold( (HYPRE_Solver) *solver,
                                                            (double)       *thresh);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgnumswee, HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
               (hypre_F90_Obj *solver, HYPRE_Int *num_sweeps, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSetAMGNumSweeps( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *num_sweeps);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgrelaxty, HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
               (hypre_F90_Obj *solver, HYPRE_Int *relax_type, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSetAMGRelaxType( (HYPRE_Solver) *solver,
                                                    (HYPRE_Int)          *relax_type);
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetprintlevel, HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
               (hypre_F90_Obj *solver, HYPRE_Int *print_level, HYPRE_Int *ierr)
{
   *ierr = (HYPRE_Int) HYPRE_BlockTridiagSetPrintLevel( (HYPRE_Solver) *solver,
                                                  (HYPRE_Int)          *print_level);
}
