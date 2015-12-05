/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
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
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagCreate(
      hypre_F90_PassObjRef (HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_blockTridiagDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagdestroy, HYPRE_BLOCKTRIDIAGDESTROY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagDestroy(
      hypre_F90_PassObj (HYPRE_Solver, solver));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetup
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetup, HYPRE_BLOCKTRIDIAGSETUP)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetup(
      hypre_F90_PassObj (HYPRE_Solver, solver), 
      hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
      hypre_F90_PassObj (HYPRE_ParVector, b), 
      hypre_F90_PassObj (HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSolve
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsolve, HYPRE_BLOCKTRIDIAGSOLVE)
   (hypre_F90_Obj *solver,
    hypre_F90_Obj *A,
    hypre_F90_Obj *b,
    hypre_F90_Obj *x,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSolve(
      hypre_F90_PassObj (HYPRE_Solver, solver), 
      hypre_F90_PassObj (HYPRE_ParCSRMatrix, A),
      hypre_F90_PassObj (HYPRE_ParVector, b), 
      hypre_F90_PassObj (HYPRE_ParVector, x));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetIndexSet
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetindexset, HYPRE_BLOCKTRIDIAGSETINDEXSET)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *n,
    hypre_F90_IntArray *inds,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetIndexSet(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassInt (n), 
      hypre_F90_PassIntArray (inds));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGStrengthThreshold
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgstrengt, HYPRE_BLOCKTRIDIAGSETAMGSTRENGT)
   (hypre_F90_Obj *solver,
    hypre_F90_Dbl *thresh,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetAMGStrengthThreshold(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassDbl (thresh));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGNumSweeps
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgnumswee, HYPRE_BLOCKTRIDIAGSETAMGNUMSWEE)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *num_sweeps,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetAMGNumSweeps(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassInt (num_sweeps));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetAMGRelaxType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetamgrelaxty, HYPRE_BLOCKTRIDIAGSETAMGRELAXTY)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *relax_type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetAMGRelaxType(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassInt (relax_type));
}

/*--------------------------------------------------------------------------
 * HYPRE_BlockTridiagSetPrintLevel
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_blocktridiagsetprintlevel, HYPRE_BLOCKTRIDIAGSETPRINTLEVEL)
   (hypre_F90_Obj *solver,
    hypre_F90_Int *print_level,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_BlockTridiagSetPrintLevel(
      hypre_F90_PassObj (HYPRE_Solver, solver),
      hypre_F90_PassInt (print_level));
}
