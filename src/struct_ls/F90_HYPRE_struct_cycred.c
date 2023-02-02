/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structcycredcreate, HYPRE_STRUCTCYCREDCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructCycRedCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassObjRef (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structcycreddestroy, HYPRE_STRUCTCYCREDDESTROY)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructCycRedDestroy(
                hypre_F90_PassObj (HYPRE_StructSolver, solver) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structcycredsetup, HYPRE_STRUCTCYCREDSETUP)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructCycRedSetup(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structcycredsolve, HYPRE_STRUCTCYCREDSOLVE)
( hypre_F90_Obj *solver,
  hypre_F90_Obj *A,
  hypre_F90_Obj *b,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructCycRedSolve(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassObj (HYPRE_StructMatrix, A),
                hypre_F90_PassObj (HYPRE_StructVector, b),
                hypre_F90_PassObj (HYPRE_StructVector, x)      ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structcycredsettdim, HYPRE_STRUCTCYCREDSETTDIM)
( hypre_F90_Obj *solver,
  hypre_F90_Int *tdim,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructCycRedSetTDim(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (tdim) ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structcycredsetbase, HYPRE_STRUCTCYCREDSETBASE)
( hypre_F90_Obj *solver,
  hypre_F90_Int *ndim,
  hypre_F90_IntArray *base_index,
  hypre_F90_IntArray *base_stride,
  hypre_F90_Int *ierr           )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructCycRedSetBase(
                hypre_F90_PassObj (HYPRE_StructSolver, solver),
                hypre_F90_PassInt (ndim),
                hypre_F90_PassIntArray (base_index),
                hypre_F90_PassIntArray (base_stride) ) );
}

#ifdef __cplusplus
}
#endif
