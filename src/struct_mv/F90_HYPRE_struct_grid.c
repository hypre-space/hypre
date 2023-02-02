/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructGrid interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructGridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridcreate, HYPRE_STRUCTGRIDCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_Int *dim,
  hypre_F90_Obj *grid,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructGridCreate(
                hypre_F90_PassComm (comm),
                hypre_F90_PassInt (dim),
                hypre_F90_PassObjRef (HYPRE_StructGrid, grid) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgriddestroy, HYPRE_STRUCTGRIDDESTROY)
( hypre_F90_Obj *grid,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructGridDestroy(
                hypre_F90_PassObj (HYPRE_StructGrid, grid) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridSetExtents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridsetextents, HYPRE_STRUCTGRIDSETEXTENTS)
( hypre_F90_Obj *grid,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructGridSetExtents(
                hypre_F90_PassObj (HYPRE_StructGrid, grid),
                hypre_F90_PassIntArray (ilower),
                hypre_F90_PassIntArray (iupper) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SetStructGridPeriodicity
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridsetperiodic, HYPRE_STRUCTGRIDSETPERIODIC)
( hypre_F90_Obj *grid,
  hypre_F90_IntArray *periodic,
  hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructGridSetPeriodic(
                hypre_F90_PassObj (HYPRE_StructGrid, grid),
                hypre_F90_PassIntArray (periodic)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridassemble, HYPRE_STRUCTGRIDASSEMBLE)
( hypre_F90_Obj *grid,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructGridAssemble(
                hypre_F90_PassObj (HYPRE_StructGrid, grid)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructGridSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structgridsetnumghost, HYPRE_STRUCTGRIDSETNUMGHOST)
( hypre_F90_Obj *grid,
  hypre_F90_IntArray *num_ghost,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructGridSetNumGhost(
                hypre_F90_PassObj (HYPRE_StructGrid, grid),
                hypre_F90_PassIntArray (num_ghost)) );
}

#ifdef __cplusplus
}
#endif
