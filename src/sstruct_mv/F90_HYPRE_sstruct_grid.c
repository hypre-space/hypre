/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructGrid interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_SStructGridCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridcreate, HYPRE_SSTRUCTGRIDCREATE)
(hypre_F90_Comm   *comm,
 hypre_F90_Int    *ndim,
 hypre_F90_Int    *nparts,
 hypre_F90_ObjRef *grid_ptr,
 hypre_F90_Int    *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridCreate(
              hypre_F90_PassComm   (comm),
              hypre_F90_PassInt    (ndim),
              hypre_F90_PassInt    (nparts),
              hypre_F90_PassObjRef (HYPRE_SStructGrid, grid_ptr) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgriddestroy, HYPRE_SSTRUCTGRIDDESTROY)
(hypre_F90_Obj *grid,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridDestroy(
              hypre_F90_PassObj (HYPRE_SStructGrid, grid) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetExtents
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetextents, HYPRE_SSTRUCTGRIDSETEXTENTS)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetExtents(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt      (part),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetVariables
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetvariables, HYPRE_SSTRUCTGRIDSETVARIABLES)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_Int      *nvars,
 hypre_F90_IntArray *vartypes,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetVariables(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt      (part),
              hypre_F90_PassInt      (nvars),
              hypre_F90_PassIntArray (vartypes) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridAddVariables
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridaddvariables, HYPRE_SSTRUCTGRIDADDVARIABLES)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_IntArray *index,
 hypre_F90_Int      *nvars,
 hypre_F90_IntArray *vartypes,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridAddVariables(
              hypre_F90_PassObj(HYPRE_SStructGrid, grid),
              hypre_F90_PassInt(part),
              hypre_F90_PassIntArray(index),
              hypre_F90_PassInt(nvars),
              hypre_F90_PassIntArray(vartypes));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetFEMOrdering
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetfemordering, HYPRE_SSTRUCTGRIDSETFEMORDERING)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_IntArray *ordering,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetFEMOrdering(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt      (part),
              hypre_F90_PassIntArray (ordering) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetNeighborPart
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetneighborpart, HYPRE_SSTRUCTGRIDSETNEIGHBORPART)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_Int      *nbor_part,
 hypre_F90_IntArray *nbor_ilower,
 hypre_F90_IntArray *nbor_iupper,
 hypre_F90_IntArray *index_map,
 hypre_F90_IntArray *index_dir,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetNeighborPart(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt      (part),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt      (nbor_part),
              hypre_F90_PassIntArray (nbor_ilower),
              hypre_F90_PassIntArray (nbor_iupper),
              hypre_F90_PassIntArray (index_map),
              hypre_F90_PassIntArray (index_dir) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetSharedPart
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetsharedpart, HYPRE_SSTRUCTGRIDSETSHAREDPART)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_IntArray *ilower,
 hypre_F90_IntArray *iupper,
 hypre_F90_IntArray *offset,
 hypre_F90_Int      *shared_part,
 hypre_F90_IntArray *shared_ilower,
 hypre_F90_IntArray *shared_iupper,
 hypre_F90_IntArray *shared_offset,
 hypre_F90_IntArray *index_map,
 hypre_F90_IntArray *index_dir,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetSharedPart(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt      (part),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassIntArray (offset),
              hypre_F90_PassInt      (shared_part),
              hypre_F90_PassIntArray (shared_ilower),
              hypre_F90_PassIntArray (shared_iupper),
              hypre_F90_PassIntArray (shared_offset),
              hypre_F90_PassIntArray (index_map),
              hypre_F90_PassIntArray (index_dir) );
}

/*--------------------------------------------------------------------------
 * *** placeholder ***
 *  HYPRE_SStructGridAddUnstructuredPart
 *--------------------------------------------------------------------------*/

#if 0

void
hypre_F90_IFACE(hypre_sstructgridaddunstructure, HYPRE_SSTRUCTGRIDADDUNSTRUCTURE)
(hypre_F90_Obj *grid,
 hypre_F90_Int *ilower,
 hypre_F90_Int *iupper,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridAddUnstructuredPart(
              hypre_F90_PassObj (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt (ilower),
              hypre_F90_PassInt (iupper) );
}
#endif

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridassemble, HYPRE_SSTRUCTGRIDASSEMBLE)
(hypre_F90_Obj *grid,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridAssemble(
              hypre_F90_PassObj (HYPRE_SStructGrid, grid) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetPeriodic
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetperiodic, HYPRE_SSTRUCTGRIDSETPERIODIC)
(hypre_F90_Obj      *grid,
 hypre_F90_Int      *part,
 hypre_F90_IntArray *periodic,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetPeriodic(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassInt      (part),
              hypre_F90_PassIntArray (periodic) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructGridSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructgridsetnumghost, HYPRE_SSTRUCTGRIDSETNUMGHOST)
(hypre_F90_Obj      *grid,
 hypre_F90_IntArray *num_ghost,
 hypre_F90_Int      *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_SStructGridSetNumGhost(
              hypre_F90_PassObj      (HYPRE_SStructGrid, grid),
              hypre_F90_PassIntArray (num_ghost) );
}

#ifdef __cplusplus
}
#endif
