/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_StructMatrix interface
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixcreate, HYPRE_STRUCTMATRIXCREATE)
(hypre_F90_Comm *comm,
 hypre_F90_Obj *grid,
 hypre_F90_Obj *stencil,
 hypre_F90_Obj *matrix,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixCreate(
              hypre_F90_PassComm (comm),
              hypre_F90_PassObj (HYPRE_StructGrid, grid),
              hypre_F90_PassObj (HYPRE_StructStencil, stencil),
              hypre_F90_PassObjRef (HYPRE_StructMatrix, matrix)   );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixdestroy, HYPRE_STRUCTMATRIXDESTROY)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixDestroy(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixinitialize, HYPRE_STRUCTMATRIXINITIALIZE)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixInitialize(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetvalues, HYPRE_STRUCTMATRIXSETVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (grid_index),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetboxvalues, HYPRE_STRUCTMATRIXSETBOXVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetBoxValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetboxvalues, HYPRE_STRUCTMATRIXGETBOXVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixGetBoxValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetconstantva, HYPRE_STRUCTMATRIXSETCONSTANTVA)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetConstantValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixaddtovalues, HYPRE_STRUCTMATRIXADDTOVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *grid_index,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixAddToValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (grid_index),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)           );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixaddtoboxvalues, HYPRE_STRUCTMATRIXADDTOBOXVALUES)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *ilower,
  hypre_F90_IntArray *iupper,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixAddToBoxValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (ilower),
              hypre_F90_PassIntArray (iupper),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAddToConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixaddtoconstant, HYPRE_STRUCTMATRIXADDTOCONSTANT)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *num_stencil_indices,
  hypre_F90_IntArray *stencil_indices,
  hypre_F90_ComplexArray *values,
  hypre_F90_Int *ierr              )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetConstantValues(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (num_stencil_indices),
              hypre_F90_PassIntArray (stencil_indices),
              hypre_F90_PassComplexArray (values)        );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixassemble, HYPRE_STRUCTMATRIXASSEMBLE)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixAssemble(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetNumGhost
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetnumghost, HYPRE_STRUCTMATRIXSETNUMGHOST)
( hypre_F90_Obj *matrix,
  hypre_F90_IntArray *num_ghost,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetNumGhost(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassIntArray (num_ghost) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixGetGrid
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixgetgrid, HYPRE_STRUCTMATRIXGETGRID)
( hypre_F90_Obj *matrix,
  hypre_F90_Obj *grid,
  hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixGetGrid(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassObjRef (HYPRE_StructGrid, grid) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetsymmetric, HYPRE_STRUCTMATRIXSETSYMMETRIC)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *symmetric,
  hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetSymmetric(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (symmetric) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixSetConstantEntries
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixsetconstanten, HYPRE_STRUCTMATRIXSETCONSTANTEN)
( hypre_F90_Obj *matrix,
  hypre_F90_Int *nentries,
  hypre_F90_IntArray *entries,
  hypre_F90_Int *ierr                )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixSetConstantEntries(
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (nentries),
              hypre_F90_PassIntArray (entries)           );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixprint, HYPRE_STRUCTMATRIXPRINT)
(
   hypre_F90_Obj *matrix,
   hypre_F90_Int *all,
   hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixPrint(
              "HYPRE_StructMatrix.out",
              hypre_F90_PassObj (HYPRE_StructMatrix, matrix),
              hypre_F90_PassInt (all));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structmatrixmatvec, HYPRE_STRUCTMATRIXMATVEC)
( hypre_F90_Complex *alpha,
  hypre_F90_Obj *A,
  hypre_F90_Obj *x,
  hypre_F90_Complex *beta,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int) HYPRE_StructMatrixMatvec(
              hypre_F90_PassComplex (alpha),
              hypre_F90_PassObj (HYPRE_StructMatrix, A),
              hypre_F90_PassObj (HYPRE_StructVector, x),
              hypre_F90_PassComplex (beta),
              hypre_F90_PassObj (HYPRE_StructVector, y)  );
}

#ifdef __cplusplus
}
#endif
