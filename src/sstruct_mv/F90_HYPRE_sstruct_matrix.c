/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.17 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructMatrix interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixcreate, HYPRE_SSTRUCTMATRIXCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *graph,
    hypre_F90_Obj *matrix_ptr,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObj (HYPRE_SStructGraph, graph),
          hypre_F90_PassObjRef (HYPRE_SStructMatrix, matrix_ptr) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixdestroy, HYPRE_SSTRUCTMATRIXDESTROY)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixDestroy(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixinitialize, HYPRE_SSTRUCTMATRIXINITIALIZE)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixInitialize(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetvalues, HYPRE_SSTRUCTMATRIXSETVALUES)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_Int *var,
    hypre_F90_Int *nentries,
    hypre_F90_IntArray *entries,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixSetValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (nentries),
          hypre_F90_PassIntArray (entries),
          hypre_F90_PassDblArray (values) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtovalues, HYPRE_SSTRUCTMATRIXADDTOVALUES)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_Int *var,
    hypre_F90_Int *nentries,
    hypre_F90_IntArray *entries,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixAddToValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (nentries),
          hypre_F90_PassIntArray (entries),
          hypre_F90_PassDblArray (values)) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddFEMValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddfemvalues, HYPRE_SSTRUCTMATRIXADDFEMVALUES)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixAddFEMValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassDblArray (values)) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetvalues, HYPRE_SSTRUCTMATRIXGETVALUES)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_Int *var,
    hypre_F90_Int *nentries,
    hypre_F90_IntArray *entries,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixGetValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (nentries),
          hypre_F90_PassIntArray (entries),
          hypre_F90_PassDblArray (values)) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetboxvalues, HYPRE_SSTRUCTMATRIXSETBOXVALUES)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *ilower,
    hypre_F90_IntArray *iupper,
    hypre_F90_Int *var,
    hypre_F90_Int *nentries,
    hypre_F90_IntArray *entries,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixSetBoxValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (ilower),
          hypre_F90_PassIntArray (iupper),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (nentries),
          hypre_F90_PassIntArray (entries),
          hypre_F90_PassDblArray (values)));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtoboxvalu, HYPRE_SSTRUCTMATRIXADDTOBOXVALU)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *ilower,
    hypre_F90_IntArray *iupper,
    hypre_F90_Int *var,
    hypre_F90_Int *nentries,
    hypre_F90_IntArray *entries,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixAddToBoxValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (ilower),
          hypre_F90_PassIntArray (iupper),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (nentries),
          hypre_F90_PassIntArray (entries),
          hypre_F90_PassDblArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetboxvalues, HYPRE_SSTRUCTMATRIXGETBOXVALUES)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_IntArray *ilower,
    hypre_F90_IntArray *iupper,
    hypre_F90_Int *var,
    hypre_F90_Int *nentries,
    hypre_F90_IntArray *entries,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixGetBoxValues(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (ilower),
          hypre_F90_PassIntArray (iupper),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (nentries),
          hypre_F90_PassIntArray (entries),
          hypre_F90_PassDblArray (values)));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixassemble, HYPRE_SSTRUCTMATRIXASSEMBLE)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixAssemble(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetsymmetric, HYPRE_SSTRUCTMATRIXSETSYMMETRIC)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *part,
    hypre_F90_Int *var,
    hypre_F90_Int *to_var,
    hypre_F90_Int *symmetric,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixSetSymmetric(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (part),
          hypre_F90_PassInt (var),
          hypre_F90_PassInt (to_var),
          hypre_F90_PassInt (symmetric) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetNSSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetnssymmetr, HYPRE_SSTRUCTMATRIXSETNSSYMMETR)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *symmetric,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixSetNSSymmetric(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (symmetric) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetobjecttyp, HYPRE_SSTRUCTMATRIXSETOBJECTTYP)
   (hypre_F90_Obj *matrix,
    hypre_F90_Int *type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixSetObjectType(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (type) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetobject, HYPRE_SSTRUCTMATRIXGETOBJECT)
   (hypre_F90_Obj *matrix,
    hypre_F90_Obj *object,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixGetObject(
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          (void **)              object )) ;
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixprint, HYPRE_SSTRUCTMATRIXPRINT)
   (char *filename,
    hypre_F90_Obj *matrix,
    hypre_F90_Int *all,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixPrint(
          (char *)           filename,
          hypre_F90_PassObj (HYPRE_SStructMatrix, matrix),
          hypre_F90_PassInt (all) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixmatvec, HYPRE_SSTRUCTMATRIXMATVEC)
   (hypre_F90_Dbl *alpha,
    hypre_F90_Obj *A,
    hypre_F90_Obj *x,
    hypre_F90_Dbl *beta,
    hypre_F90_Obj *y,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructMatrixMatvec(
          hypre_F90_PassDbl (alpha),
          hypre_F90_PassObj (HYPRE_SStructMatrix, A),
          hypre_F90_PassObj (HYPRE_SStructVector, x),
          hypre_F90_PassDbl (beta),
          hypre_F90_PassObj (HYPRE_SStructVector, y) )) ;
}
