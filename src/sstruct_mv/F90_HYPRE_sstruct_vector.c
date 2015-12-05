/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.15 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_SStructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcreate, HYPRE_SSTRUCTVECTORCREATE)
   (hypre_F90_Comm *comm,
    hypre_F90_Obj *grid,
    hypre_F90_Obj *vector_ptr,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorCreate(
          hypre_F90_PassComm (comm),
          hypre_F90_PassObj (HYPRE_SStructGrid, grid),
          hypre_F90_PassObjRef (HYPRE_SStructVector, vector_ptr) ) );
}

/*--------------------------------------------------------------------------
  HYPRE_SStructVectorDestroy
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectordestroy, HYPRE_SSTRUCTVECTORDESTROY)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorDestroy(
          hypre_F90_PassObj (HYPRE_SStructVector, vector) ) );
}

/*---------------------------------------------------------
  HYPRE_SStructVectorInitialize
  * ----------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorinitialize, HYPRE_SSTRUCTVECTORINITIALIZE)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorInitialize(
          hypre_F90_PassObj (HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetvalues, HYPRE_SSTRUCTVECTORSETVALUES)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_Int *var,
    hypre_F90_Dbl *value,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorSetValues(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassInt (var),
          hypre_F90_PassDblRef (value) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtovalues, HYPRE_SSTRUCTVECTORADDTOVALUES)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_Int *var,
    hypre_F90_Dbl *value,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorAddToValues(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassInt (var),
          hypre_F90_PassDblRef (value) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetvalues, HYPRE_SSTRUCTVECTORGETVALUES)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *part,
    hypre_F90_IntArray *index,
    hypre_F90_Int *var,
    hypre_F90_Dbl *value,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorGetValues(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (index),
          hypre_F90_PassInt (var),
          hypre_F90_PassDblRef (value) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetboxvalues, HYPRE_SSTRUCTVECTORSETBOXVALUES)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *part,
    hypre_F90_IntArray *ilower,
    hypre_F90_IntArray *iupper,
    hypre_F90_Int *var,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorSetBoxValues(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (ilower),
          hypre_F90_PassIntArray (iupper),
          hypre_F90_PassInt (var),
          hypre_F90_PassDblArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtoboxvalu, HYPRE_SSTRUCTVECTORADDTOBOXVALU)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *part,
    hypre_F90_IntArray *ilower,
    hypre_F90_IntArray *iupper,
    hypre_F90_Int *var,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorAddToBoxValues(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (ilower),
          hypre_F90_PassIntArray (iupper),
          hypre_F90_PassInt (var),
          hypre_F90_PassDblArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetboxvalues, HYPRE_SSTRUCTVECTORGETBOXVALUES)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *part,
    hypre_F90_IntArray *ilower,
    hypre_F90_IntArray *iupper,
    hypre_F90_Int *var,
    hypre_F90_DblArray *values,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorGetBoxValues(
          (HYPRE_SStructVector ) *vector,
          hypre_F90_PassInt (part),
          hypre_F90_PassIntArray (ilower),
          hypre_F90_PassIntArray (iupper),
          hypre_F90_PassInt (var),
          hypre_F90_PassDblArray (values) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorassemble, HYPRE_SSTRUCTVECTORASSEMBLE)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorAssemble(
          hypre_F90_PassObj (HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGather
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgather, HYPRE_SSTRUCTVECTORGATHER)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorGather(
          hypre_F90_PassObj (HYPRE_SStructVector, vector) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetconstantv, HYPRE_SSTRUCTVECTORSETCONSTANTV)
   (hypre_F90_Obj *vector,
    hypre_F90_Dbl *value,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorSetConstantValues(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassDbl (value)));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetobjecttyp, HYPRE_SSTRUCTVECTORSETOBJECTTYP)
   (hypre_F90_Obj *vector,
    hypre_F90_Int *type,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorSetObjectType(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (type) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetobject, HYPRE_SSTRUCTVECTORGETOBJECT)
   (hypre_F90_Obj *vector,
    hypre_F90_Obj *object,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorGetObject(
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          (void **)              object ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorprint, HYPRE_SSTRUCTVECTORPRINT)
   (char *filename,
    hypre_F90_Obj *vector,
    hypre_F90_Int *all,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorPrint(
          (char * )        filename,
          hypre_F90_PassObj (HYPRE_SStructVector, vector),
          hypre_F90_PassInt (all) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcopy, HYPRE_SSTRUCTVECTORCOPY)
   (hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorCopy(
          hypre_F90_PassObj (HYPRE_SStructVector, x),
          hypre_F90_PassObj (HYPRE_SStructVector, y) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorscale, HYPRE_SSTRUCTVECTORSCALE)
   (hypre_F90_Dbl *alpha,
    hypre_F90_Obj *y,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructVectorScale(
          hypre_F90_PassDbl (alpha),
          hypre_F90_PassObj (HYPRE_SStructVector, y) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructinnerprod, HYPRE_SSTRUCTINNERPROD)
   (hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    hypre_F90_Dbl *result,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructInnerProd(
          hypre_F90_PassObj (HYPRE_SStructVector, x),
          hypre_F90_PassObj (HYPRE_SStructVector, y),
          hypre_F90_PassDblRef (result) ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructaxpy, HYPRE_SSTRUCTAXPY)
   (hypre_F90_Dbl *alpha,
    hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
      (HYPRE_SStructAxpy(
          hypre_F90_PassDbl (alpha),
          hypre_F90_PassObj (HYPRE_SStructVector, x),
          hypre_F90_PassObj (HYPRE_SStructVector, y) ) );
}
