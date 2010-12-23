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
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixCreate(
                     (MPI_Comm)              *comm,
                     (HYPRE_SStructGraph)    *graph,
                     (HYPRE_SStructMatrix *)  matrix_ptr ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixdestroy, HYPRE_SSTRUCTMATRIXDESTROY)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixDestroy(
                     (HYPRE_SStructMatrix) *matrix ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixinitialize, HYPRE_SSTRUCTMATRIXINITIALIZE)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixInitialize(
                     (HYPRE_SStructMatrix) *matrix ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetvalues, HYPRE_SSTRUCTMATRIXSETVALUES)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *index,
    HYPRE_Int  *var,
    HYPRE_Int  *nentries,
    HYPRE_Int  *entries,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixSetValues(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (HYPRE_Int)                 *nentries,
                     (HYPRE_Int *)                entries,
                     (double *)             values ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtovalues, HYPRE_SSTRUCTMATRIXADDTOVALUES)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *index,
    HYPRE_Int  *var,
    HYPRE_Int  *nentries,
    HYPRE_Int  *entries,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixAddToValues(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (HYPRE_Int)                 *nentries,
                     (HYPRE_Int *)                entries,
                     (double *)             values) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddFEMValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddfemvalues, HYPRE_SSTRUCTMATRIXADDFEMVALUES)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *index,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixAddFEMValues(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (double *)             values) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetvalues, HYPRE_SSTRUCTMATRIXGETVALUES)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *index,
    HYPRE_Int  *var,
    HYPRE_Int  *nentries,
    HYPRE_Int  *entries,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixGetValues(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (HYPRE_Int)                 *nentries,
                     (HYPRE_Int *)                entries,
                     (double *)             values) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetboxvalues, HYPRE_SSTRUCTMATRIXSETBOXVALUES)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *ilower,
    HYPRE_Int  *iupper,
    HYPRE_Int  *var,
    HYPRE_Int  *nentries,
    HYPRE_Int  *entries,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixSetBoxValues(
                     (HYPRE_SStructMatrix)  *matrix,
                     (HYPRE_Int)                  *part,
                     (HYPRE_Int *)                 ilower,
                     (HYPRE_Int *)                 iupper,
                     (HYPRE_Int)                  *var,
                     (HYPRE_Int)                  *nentries,
                     (HYPRE_Int *)                 entries,
                     (double *)              values));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixaddtoboxvalu, HYPRE_SSTRUCTMATRIXADDTOBOXVALU)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *ilower,
    HYPRE_Int  *iupper,
    HYPRE_Int  *var,
    HYPRE_Int  *nentries,
    HYPRE_Int  *entries,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixAddToBoxValues(
                     (HYPRE_SStructMatrix)  *matrix,
                     (HYPRE_Int)                  *part,
                     (HYPRE_Int *)                 ilower,
                     (HYPRE_Int *)                 iupper,
                     (HYPRE_Int)                  *var,
                     (HYPRE_Int)                  *nentries,
                     (HYPRE_Int *)                 entries,
                     (double *)              values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetboxvalues, HYPRE_SSTRUCTMATRIXGETBOXVALUES)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *ilower,
    HYPRE_Int  *iupper,
    HYPRE_Int  *var,
    HYPRE_Int  *nentries,
    HYPRE_Int  *entries,
    double     *values,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixGetBoxValues(
                     (HYPRE_SStructMatrix)  *matrix,
                     (HYPRE_Int)                  *part,
                     (HYPRE_Int *)                 ilower,
                     (HYPRE_Int *)                 iupper,
                     (HYPRE_Int)                  *var,
                     (HYPRE_Int)                  *nentries,
                     (HYPRE_Int *)                 entries,
                     (double *)              values));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixassemble, HYPRE_SSTRUCTMATRIXASSEMBLE)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixAssemble(
                     (HYPRE_SStructMatrix) *matrix ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetsymmetric, HYPRE_SSTRUCTMATRIXSETSYMMETRIC)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *part,
    HYPRE_Int  *var,
    HYPRE_Int  *to_var,
    HYPRE_Int  *symmetric,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixSetSymmetric(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int)                 *var,
                     (HYPRE_Int)                 *to_var,
                     (HYPRE_Int)                 *symmetric ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetNSSymmetric
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetnssymmetr, HYPRE_SSTRUCTMATRIXSETNSSYMMETR)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *symmetric,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixSetNSSymmetric(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *symmetric ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixsetobjecttyp, HYPRE_SSTRUCTMATRIXSETOBJECTTYP)
   (hypre_F90_Obj *matrix,
    HYPRE_Int  *type,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixSetObjectType(
                     (HYPRE_SStructMatrix) *matrix,
                     (HYPRE_Int)                 *type ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetobject, HYPRE_SSTRUCTMATRIXGETOBJECT)
   (hypre_F90_Obj *matrix,
    hypre_F90_Obj *object,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixGetObject(
                     (HYPRE_SStructMatrix) *matrix,
                     (void **)              object )) ;
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixGetObject2
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixgetobject2, HYPRE_SSTRUCTMATRIXGETOBJECT2)
   (hypre_F90_Obj *matrix,
    hypre_F90_Obj *object,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixGetObject2(
                     (HYPRE_SStructMatrix) *matrix,
                     (void **)               object )) ;
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixprint, HYPRE_SSTRUCTMATRIXPRINT)
   (const char *filename,
    hypre_F90_Obj *matrix,
    HYPRE_Int  *all,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixPrint(
                     (const char *)           filename,
                     (HYPRE_SStructMatrix)   *matrix,
                     (HYPRE_Int)                   *all ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructMatrixMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructmatrixmatvec, HYPRE_SSTRUCTMATRIXMATVEC)
   (double     *alpha,
    hypre_F90_Obj *A,
    hypre_F90_Obj *x,
    double     *beta,
    hypre_F90_Obj *y,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructMatrixMatvec(
                     (double)              *alpha,
                     (HYPRE_SStructMatrix) *A,
                     (HYPRE_SStructVector) *x,
                     (double)              *beta,
                     (HYPRE_SStructVector) *y )) ;
}
