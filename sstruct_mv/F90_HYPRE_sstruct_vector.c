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
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorCreate(
                     (MPI_Comm)             *comm,
                     (HYPRE_SStructGrid)    *grid,
                     (HYPRE_SStructVector *) vector_ptr ) );
}

/*--------------------------------------------------------------------------
  HYPRE_SStructVectorDestroy
  *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectordestroy, HYPRE_SSTRUCTVECTORDESTROY)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorDestroy(
                     (HYPRE_SStructVector) *vector ) );
}

/*---------------------------------------------------------
  HYPRE_SStructVectorInitialize
  * ----------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorinitialize, HYPRE_SSTRUCTVECTORINITIALIZE)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorInitialize(
                     (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetvalues, HYPRE_SSTRUCTVECTORSETVALUES)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *part,
    HYPRE_Int      *index,
    HYPRE_Int      *var,
    double   *value,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorSetValues(
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (double *)             value ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtovalues, HYPRE_SSTRUCTVECTORADDTOVALUES)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *part,
    HYPRE_Int      *index,
    HYPRE_Int      *var,
    double   *value,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorAddToValues(
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (double *)             value ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetvalues, HYPRE_SSTRUCTVECTORGETVALUES)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *part,
    HYPRE_Int      *index,
    HYPRE_Int      *var,
    double   *value,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorGetValues(
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                index,
                     (HYPRE_Int)                 *var,
                     (double *)             value ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetboxvalues, HYPRE_SSTRUCTVECTORSETBOXVALUES)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *part,
    HYPRE_Int      *ilower,
    HYPRE_Int      *iupper,
    HYPRE_Int      *var,
    double   *values,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorSetBoxValues(
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                ilower,
                     (HYPRE_Int *)                iupper,
                     (HYPRE_Int)                 *var,
                     (double *)             values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectoraddtoboxvalu, HYPRE_SSTRUCTVECTORADDTOBOXVALU)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *part,
    HYPRE_Int      *ilower,
    HYPRE_Int      *iupper,
    HYPRE_Int      *var,
    double   *values,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorAddToBoxValues(
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *part,
                     (HYPRE_Int *)                ilower,
                     (HYPRE_Int *)                iupper,
                     (HYPRE_Int)                 *var,
                     (double *)             values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetboxvalues, HYPRE_SSTRUCTVECTORGETBOXVALUES)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *part,
    HYPRE_Int      *ilower,
    HYPRE_Int      *iupper,
    HYPRE_Int      *var,
    double   *values,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorGetBoxValues(
                     (HYPRE_SStructVector ) *vector,
                     (HYPRE_Int)                  *part,
                     (HYPRE_Int *)                 ilower,
                     (HYPRE_Int *)                 iupper,
                     (HYPRE_Int)                  *var,
                     (double *)              values ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorassemble, HYPRE_SSTRUCTVECTORASSEMBLE)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorAssemble(
                     (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGather
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgather, HYPRE_SSTRUCTVECTORGATHER)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorGather(
                     (HYPRE_SStructVector) *vector ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetconstantv, HYPRE_SSTRUCTVECTORSETCONSTANTV)
   (hypre_F90_Obj *vector,
    double   *value,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorSetConstantValues(
                     (HYPRE_SStructVector) *vector,
                     (double)              *value));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetobjecttyp, HYPRE_SSTRUCTVECTORSETOBJECTTYP)
   (hypre_F90_Obj *vector,
    HYPRE_Int      *type,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorSetObjectType(
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *type ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorgetobject, HYPRE_SSTRUCTVECTORGETOBJECT)
   (hypre_F90_Obj *vector,
    hypre_F90_Obj *object,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorGetObject(
                     (HYPRE_SStructVector) *vector,
                     (void **)              object ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorprint, HYPRE_SSTRUCTVECTORPRINT)
   (const char *filename,
    hypre_F90_Obj *vector,
    HYPRE_Int  *all,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorPrint(
                     (const char * )        filename,
                     (HYPRE_SStructVector) *vector,
                     (HYPRE_Int)                 *all ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorcopy, HYPRE_SSTRUCTVECTORCOPY)
   (hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorCopy(
                     (HYPRE_SStructVector) *x,
                     (HYPRE_SStructVector) *y ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorscale, HYPRE_SSTRUCTVECTORSCALE)
   (double   *alpha,
    hypre_F90_Obj *y,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructVectorScale(
                     (double)              *alpha,
                     (HYPRE_SStructVector) *y ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructinnerprod, HYPRE_SSTRUCTINNERPROD)
   (hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    double     *result,
    HYPRE_Int  *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructInnerProd(
                     (HYPRE_SStructVector) *x,
                     (HYPRE_SStructVector) *y,
                     (double *)             result ) );
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructaxpy, HYPRE_SSTRUCTAXPY)
   (double   *alpha,
    hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    HYPRE_Int      *ierr)
{
   *ierr = (HYPRE_Int) (HYPRE_SStructAxpy(
                     (double)              *alpha,
                     (HYPRE_SStructVector) *x,
                     (HYPRE_SStructVector) *y ) );
}
