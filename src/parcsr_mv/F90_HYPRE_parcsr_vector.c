/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParVector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcreate, HYPRE_PARVECTORCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Int *global_size,
     hypre_F90_IntArray *partitioning,
     hypre_F90_Obj *vector,
     hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int) HYPRE_ParVectorCreate(
      hypre_F90_PassComm (comm),
      hypre_F90_PassInt (global_size),
      hypre_F90_PassIntArray (partitioning),
      hypre_F90_PassObjRef (HYPRE_ParVector, vector) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parmultivectorcreate, HYPRE_PARMULTIVECTORCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Int *global_size,
     hypre_F90_IntArray *partitioning,
     hypre_F90_Int *number_vectors,
     hypre_F90_Obj *vector,
     hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int) HYPRE_ParMultiVectorCreate(
      hypre_F90_PassComm (comm),
      hypre_F90_PassInt (global_size),
      hypre_F90_PassIntArray (partitioning),
      hypre_F90_PassInt (number_vectors),
      hypre_F90_PassObjRef (HYPRE_ParVector, vector) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectordestroy, HYPRE_PARVECTORDESTROY)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorDestroy(
           hypre_F90_PassObj (HYPRE_ParVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinitialize, HYPRE_PARVECTORINITIALIZE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorInitialize(
           hypre_F90_PassObj (HYPRE_ParVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorRead
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectorread, HYPRE_PARVECTORREAD)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *vector,
     char     *file_name,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorRead(
           hypre_F90_PassComm (comm),
           (char *)    file_name,
           hypre_F90_PassObjRef (HYPRE_ParVector, vector) ) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorprint, HYPRE_PARVECTORPRINT)
   ( hypre_F90_Obj *vector,
     char     *fort_file_name,
     hypre_F90_Int *fort_file_name_size,
     hypre_F90_Int *ierr       )
{
   HYPRE_Int i;
   char *c_file_name;

   c_file_name = hypre_CTAlloc(char, *fort_file_name_size);

   for (i = 0; i < *fort_file_name_size; i++)
   {
      c_file_name[i] = fort_file_name[i];
   }

   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorPrint(
           hypre_F90_PassObj (HYPRE_ParVector, vector),
           (char *)           c_file_name ) );

   hypre_TFree(c_file_name);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetconstantvalue, HYPRE_PARVECTORSETCONSTANTVALUE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Dbl *value,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorSetConstantValues(
           hypre_F90_PassObj (HYPRE_ParVector, vector),
           hypre_F90_PassDbl (value)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetrandomvalues, HYPRE_PARVECTORSETRANDOMVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *seed,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorSetRandomValues(
           hypre_F90_PassObj (HYPRE_ParVector, vector),
           hypre_F90_PassInt (seed)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCopy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcopy, HYPRE_PARVECTORCOPY)
   ( hypre_F90_Obj *x,
     hypre_F90_Obj *y,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorCopy(
           hypre_F90_PassObj (HYPRE_ParVector, x),
           hypre_F90_PassObj (HYPRE_ParVector, y)) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCloneShallow
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcloneshallow, HYPRE_PARVECTORCLONESHALLOW)
   ( hypre_F90_Obj *x,
     hypre_F90_Obj *xclone,
     hypre_F90_Int *ierr    )
{
   *xclone = (hypre_F90_Obj)
      ( HYPRE_ParVectorCloneShallow(
           hypre_F90_PassObj (HYPRE_ParVector, x) ) );
   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorScale
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorscale, HYPRE_PARVECTORSCALE)
   ( hypre_F90_Dbl *value,
     hypre_F90_Obj *x,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorScale(
           hypre_F90_PassDbl (value),
           hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectoraxpy, HYPRE_PARVECTORAXPY)
   ( hypre_F90_Dbl *value,
     hypre_F90_Obj *x,
     hypre_F90_Obj *y,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorAxpy(
           hypre_F90_PassDbl (value),
           hypre_F90_PassObj (HYPRE_ParVector, x),
           hypre_F90_PassObj (HYPRE_ParVector, y) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorinnerprod, HYPRE_PARVECTORINNERPROD)
   (hypre_F90_Obj *x,
    hypre_F90_Obj *y,
    hypre_F90_Dbl *prod,
    hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_ParVectorInnerProd(
           hypre_F90_PassObj (HYPRE_ParVector, x),
           hypre_F90_PassObj (HYPRE_ParVector, y),
           hypre_F90_PassDblRef (prod) ) );
}
