/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorcreate, HYPRE_IJVECTORCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Int *jlower,
     hypre_F90_Int *jupper,
     hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassInt (jlower),
           hypre_F90_PassInt (jupper), 
           hypre_F90_PassObjRef (HYPRE_IJVector, vector)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectordestroy, HYPRE_IJVECTORDESTROY)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorDestroy(
           hypre_F90_PassObj (HYPRE_IJVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorinitialize, HYPRE_IJVECTORINITIALIZE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorInitialize(
           hypre_F90_PassObj (HYPRE_IJVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorsetvalues, HYPRE_IJVECTORSETVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *num_values,
     hypre_F90_IntArray *indices,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorSetValues(
           hypre_F90_PassObj (HYPRE_IJVector, vector),
           hypre_F90_PassInt (num_values),
           hypre_F90_PassIntArray (indices),
           hypre_F90_PassDblArray (values)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAddToValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectoraddtovalues, HYPRE_IJVECTORADDTOVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *num_values,
     hypre_F90_IntArray *indices,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorAddToValues(
           hypre_F90_PassObj (HYPRE_IJVector, vector),
           hypre_F90_PassInt (num_values),
           hypre_F90_PassIntArray (indices),
           hypre_F90_PassDblArray (values)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorAssemble
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorassemble, HYPRE_IJVECTORASSEMBLE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorAssemble(
           hypre_F90_PassObj (HYPRE_IJVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetvalues, HYPRE_IJVECTORGETVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *num_values,
     hypre_F90_IntArray *indices,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorGetValues(
           hypre_F90_PassObj (HYPRE_IJVector, vector),
           hypre_F90_PassInt (num_values),
           hypre_F90_PassIntArray (indices),
           hypre_F90_PassDblArray (values)      ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetMaxOffProcElmts
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetmaxoffprocelmt, HYPRE_IJVECTORSETMAXOFFPROCELMT)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *max_off_proc_elmts,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int) 
      ( HYPRE_IJVectorSetMaxOffProcElmts(
           hypre_F90_PassObj (HYPRE_IJVector, vector),
           hypre_F90_PassInt (max_off_proc_elmts) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorSetObjectType
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_ijvectorsetobjecttype, HYPRE_IJVECTORSETOBJECTTYPE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *type,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_IJVectorSetObjectType(
           hypre_F90_PassObj (HYPRE_IJVector, vector),
           hypre_F90_PassInt (type)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObjectType
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetobjecttype, HYPRE_IJVECTORGETOBJECTTYPE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *type,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      (HYPRE_IJVectorGetObjectType(
          hypre_F90_PassObj (HYPRE_IJVector, vector),
          hypre_F90_PassIntRef (type)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetLocalRange
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetlocalrange, HYPRE_IJVECTORGETLOCALRANGE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *jlower,
     hypre_F90_Int *jupper,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      (HYPRE_IJVectorGetLocalRange(
          hypre_F90_PassObj (HYPRE_IJVector, vector),
          hypre_F90_PassIntRef (jlower),
          hypre_F90_PassIntRef (jupper)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorGetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorgetobject, HYPRE_IJVECTORGETOBJECT)
   ( hypre_F90_Obj *vector,
     hypre_F90_Obj *object,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      (HYPRE_IJVectorGetObject(
          hypre_F90_PassObj (HYPRE_IJVector, vector),
          (void **)         object  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorRead
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorread, HYPRE_IJVECTORREAD)
   ( char     *filename,
     hypre_F90_Comm *comm,
     hypre_F90_Int *object_type,
     hypre_F90_Obj *vector,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      (HYPRE_IJVectorRead(
          (char *)            filename,
          hypre_F90_PassComm (comm),
          hypre_F90_PassInt (object_type),
          hypre_F90_PassObjRef (HYPRE_IJVector, vector)       ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_IJVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijvectorprint, HYPRE_IJVECTORPRINT)
   ( hypre_F90_Obj *vector,
     char     *filename,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      (HYPRE_IJVectorPrint(
          hypre_F90_PassObj (HYPRE_IJVector, vector),
          (char *)          filename ) );
}
