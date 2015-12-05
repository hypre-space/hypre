/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.13 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_StructVector interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorcreate, HYPRE_STRUCTVECTORCREATE)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *grid,
     hypre_F90_Obj *vector,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorCreate(
           hypre_F90_PassComm (comm),
           hypre_F90_PassObj (HYPRE_StructGrid, grid),
           hypre_F90_PassObjRef (HYPRE_StructVector, vector)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorDestroy
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectordestroy, HYPRE_STRUCTVECTORDESTROY)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorDestroy(
           hypre_F90_PassObj (HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorInitialize
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorinitialize, HYPRE_STRUCTVECTORINITIALIZE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorInitialize(
           hypre_F90_PassObj (HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorsetvalues, HYPRE_STRUCTVECTORSETVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *grid_index,
     hypre_F90_Dbl *values,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorSetValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (grid_index),
           hypre_F90_PassDbl (values)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorsetboxvalues, HYPRE_STRUCTVECTORSETBOXVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *ilower,
     hypre_F90_IntArray *iupper,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorSetBoxValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (ilower),
           hypre_F90_PassIntArray (iupper),
           hypre_F90_PassDblArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectoraddtovalues, HYPRE_STRUCTVECTORADDTOVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *grid_index,
     hypre_F90_Dbl *values,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorAddToValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (grid_index),
           hypre_F90_PassDbl (values)     ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAddToBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectoraddtoboxvalue, HYPRE_STRUCTVECTORADDTOBOXVALUE)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *ilower,
     hypre_F90_IntArray *iupper,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorAddToBoxValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (ilower),
           hypre_F90_PassIntArray (iupper),
           hypre_F90_PassDblArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorScaleValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorscalevalues, HYPRE_STRUCTVECTORSCALEVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_Dbl *factor,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorScaleValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassDbl (factor) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorgetvalues, HYPRE_STRUCTVECTORGETVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *grid_index,
     hypre_F90_Dbl *values_ptr,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorGetValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (grid_index),
           hypre_F90_PassDblRef (values_ptr) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetBoxValues
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorgetboxvalues, HYPRE_STRUCTVECTORGETBOXVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *ilower,
     hypre_F90_IntArray *iupper,
     hypre_F90_DblArray *values,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorGetBoxValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (ilower),
           hypre_F90_PassIntArray (iupper),
           hypre_F90_PassDblArray (values)  ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorAssemble
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectorassemble, HYPRE_STRUCTVECTORASSEMBLE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorAssemble(
           hypre_F90_PassObj (HYPRE_StructVector, vector) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetNumGhost
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structvectorsetnumghost, HYPRE_STRUCTVECTORSETNUMGHOST)
   ( hypre_F90_Obj *vector,
     hypre_F90_IntArray *num_ghost,
     hypre_F90_Int *ierr      )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorSetNumGhost(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassIntArray (num_ghost) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorCopy
 *--------------------------------------------------------------------------*/
 
void
hypre_F90_IFACE(hypre_structvectorcopy, HYPRE_STRUCTVECTORCOPY)
   ( hypre_F90_Obj *x,
     hypre_F90_Obj *y,
     hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorCopy(
           hypre_F90_PassObj (HYPRE_StructVector, x),
           hypre_F90_PassObj (HYPRE_StructVector, y) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetconstantva, HYPRE_STRUCTVECTORSETCONSTANTVA)
   ( hypre_F90_Obj *vector,
     hypre_F90_Dbl *values,
     hypre_F90_Int *ierr   )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorSetConstantValues(
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassDbl (values) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorGetMigrateCommPkg
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorgetmigratecom, HYPRE_STRUCTVECTORGETMIGRATECOM)
   ( hypre_F90_Obj *from_vector,
     hypre_F90_Obj *to_vector,
     hypre_F90_Obj *comm_pkg,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorGetMigrateCommPkg(
           hypre_F90_PassObj (HYPRE_StructVector, from_vector),
           hypre_F90_PassObj (HYPRE_StructVector, to_vector),
           hypre_F90_PassObjRef (HYPRE_CommPkg, comm_pkg)    ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorMigrate
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_structvectormigrate, HYPRE_STRUCTVECTORMIGRATE)
   ( hypre_F90_Obj *comm_pkg,
     hypre_F90_Obj *from_vector,
     hypre_F90_Obj *to_vector,
     hypre_F90_Int *ierr        )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorMigrate(
           hypre_F90_PassObj (HYPRE_CommPkg, comm_pkg),
           hypre_F90_PassObj (HYPRE_StructVector, from_vector),
           hypre_F90_PassObj (HYPRE_StructVector, to_vector)   ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CommPkgDestroy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_destroycommpkg, HYPRE_DESTROYCOMMPKG)
   ( hypre_F90_Obj *comm_pkg,
     hypre_F90_Int *ierr     )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_CommPkgDestroy(
           hypre_F90_PassObj (HYPRE_CommPkg, comm_pkg) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorPrint
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorprint, HYPRE_STRUCTVECTORPRINT)
   (
      hypre_F90_Obj *vector,
      hypre_F90_Int *all,
      hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
      ( HYPRE_StructVectorPrint(
           "HYPRE_StructVector.out",
           hypre_F90_PassObj (HYPRE_StructVector, vector),
           hypre_F90_PassInt (all)) );
}
