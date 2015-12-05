/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * par_vector Fortran interface
 *
 *****************************************************************************/

#include "headers.h"
#include "fortran.h"

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetDataOwner
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectordataowner, HYPRE_SETPARVECTORDATAOWNER)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *owns_data,
     hypre_F90_Int *ierr       )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorSetDataOwner(
           (hypre_ParVector *) *vector,
           hypre_F90_PassInt (owns_data) ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorPartitioningO
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorpartitioningo, HYPRE_SETPARVECTORPARTITIONINGO)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *owns_partitioning,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorSetPartitioningOwner(
           (hypre_ParVector *) *vector,
           hypre_F90_PassInt (owns_partitioning) ) );
}

/*--------------------------------------------------------------------------
 * hypre_SetParVectorConstantValue 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorconstantvalue, HYPRE_SETPARVECTORCONSTANTVALUE)
   ( hypre_F90_Obj *vector,
     hypre_F90_Dbl *value,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorSetConstantValues(
           (hypre_ParVector *) *vector,
           hypre_F90_PassDbl (value)   ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorSetRandomValues 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_setparvectorrandomvalues, HYPRE_SETPARVECTORRANDOMVALUES)
   ( hypre_F90_Obj *vector,
     hypre_F90_Int *seed,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorSetRandomValues(
           (hypre_ParVector *) *vector,
           hypre_F90_PassInt (seed)    ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorCopy 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_copyparvector, HYPRE_COPYPARVECTOR)
   ( hypre_F90_Obj *x,
     hypre_F90_Obj *y,
     hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorCopy(
           (hypre_ParVector *) *x,
           (hypre_ParVector *) *y  ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorScale 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_scaleparvector, HYPRE_SCALEPARVECTOR)
   ( hypre_F90_Obj *vector,
     hypre_F90_Dbl *scale,
     hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorScale(
           hypre_F90_PassDbl (scale),
           (hypre_ParVector *) *vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorAxpy 
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_paraxpy, HYPRE_PARAXPY)
   ( hypre_F90_Dbl *a,
     hypre_F90_Obj *x,
     hypre_F90_Obj *y,
     hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
      ( hypre_ParVectorAxpy(
           hypre_F90_PassDbl (a),
           (hypre_ParVector *) *x,
           (hypre_ParVector *) *y  ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorInnerProd
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parinnerprod, HYPRE_PARINNERPROD)
   ( hypre_F90_Obj *x,
     hypre_F90_Obj *y,
     hypre_F90_Dbl *inner_prod, 
     hypre_F90_Int *ierr           )
{
   *inner_prod = (hypre_F90_Dbl)
      ( hypre_ParVectorInnerProd(
           (hypre_ParVector *) *x,
           (hypre_ParVector *) *y  ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_VectorToParVector
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_vectortoparvector, HYPRE_VECTORTOPARVECTOR)
   ( hypre_F90_Comm *comm,
     hypre_F90_Obj *vector,
     hypre_F90_IntArray *vec_starts,
     hypre_F90_Obj *par_vector,
     hypre_F90_Int *ierr        )
{
   *par_vector = (hypre_F90_Obj)
      ( hypre_VectorToParVector(
           hypre_F90_PassComm (comm),
           (hypre_Vector *) *vector,
           hypre_F90_PassIntArray (vec_starts) ) );

   *ierr = 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorToVectorAll
 *--------------------------------------------------------------------------*/

void 
hypre_F90_IFACE(hypre_parvectortovectorall, HYPRE_PARVECTORTOVECTORALL)
   ( hypre_F90_Obj *par_vector,
     hypre_F90_Obj *vector,
     hypre_F90_Int *ierr        )
{
   *vector = (hypre_F90_Obj)(
      hypre_ParVectorToVectorAll
      ( (hypre_ParVector *) *par_vector ) );

   *ierr = 0;
}
