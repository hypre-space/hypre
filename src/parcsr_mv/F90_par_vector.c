/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * par_vector Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

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
 * hypre_SetParVectorConstantValue
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_setparvectorconstantvalue, HYPRE_SETPARVECTORCONSTANTVALUE)
( hypre_F90_Obj *vector,
  hypre_F90_Complex *value,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorSetConstantValues(
                (hypre_ParVector *) *vector,
                hypre_F90_PassComplex (value)   ) );
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
  hypre_F90_Complex *scale,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorScale(
                hypre_F90_PassComplex (scale),
                (hypre_ParVector *) *vector ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_paraxpy, HYPRE_PARAXPY)
( hypre_F90_Complex *a,
  hypre_F90_Obj *x,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int)
           ( hypre_ParVectorAxpy(
                hypre_F90_PassComplex (a),
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
  hypre_F90_Complex *inner_prod,
  hypre_F90_Int *ierr           )
{
   *inner_prod = (hypre_F90_Complex)
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
  hypre_F90_BigIntArray *vec_starts,
  hypre_F90_Obj *par_vector,
  hypre_F90_Int *ierr        )
{
   *par_vector = (hypre_F90_Obj)
                 ( hypre_VectorToParVector(
                      hypre_F90_PassComm (comm),
                      (hypre_Vector *) *vector,
                      hypre_F90_PassBigIntArray (vec_starts) ) );

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
                ( (hypre_ParVector *) *par_vector ));

   *ierr = 0;
}

#ifdef __cplusplus
}
#endif
