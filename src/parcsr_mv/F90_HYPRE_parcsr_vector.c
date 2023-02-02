/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_ParVector Fortran interface
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorcreate, HYPRE_PARVECTORCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_BigInt *global_size,
  hypre_F90_BigIntArray *partitioning,
  hypre_F90_Obj *vector,
  hypre_F90_Int *ierr )
{
   *ierr = (hypre_F90_Int) HYPRE_ParVectorCreate(
              hypre_F90_PassComm (comm),
              hypre_F90_PassBigInt (global_size),
              hypre_F90_PassBigIntArray (partitioning),
              hypre_F90_PassObjRef (HYPRE_ParVector, vector) );

}

/*--------------------------------------------------------------------------
 * HYPRE_ParMultiVectorCreate
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parmultivectorcreate, HYPRE_PARMULTIVECTORCREATE)
( hypre_F90_Comm *comm,
  hypre_F90_BigInt *global_size,
  hypre_F90_BigIntArray *partitioning,
  hypre_F90_Int *number_vectors,
  hypre_F90_Obj *vector,
  hypre_F90_Int *ierr          )
{
   *ierr = (hypre_F90_Int) HYPRE_ParMultiVectorCreate(
              hypre_F90_PassComm (comm),
              hypre_F90_PassBigInt (global_size),
              hypre_F90_PassBigIntArray (partitioning),
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

   c_file_name = hypre_CTAlloc(char,  *fort_file_name_size, HYPRE_MEMORY_HOST);

   for (i = 0; i < *fort_file_name_size; i++)
   {
      c_file_name[i] = fort_file_name[i];
   }

   *ierr = (hypre_F90_Int)
           ( HYPRE_ParVectorPrint(
                hypre_F90_PassObj (HYPRE_ParVector, vector),
                (char *)           c_file_name ) );

   hypre_TFree(c_file_name, HYPRE_MEMORY_HOST);

}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorSetConstantValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectorsetconstantvalue, HYPRE_PARVECTORSETCONSTANTVALUE)
( hypre_F90_Obj *vector,
  hypre_F90_Complex *value,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParVectorSetConstantValues(
                hypre_F90_PassObj (HYPRE_ParVector, vector),
                hypre_F90_PassComplex (value)) );
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
( hypre_F90_Complex *value,
  hypre_F90_Obj *x,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParVectorScale(
                hypre_F90_PassComplex (value),
                hypre_F90_PassObj (HYPRE_ParVector, x) ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParVectorAxpy
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_parvectoraxpy, HYPRE_PARVECTORAXPY)
( hypre_F90_Complex *value,
  hypre_F90_Obj *x,
  hypre_F90_Obj *y,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParVectorAxpy(
                hypre_F90_PassComplex (value),
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
 hypre_F90_Complex *prod,
 hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_ParVectorInnerProd(
                hypre_F90_PassObj (HYPRE_ParVector, x),
                hypre_F90_PassObj (HYPRE_ParVector, y),
                hypre_F90_PassRealRef (prod) ) );
}

#ifdef __cplusplus
}
#endif
