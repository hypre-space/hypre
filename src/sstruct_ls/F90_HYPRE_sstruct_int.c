/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_SStructInt Fortran interface
 *
 *****************************************************************************/

#include "_hypre_sstruct_ls.h"
#include "fortran.h"
#include "HYPRE_MatvecFunctions.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 *  HYPRE_SStructPVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructpvectorsetrandomva, HYPRE_SSTRUCTPVECTORSETRANDOMVA)
(hypre_F90_Obj *pvector,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_SStructPVectorSetRandomValues(
                (hypre_SStructPVector *) pvector,
                hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructvectorsetrandomval, HYPRE_SSTRUCTVECTORSETRANDOMVAL)
(hypre_F90_Obj *vector,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_SStructVectorSetRandomValues(
                (hypre_SStructVector *) vector,
                hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetrandomvalues, HYPRE_SSTRUCTSETRANDOMVALUES)
(hypre_F90_Obj *v,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_SStructSetRandomValues(
                (void *) v, hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructVectorSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetupinterpreter, HYPRE_SSTRUCTSETUPINTERPRETER)
(hypre_F90_Obj *i,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 *  HYPRE_SStructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_sstructsetupmatvec, HYPRE_SSTRUCTSETUPMATVEC)
(hypre_F90_Obj *mv,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_SStructSetupMatvec(
                hypre_F90_PassObjRef (HYPRE_MatvecFunctions, mv)));
}

#ifdef __cplusplus
}
#endif
