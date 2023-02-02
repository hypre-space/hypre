/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * HYPRE_StructVectorSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structvectorsetrandomvalu, HYPRE_STRUCTVECTORSETRANDOMVALU)
(hypre_F90_Obj *vector,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_StructVectorSetRandomValues(
                (hypre_StructVector *) vector,
                hypre_F90_PassInt (seed) ));
}


/*--------------------------------------------------------------------------
 * HYPRE_StructSetRandomValues
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetrandomvalues, HYPRE_STRUCTSETRANDOMVALUES)
(hypre_F90_Obj *vector,
 hypre_F90_Int *seed,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( hypre_StructSetRandomValues(
                (hypre_StructVector *) vector,
                hypre_F90_PassInt (seed) ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupInterpreter
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupinterpreter, HYPRE_STRUCTSETUPINTERPRETER)
(hypre_F90_Obj *i,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSetupInterpreter(
                (mv_InterfaceInterpreter *) i ));
}

/*--------------------------------------------------------------------------
 * HYPRE_StructSetupMatvec
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_structsetupmatvec, HYPRE_STRUCTSETUPMATVEC)
(hypre_F90_Obj *mv,
 hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int)
           ( HYPRE_StructSetupMatvec(
                hypre_F90_PassObjRef (HYPRE_MatvecFunctions, mv)));
}

#ifdef __cplusplus
}
#endif
