/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * hypre_IJMatrix Fortran interface
 *
 *****************************************************************************/

#include "./_hypre_IJ_mv.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------------------------------
 * hypre_IJMatrixSetObject
 *--------------------------------------------------------------------------*/

void
hypre_F90_IFACE(hypre_ijmatrixsetobject, HYPRE_IJMATRIXSETOBJECT)
( hypre_F90_Obj *matrix,
  hypre_F90_Obj *object,
  hypre_F90_Int *ierr    )
{
   *ierr = (hypre_F90_Int)
           ( hypre_IJMatrixSetObject(
                hypre_F90_PassObj (HYPRE_IJMatrix, matrix),
                (void *)         *object  ) );
}

#ifdef __cplusplus
}
#endif
