/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "fortran.h"

#ifdef __cplusplus
extern "C" {
#endif

void
hypre_F90_IFACE(hypre_init, HYPRE_INIT)
   (hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_Init();
}

void
hypre_F90_IFACE(hypre_finalize, HYPRE_FINALIZE)
   (hypre_F90_Int *ierr)
{
   *ierr = (hypre_F90_Int) HYPRE_Finalize();
}

#ifdef __cplusplus
}
#endif
