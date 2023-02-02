/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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
hypre_F90_IFACE(hypre_geterror, HYPRE_GETERROR)
(hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) HYPRE_GetError();
}

void
hypre_F90_IFACE(hypre_checkerror, HYPRE_CHECKERROR)
(hypre_F90_Int *ierr,
 hypre_F90_Int *hypre_error_code,
 hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) HYPRE_CheckError(
                hypre_F90_PassInt(ierr),
                hypre_F90_PassInt(hypre_error_code));
}

void
hypre_F90_IFACE(hypre_geterrorarg, HYPRE_GETERRORARG)
(hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) HYPRE_GetErrorArg();
}

void
hypre_F90_IFACE(hypre_clearallerrors, HYPRE_CLEARALLERRORS)
(hypre_F90_Int *result)
{
   *result = HYPRE_ClearAllErrors();
}

void
hypre_F90_IFACE(hypre_clearerror, HYPRE_CLEARERROR)
(hypre_F90_Int *hypre_error_code,
 hypre_F90_Int *result)
{
   *result = (hypre_F90_Int) HYPRE_ClearError(
                hypre_F90_PassInt(hypre_error_code));
}

#ifdef __cplusplus
}
#endif
