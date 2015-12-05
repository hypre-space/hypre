/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/

#include "_hypre_utilities.h"
#include "fortran.h"

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
