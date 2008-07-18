/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/


#include "_hypre_utilities.h"
#include "fortran.h"

void
hypre_F90_IFACE(hypre_geterror, HYPRE_GETERROR)(int *result)
{
   *result = (int) HYPRE_GetError();
}

void
hypre_F90_IFACE(hypre_checkerror, HYPRE_CHECKERROR)(int *ierr,
                                                    int *hypre_error_code,
                                                    int *result)
{
   *result = (int) HYPRE_CheckError(*ierr, *hypre_error_code);
}

void
hypre_F90_IFACE(hypre_geterrorarg, HYPRE_GETERRORARG)(int *result)
{
   *result = (int) HYPRE_GetErrorArg();
}
