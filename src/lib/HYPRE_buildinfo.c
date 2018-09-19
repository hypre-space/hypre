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
#include "_hypre_buildinfo.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BuildInfo( char **build_info_ptr )
{
   HYPRE_Int  len = 30;
   char      *build_info;

   /* compute string length */
   len += strlen(HYPRE_BUILD_INFO);

   build_info = hypre_CTAlloc(char, len, HYPRE_MEMORY_HOST);

   hypre_sprintf(build_info, "Build info:\n\n%s", HYPRE_BUILD_INFO);

   *build_info_ptr = build_info;

   return hypre_error_flag;
}

