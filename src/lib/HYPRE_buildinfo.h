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

#ifndef HYPRE_BUILDINFO_HEADER
#define HYPRE_BUILDINFO_HEADER

#include "HYPRE_utilities.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocates and returns a string with build options in it.
 **/
HYPRE_Int
HYPRE_BuildInfo( char **build_info_ptr );

#ifdef __cplusplus
}
#endif

#endif

