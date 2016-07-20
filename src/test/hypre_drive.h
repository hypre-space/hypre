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

#ifndef HYPRE_DRIVE_HEADER
#define HYPRE_DRIVE_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"

/*--------------------------------------------------------------------------
 * Prototypes for driver
 *--------------------------------------------------------------------------*/

HYPRE_Int
ArgNext(
   char        *argv[],
   HYPRE_Int   *argi_ptr,
   HYPRE_Int   *argn_ptr );

HYPRE_Int
ArgStripBraces(
   char        *argv[],
   HYPRE_Int    argi,
   HYPRE_Int    argn,
   char      ***argv_ptr,
   HYPRE_Int   *argi_ptr,
   HYPRE_Int   *argn_ptr );

#endif
