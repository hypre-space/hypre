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

#ifndef HYPRE_DRIVE_SYSTEM_HEADER
#define HYPRE_DRIVE_SYSTEM_HEADER

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "_hypre_utilities.h"
#include "HYPRE_sstruct_mv.h"
#include "HYPRE_struct_mv.h"
#include "HYPRE_IJ_mv.h"

/*--------------------------------------------------------------------------
 * Prototypes for driver system
 *--------------------------------------------------------------------------*/

/* hypre_system_struct.c */

HYPRE_Int
hypre_DriveSystemStructHelp();

HYPRE_Int
hypre_DriveSystemStructCreate(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Int   object_type,
   HYPRE_StructMatrix *A_ptr,
   HYPRE_StructVector *b_ptr,
   HYPRE_StructVector *x_ptr);

HYPRE_Int
hypre_DriveSystemStructDestroy(
   HYPRE_StructMatrix A,
   HYPRE_StructVector b,
   HYPRE_StructVector x);

/* hypre_system_sstruct.c */

HYPRE_Int
hypre_DriveSystemSStructHelp();

HYPRE_Int
hypre_DriveSystemSStructCreate(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Int   object_type,
   HYPRE_SStructMatrix *A_ptr,
   HYPRE_SStructVector *b_ptr,
   HYPRE_SStructVector *x_ptr);

HYPRE_Int
hypre_DriveSystemSStructDestroy(
   HYPRE_SStructMatrix A,
   HYPRE_SStructVector b,
   HYPRE_SStructVector x);

/* hypre_system_ij.c */

HYPRE_Int
hypre_DriveSystemIJHelp();

HYPRE_Int
hypre_DriveSystemIJCreate(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Int   object_type,
   HYPRE_IJMatrix *A_ptr,
   HYPRE_IJVector *b_ptr,
   HYPRE_IJVector *x_ptr);

HYPRE_Int
hypre_DriveSystemIJDestroy(
   HYPRE_IJMatrix A,
   HYPRE_IJVector b,
   HYPRE_IJVector x);

#endif
