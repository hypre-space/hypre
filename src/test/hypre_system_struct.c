/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.52 $
 ***********************************************************************EHEADER*/

#include "hypre_system.h"

/*--------------------------------------------------------------------------
 * Struct System Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSystemStructHelp( )
{
   hypre_printf("System Struct Options: [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSystemStructCreate(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Int   object_type,
   HYPRE_StructMatrix *A_ptr,
   HYPRE_StructVector *b_ptr,
   HYPRE_StructVector *x_ptr )
{
   return 0;
}

HYPRE_Int
hypre_DriveSystemStructDestroy(
   HYPRE_StructMatrix A,
   HYPRE_StructVector b,
   HYPRE_StructVector x )
{
   return 0;
}
