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
 * IJ System Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSystemIJHelp( )
{
   hypre_printf("System IJ Options: [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSystemIJCreate(
   char       *argv[],
   HYPRE_Int   argi,
   HYPRE_Int   argn,
   HYPRE_Int   object_type,
   HYPRE_IJMatrix *A_ptr,
   HYPRE_IJVector *b_ptr,
   HYPRE_IJVector *x_ptr )
{
   return 0;
}

HYPRE_Int
hypre_DriveSystemIJDestroy(
   HYPRE_IJMatrix A,
   HYPRE_IJVector b,
   HYPRE_IJVector x )
{
   return 0;
}

