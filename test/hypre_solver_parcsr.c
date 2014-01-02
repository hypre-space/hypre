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

#include "hypre_solver.h"

/*--------------------------------------------------------------------------
 * ParCSR Solver Options
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_DriveSolverParCSRHelp( )
{
   hypre_printf("Solver ParCSR Options: [<options>]\n");
   hypre_printf("\n");

   return 0;
}

HYPRE_Int
hypre_DriveSolveParCSR(
   char      *argv[],
   HYPRE_Int  argi,
   HYPRE_Int  argn,
   HYPRE_ParCSRMatrix A,
   HYPRE_ParVector    b,
   HYPRE_ParVector    x )
{
   return 0;
}
