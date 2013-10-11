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





/******************************************************************************
 *
 * AMG solve routine (Fortran 90 interface)
 *
 *****************************************************************************/

#include "headers.h"


/*--------------------------------------------------------------------------
 * HYPRE_AMGSolve
 *--------------------------------------------------------------------------*/

void      hypre_NAME_C_FOR_FORTRAN(amg_solve)(Solve_err_flag, u_data, f_data,
					n, tol, data)
HYPRE_Int      *Solve_err_flag;
HYPRE_Real   *u_data;
HYPRE_Real   *f_data;
HYPRE_Int      *n;
HYPRE_Real   *tol;
HYPRE_Int      *data;
{
   hypre_Vector   *u;
   hypre_Vector   *f;


   u = hypre_NewVector(u_data, *n);
   f = hypre_NewVector(f_data, *n);

   *Solve_err_flag =  HYPRE_AMGSolve(u, f, *tol, (void *) *data);
}

