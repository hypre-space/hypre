/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_LSI_Uzawa interface
 *
 *****************************************************************************/

#ifndef __HYPRE_UZAWA__
#define __HYPRE_UZAWA__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "HYPRE.h"
#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/_hypre_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LSI_UZAWA.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_UzawaCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_LSI_UzawaDestroy(HYPRE_Solver solver);
extern int HYPRE_LSI_UzawaSetMaxIterations(HYPRE_Solver solver, int iter);
extern int HYPRE_LSI_UzawaSetTolerance(HYPRE_Solver solver, double tol);
extern int HYPRE_LSI_UzawaSetParams(HYPRE_Solver solver, char *params);
extern int HYPRE_LSI_UzawaGetNumIterations(HYPRE_Solver solver, int *iter);
extern int HYPRE_LSI_UzawaSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_LSI_UzawaSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x);

#ifdef __cplusplus
}
#endif

#endif

