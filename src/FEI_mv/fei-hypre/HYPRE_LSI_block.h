/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_LSI_BlockP interface
 *
 *****************************************************************************/

#ifndef __HYPRE_BLOCKP__
#define __HYPRE_BLOCKP__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "HYPRE_LSI_blkprec.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_BlockPrecondCreate(MPI_Comm comm, HYPRE_Solver *solver);
extern int HYPRE_LSI_BlockPrecondDestroy(HYPRE_Solver solver);
extern int HYPRE_LSI_BlockPrecondSetLumpedMasses(HYPRE_Solver solver,
                                                 int,double *);
extern int HYPRE_LSI_BlockPrecondSetParams(HYPRE_Solver solver, char *params);
extern int HYPRE_LSI_BlockPrecondSetLookup(HYPRE_Solver solver, HYPRE_Lookup *);
extern int HYPRE_LSI_BlockPrecondSetup(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector b,HYPRE_ParVector x);
extern int HYPRE_LSI_BlockPrecondSolve(HYPRE_Solver solver,HYPRE_ParCSRMatrix A,
                                       HYPRE_ParVector b, HYPRE_ParVector x);
extern int HYPRE_LSI_BlockPrecondSetA11Tolerance(HYPRE_Solver solver, double);

#ifdef __cplusplus
}
#endif

#endif

