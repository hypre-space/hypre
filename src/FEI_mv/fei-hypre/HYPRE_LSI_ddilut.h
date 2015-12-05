/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * HYPRE_DDILUT interface
 *
 *****************************************************************************/

#ifndef __HYPRE_DDILUT__
#define __HYPRE_DDILUT__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern int HYPRE_LSI_DDIlutCreate( MPI_Comm comm, HYPRE_Solver *solver );
extern int HYPRE_LSI_DDIlutDestroy( HYPRE_Solver solver );
extern int HYPRE_LSI_DDIlutSetFillin( HYPRE_Solver solver, double fillin);
extern int HYPRE_LSI_DDIlutSetOutputLevel( HYPRE_Solver solver, int level);
extern int HYPRE_LSI_DDIlutSetDropTolerance( HYPRE_Solver solver, double thresh);
extern int HYPRE_LSI_DDIlutSetOverlap( HYPRE_Solver solver );
extern int HYPRE_LSI_DDIlutSetReorder( HYPRE_Solver solver );
extern int HYPRE_LSI_DDIlutSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b,   HYPRE_ParVector x );
extern int HYPRE_LSI_DDIlutSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                  HYPRE_ParVector b,   HYPRE_ParVector x );

#ifdef __cplusplus
}
#endif

#endif

