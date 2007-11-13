/*BHEADER**********************************************************************
 * Copyright (c) 2007, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision$
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

