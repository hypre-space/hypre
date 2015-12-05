/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 1.7 $
 ***********************************************************************EHEADER*/

/*--------------------------------------------------------------------------
 * Header file for test drivers
 *--------------------------------------------------------------------------*/
#ifndef HYPRE_TEST_INCLUDES
#define HYPRE_TEST_INCLUDES

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "HYPRE_krylov.h"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_struct_ls.h"
#include "HYPRE_sstruct_ls.h"
 
#define HYPRE_BICGSTAB   99100
#define HYPRE_BOOMERAMG  99110
#define HYPRE_CGNR       99120
#define HYPRE_DIAGSCALE  99130
#define HYPRE_EUCLID     99140
#define HYPRE_GMRES      99150
#define HYPRE_GSMG       99155
#define HYPRE_HYBRID     99160
#define HYPRE_JACOBI     99170
#define HYPRE_PARASAILS  99180
#define HYPRE_PCG        99190
#define HYPRE_PFMG_1     99200
#define HYPRE_PILUT      99210
#define HYPRE_SCHWARZ    99220
#define HYPRE_SMG_1      99230
#define HYPRE_SPARSEMSG  99240
#define HYPRE_SPLIT      99250
#define HYPRE_SPLITPFMG  99260
#define HYPRE_SPLITSMG   99270
#define HYPRE_SYSPFMG    99280

/****************************************************************************
 * Prototypes for testing routines
 ***************************************************************************/
HYPRE_Int hypre_set_precond(HYPRE_Int matrix_id, HYPRE_Int solver_id, HYPRE_Int precond_id, 
                      void *solver, void *precond);

HYPRE_Int hypre_set_precond_params(HYPRE_Int precond_id, void *precond);

HYPRE_Int hypre_destroy_precond(HYPRE_Int precond_id, void *precond);

/****************************************************************************
 * Variables for testing routines
 ***************************************************************************/
HYPRE_Int      k_dim = 5;
HYPRE_Int      gsmg_samples = 5;
HYPRE_Int      poutdat = 1;
HYPRE_Int      hybrid = 1;
HYPRE_Int      coarsen_type = 6;
HYPRE_Int      measure_type = 0;
HYPRE_Int      smooth_type = 6;
HYPRE_Int      num_functions = 1;
HYPRE_Int      smooth_num_levels = 0;
HYPRE_Int      smooth_num_sweeps = 1;
HYPRE_Int      num_sweep = 1;
HYPRE_Int      max_levels = 25;
HYPRE_Int      variant = 0;
HYPRE_Int      overlap = 1;
HYPRE_Int      domain_type = 2;
HYPRE_Int      nonzeros_to_keep = -1;

HYPRE_Int      interp_type; 
HYPRE_Int      cycle_type;
HYPRE_Int      relax_default;
HYPRE_Int     *dof_func;
HYPRE_Int     *num_grid_sweeps;  
HYPRE_Int     *grid_relax_type;   
HYPRE_Int    **grid_relax_points;

double   tol = 1.e-8;
double   pc_tol = 0.;
double   drop_tol = -1.;
double   max_row_sum = 1.;
double   schwarz_rlx_weight = 1.;
double   sai_threshold = 0.1;
double   sai_filter = 0.1;

double   strong_threshold;
double   trunc_factor;
double  *relax_weight; 
double  *omega;

#endif
