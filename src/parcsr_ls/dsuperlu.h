/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#ifndef hypre_DSLU_DATA_HEADER
#define hypre_DSLU_DATA_HEADER

#include "superlu_ddefs.h"

/*--------------------------------------------------------------------------
 * hypre_DSLUData
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_BigInt            global_num_rows;
   SuperMatrix             A_dslu;
   hypre_double           *berr;
   dLUstruct_t             dslu_data_LU;
   SuperLUStat_t           dslu_data_stat;
   superlu_dist_options_t  dslu_options;
   gridinfo_t              dslu_data_grid;
   dScalePermstruct_t      dslu_ScalePermstruct;
   dSOLVEstruct_t          dslu_solve;
}
hypre_DSLUData;

#endif
