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
   /* Base solver data structure */
   hypre_Solver            base;
   HYPRE_Int               print_level;
   HYPRE_BigInt            global_num_rows;
   SuperMatrix            *A_dslu;
   hypre_double           *berr;
   dLUstruct_t            *dslu_data_LU;
   SuperLUStat_t          *dslu_data_stat;
   superlu_dist_options_t *dslu_options;
   gridinfo_t             *dslu_data_grid;
   dScalePermstruct_t     *dslu_ScalePermstruct;
   dSOLVEstruct_t         *dslu_solve;
} hypre_DSLUData;

/*--------------------------------------------------------------------------
 * Accessor macros
 *--------------------------------------------------------------------------*/

#define hypre_DSLUDataPrintLevel(data)           ((data) -> print_level)
#define hypre_DSLUDataGlobalNumRows(data)        ((data) -> global_num_rows)
#define hypre_DSLUDataA(data)                    ((data) -> A_dslu)
#define hypre_DSLUDataBerr(data)                 ((data) -> berr)
#define hypre_DSLUDataLU(data)                   ((data) -> dslu_data_LU)
#define hypre_DSLUDataStat(data)                 ((data) -> dslu_data_stat)
#define hypre_DSLUDataOptions(data)              ((data) -> dslu_options)
#define hypre_DSLUDataGrid(data)                 ((data) -> dslu_data_grid)
#define hypre_DSLUDataScalePermstruct(data)      ((data) -> dslu_ScalePermstruct)
#define hypre_DSLUDataSolve(data)                ((data) -> dslu_solve)

#endif
