/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * ParaSails.h header file.
 *
 *****************************************************************************/

#include "Matrix.h"
#include "Numbering.h"
#include "PrunedRows.h"
#include "StoredRows.h"
#include "RowPatt.h"
#include "LoadBal.h"

#ifndef _PARASAILS_H
#define _PARASAILS_H

//#define PARASAILS_DEBUG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    HYPRE_Int        symmetric;
    HYPRE_Real thresh;
    HYPRE_Int        num_levels;
    HYPRE_Real filter;
    HYPRE_Real loadbal_beta;

    HYPRE_Real cost;          /* cost for this processor */
    HYPRE_Real setup_pattern_time;
    HYPRE_Real setup_values_time;

    Numbering *numb;
    Matrix    *M;             /* preconditioner */

    MPI_Comm   comm;
    HYPRE_Int        beg_row;
    HYPRE_Int        end_row;
    HYPRE_Int       *beg_rows;
    HYPRE_Int       *end_rows;
}
ParaSails;

ParaSails *ParaSailsCreate(MPI_Comm comm, HYPRE_Int beg_row, HYPRE_Int end_row, HYPRE_Int sym);
void ParaSailsDestroy(ParaSails *ps);
void ParaSailsSetupPattern(ParaSails *ps, Matrix *A,
  HYPRE_Real thresh, HYPRE_Int num_levels);
void ParaSailsSetupPatternExt(ParaSails *ps, Matrix *A,
  HYPRE_Real thresh_global, HYPRE_Real thresh_local, HYPRE_Int num_levels);
HYPRE_Int ParaSailsSetupValues(ParaSails *ps, Matrix *A, HYPRE_Real filter);
void ParaSailsApply(ParaSails *ps, HYPRE_Real *u, HYPRE_Real *v);
void ParaSailsApplyTrans(ParaSails *ps, HYPRE_Real *u, HYPRE_Real *v);
HYPRE_Real ParaSailsStatsPattern(ParaSails *ps, Matrix *A);
void ParaSailsStatsValues(ParaSails *ps, Matrix *A);

#ifdef __cplusplus
}
#endif

#endif /* _PARASAILS_H */
