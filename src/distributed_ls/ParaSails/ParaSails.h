/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




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

typedef struct
{
    HYPRE_Int        symmetric;
    double     thresh;
    HYPRE_Int        num_levels;
    double     filter;
    double     loadbal_beta;

    double     cost;          /* cost for this processor */
    double     setup_pattern_time;
    double     setup_values_time;

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
  double thresh, HYPRE_Int num_levels);
void ParaSailsSetupPatternExt(ParaSails *ps, Matrix *A, 
  double thresh_global, double thresh_local, HYPRE_Int num_levels);
HYPRE_Int ParaSailsSetupValues(ParaSails *ps, Matrix *A, double filter);
void ParaSailsApply(ParaSails *ps, double *u, double *v);
void ParaSailsApplyTrans(ParaSails *ps, double *u, double *v);
double ParaSailsStatsPattern(ParaSails *ps, Matrix *A);
void ParaSailsStatsValues(ParaSails *ps, Matrix *A);

#endif /* _PARASAILS_H */
