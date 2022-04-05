/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * PrunedRows.h header file.
 *
 *****************************************************************************/

#include <stdio.h>
#include "Mem.h"
#include "DiagScale.h"

#ifndef _PRUNEDROWS_H
#define _PRUNEDROWS_H

typedef struct
{
    Mem      *mem;   /* storage for arrays, indices, and values */
    HYPRE_Int      size;

    HYPRE_Int     *len;
    HYPRE_Int    **ind;
}
PrunedRows;

PrunedRows *PrunedRowsCreate(Matrix *mat, HYPRE_Int size, DiagScale *diag_scale,
  HYPRE_Real thresh);
void PrunedRowsDestroy(PrunedRows *p);
HYPRE_Int *PrunedRowsAlloc(PrunedRows *p, HYPRE_Int len);
void PrunedRowsPut(PrunedRows *p, HYPRE_Int index, HYPRE_Int len, HYPRE_Int *ind);
void PrunedRowsGet(PrunedRows *p, HYPRE_Int index, HYPRE_Int *lenp, HYPRE_Int **indp);

#endif /* _PRUNEDROWS_H */
