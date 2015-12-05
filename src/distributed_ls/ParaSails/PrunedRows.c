/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * PrunedRows - Collection of pruned rows that are cached on the local 
 * processor.  Direct access to these rows is available, via the local
 * index number.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Mem.h"
#include "Matrix.h"
#include "DiagScale.h"
#include "PrunedRows.h"

/*--------------------------------------------------------------------------
 * PrunedRowsCreate - Return (a pointer to) a pruned rows object.
 *
 * mat        - matrix used to construct the local pruned rows (input)
 *              assumes the matrix uses local indexing
 * size       - number of unique local indices on this processor;
 *              an array of this size will be allocated to access the
 *              pruned rows (input) - includes the number of local nodes
 * diag_scale - diagonal scale object used to scale the thresholding (input)
 * thresh     - threshold for pruning the matrix (input)
 *
 * The local pruned rows are stored in the first part of the len and ind 
 * arrays.
 *--------------------------------------------------------------------------*/

PrunedRows *PrunedRowsCreate(Matrix *mat, HYPRE_Int size, DiagScale *diag_scale, 
  double thresh)
{
    HYPRE_Int row, len, *ind, count, j, *data;
    double *val, temp;

    PrunedRows *p = (PrunedRows *) malloc(sizeof(PrunedRows));

    p->mem  = MemCreate();
    p->size = MAX(size, mat->end_row - mat->beg_row + 1);

    p->len = (HYPRE_Int *)  malloc(p->size * sizeof(HYPRE_Int));
    p->ind = (HYPRE_Int **) malloc(p->size * sizeof(HYPRE_Int *));

    /* Prune and store the rows on the local processor */

    for (row=0; row<=mat->end_row - mat->beg_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

        count = 1; /* automatically include the diagonal */
        for (j=0; j<len; j++)
        {
            temp = DiagScaleGet(diag_scale, row);
            if (temp*ABS(val[j])*DiagScaleGet(diag_scale, ind[j]) 
              >= thresh && ind[j] != row)
                count++;
        }

        p->ind[row] = (HYPRE_Int *) MemAlloc(p->mem, count*sizeof(HYPRE_Int));
        p->len[row] = count;

        data = p->ind[row];
        *data++ = row; /* the diagonal entry */
        for (j=0; j<len; j++)
        {
            temp = DiagScaleGet(diag_scale, row);
            if (temp*ABS(val[j])*DiagScaleGet(diag_scale, ind[j]) 
              >= thresh && ind[j] != row)
                *data++ = ind[j];
        }
    }

    return p;
}

/*--------------------------------------------------------------------------
 * PrunedRowsDestroy - Destroy a pruned rows object "p".
 *--------------------------------------------------------------------------*/

void PrunedRowsDestroy(PrunedRows *p)
{
    MemDestroy(p->mem);
    free(p->len);
    free(p->ind);
    free(p);
}

/*--------------------------------------------------------------------------
 * PrunedRowsAllocInd - Return space allocated for "len" indices in the
 * pruned rows object "p".  The indices may span several rows.
 *--------------------------------------------------------------------------*/

HYPRE_Int *PrunedRowsAlloc(PrunedRows *p, HYPRE_Int len)
{
    return (HYPRE_Int *) MemAlloc(p->mem, len*sizeof(HYPRE_Int));
}

/*--------------------------------------------------------------------------
 * PrunedRowsPut - Given a pruned row (len, ind), store it as row "index" in
 * the pruned rows object "p".  Only nonlocal pruned rows should be put using
 * this interface; the local pruned rows are put using the create function.
 *--------------------------------------------------------------------------*/

void PrunedRowsPut(PrunedRows *p, HYPRE_Int index, HYPRE_Int len, HYPRE_Int *ind)
{
    if (index >= p->size)
    {
	p->size = index*2;
#ifdef PARASAILS_DEBUG
	hypre_printf("StoredRows resize %d\n", p->size);
#endif
	p->len = (HYPRE_Int *)  realloc(p->len, p->size * sizeof(HYPRE_Int));
	p->ind = (HYPRE_Int **) realloc(p->ind, p->size * sizeof(HYPRE_Int *));
    }

    p->len[index] = len;
    p->ind[index] = ind;
}

/*--------------------------------------------------------------------------
 * PrunedRowsGet - Return the row with index "index" through the pointers 
 * "lenp" and "indp" in the pruned rows object "p".
 *--------------------------------------------------------------------------*/

void PrunedRowsGet(PrunedRows *p, HYPRE_Int index, HYPRE_Int *lenp, HYPRE_Int **indp)
{
    *lenp = p->len[index];
    *indp = p->ind[index];
}
