/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * PrunedRows - Collection of pruned rows that are cached on the local
 * processor.  Direct access to these rows is available, via the local
 * index number.
 *
 *****************************************************************************/

#include <stdlib.h>
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
  HYPRE_Real thresh)
{
    HYPRE_Int row, len, *ind, count, j, *data;
    HYPRE_Real *val, temp;

    PrunedRows *p = hypre_TAlloc(PrunedRows, 1, HYPRE_MEMORY_HOST);

    p->mem  = MemCreate();
    p->size = MAX(size, mat->end_row - mat->beg_row + 1);

    p->len = hypre_TAlloc(HYPRE_Int, p->size , HYPRE_MEMORY_HOST);
    p->ind = hypre_TAlloc(HYPRE_Int *, p->size , HYPRE_MEMORY_HOST);

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
    hypre_TFree(p->len,HYPRE_MEMORY_HOST);
    hypre_TFree(p->ind,HYPRE_MEMORY_HOST);
    hypre_TFree(p,HYPRE_MEMORY_HOST);
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
	p->len = hypre_TReAlloc(p->len,HYPRE_Int,  p->size , HYPRE_MEMORY_HOST);
	p->ind = hypre_TReAlloc(p->ind,HYPRE_Int *,  p->size , HYPRE_MEMORY_HOST);
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
