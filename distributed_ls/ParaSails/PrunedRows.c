/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * PrunedRows - Collection of pruned rows that are cached on the local 
 * processor.  Direct access to these rows is available, and is accomplished 
 * through a hash table.
 *
 * The local pruned rows are stored in the first num_local locations.
 * num_local is added to the hash table locations to get the storage locations
 * of the external pruned rows.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Mem.h"
#include "Hash.h"
#include "Matrix.h"
#include "DiagScale.h"
#include "PrunedRows.h"

/*--------------------------------------------------------------------------
 * PrunedRowsCreate - Return (a pointer to) a pruned rows object.
 *
 * mat        - matrix used to construct the local pruned rows (input)
 * size       - size of the hash table used for storing pointers to the 
 *              external pruned rows; should be about double the number 
 *              of external pruned rows expected; the local pruned rows
 *              do not use the hash table (input)
 * diag_scale - diagonal scale object used to scale the thresholding (input)
 * thresh     - threshold for pruning the matrix (input)
 *--------------------------------------------------------------------------*/

PrunedRows *PrunedRowsCreate(Matrix *mat, int size, DiagScale *diag_scale, 
  double thresh)
{
    int row, len, *ind, count, j, loc, *data;
    double *val, temp;

    PrunedRows *p = (PrunedRows *) malloc(sizeof(PrunedRows));

    p->mat  = mat;
    p->hash = HashCreate(size);
    p->mem  = MemCreate();

    p->num_local = mat->end_row - mat->beg_row + 1;

    p->len = (int *)  MemAlloc(p->mem, (size + p->num_local)*sizeof(int));
    p->ind = (int **) MemAlloc(p->mem, (size + p->num_local)*sizeof(int *));

    /* Prune and store the rows on the local processor */

    for (row=mat->beg_row; row<=mat->end_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

        count = 0;
        for (j=0; j<len; j++)
        {
            temp = DiagScaleGet(diag_scale, mat, row);
            if (temp*ABS(val[j])*DiagScaleGet(diag_scale, mat, ind[j]) 
              >= thresh || ind[j] == row)
                count++;
        }

        loc = row - mat->beg_row;
        p->ind[loc] = (int *) MemAlloc(p->mem, count*sizeof(int));
        p->len[loc] = count;

        data = p->ind[loc];
        for (j=0; j<len; j++)
        {
            temp = DiagScaleGet(diag_scale, mat, row);
            if (temp*ABS(val[j])*DiagScaleGet(diag_scale, mat, ind[j]) 
              >= thresh || ind[j] == row)
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
    HashDestroy(p->hash);
    MemDestroy(p->mem);
    free(p);
}

/*--------------------------------------------------------------------------
 * PrunedRowsAllocInd - Return space allocated for "len" indices in the
 * pruned rows object "p".  The indices may span several rows.
 *--------------------------------------------------------------------------*/

int *PrunedRowsAlloc(PrunedRows *p, int len)
{
    return (int *) MemAlloc(p->mem, len*sizeof(int));
}

/*--------------------------------------------------------------------------
 * PrunedRowsPut - Given a pruned row (len, ind), store it as row "index" in
 * the pruned rows object "p".  Only nonlocal pruned rows should be put using
 * this interface; the local pruned rows are put using the create function.
 *--------------------------------------------------------------------------*/

void PrunedRowsPut(PrunedRows *p, int index, int len, int *ind)
{
    int loc, inserted;

    if (p->mat->beg_row <= index && index <= p->mat->end_row)
    {
        printf("PrunedRowsPut: index %d is a local row.\n", index);
        PARASAILS_EXIT;
    }

    loc = HashInsert(p->hash, index, &inserted);
    assert(inserted);

    loc += p->num_local;
    p->len[loc] = len;
    p->ind[loc] = ind;
}

/*--------------------------------------------------------------------------
 * PrunedRowsGet - Return the row with index "index" through the pointers 
 * "lenp" and "indp" in the pruned rows object "p".
 *--------------------------------------------------------------------------*/

void PrunedRowsGet(PrunedRows *p, int index, int *lenp, int **indp)
{
    int loc;

    if (p->mat->beg_row <= index && index <= p->mat->end_row)
    {
        loc = index - p->mat->beg_row;
        *lenp = p->len[loc];
        *indp = p->ind[loc];
    }
    else
    {
        loc = HashLookup(p->hash, index);
        if (loc == HASH_NOTFOUND)
        {
            printf("PrunedRowsGet: index %d not found in hash table.\n", index);
            PARASAILS_EXIT;
        }

	loc += p->num_local;
        *lenp = p->len[loc];
        *indp = p->ind[loc];
    }
}

