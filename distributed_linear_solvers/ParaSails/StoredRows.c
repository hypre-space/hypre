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
 * StoredRows - Collection of rows that are cached on the local processor.
 * Direct access to these rows is available, and is accomplished through
 * a hash table.
 *
 * The local rows are not actually stored in the data structure, but the
 * StoredRowsGet function will retrieve it directly from the matrix.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Mem.h"
#include "Hash.h"
#include "Matrix.h"
#include "StoredRows.h"

/*--------------------------------------------------------------------------
 * StoredRowsCreate - Return (a pointer to) a stored rows object.
 *
 * mat        - matrix used for returning local rows (input)
 * size       - size of the hash table used for storing pointers to the
 *              external rows to be stored; should be about double the 
 *              expected number of rows to be stored; StoredRows returns
 *              local rows directly from the local part of the matrix,
 *              so the number of these rows does not need to be included
 *              in the size of the hash table (input)
 *--------------------------------------------------------------------------*/

StoredRows *StoredRowsCreate(Matrix *mat, int size)
{
    StoredRows *p = (StoredRows *) malloc(sizeof(StoredRows));

    p->mat  = mat;
    p->hash = HashCreate(size);
    p->mem  = MemCreate();

    p->len = (int *)     MemAlloc(p->mem, size * sizeof(int));
    p->ind = (int **)    MemAlloc(p->mem, size * sizeof(int *));
    p->val = (double **) MemAlloc(p->mem, size * sizeof(double *));

    p->count = 0;

    return p;
}

/*--------------------------------------------------------------------------
 * StoredRowsDestroy - Destroy a stored rows object "p".
 *--------------------------------------------------------------------------*/

void StoredRowsDestroy(StoredRows *p)
{
    HashDestroy(p->hash);
    MemDestroy(p->mem);
    free(p);
}

/*--------------------------------------------------------------------------
 * StoredRowsAllocInd - Return space allocated for "len" indices in the
 * stored rows object "p".  The indices may span several rows.
 *--------------------------------------------------------------------------*/

int *StoredRowsAllocInd(StoredRows *p, int len)
{
    return (int *) MemAlloc(p->mem, len*sizeof(int));
}

/*--------------------------------------------------------------------------
 * StoredRowsAllocVal - Return space allocated for "len" values in the
 * stored rows object "p".  The values may span several rows.
 *--------------------------------------------------------------------------*/

double *StoredRowsAllocVal(StoredRows *p, int len)
{
    return (double *) MemAlloc(p->mem, len*sizeof(double));
}

/*--------------------------------------------------------------------------
 * StoredRowsPut - Given a row (len, ind, val), store it as row "index" in
 * the stored rows object "p".  Only nonlocal stored rows should be put using
 * this interface; the local stored rows are put using the create function.
 *--------------------------------------------------------------------------*/

void StoredRowsPut(StoredRows *p, int index, int len, int *ind, double *val)
{
    int loc, inserted;

    if (p->mat->beg_row <= index && index <= p->mat->end_row)
    {
        printf("StoredRowsPut: index %d is a local row.\n", index);
        PARASAILS_EXIT;
    }

    loc = HashInsert(p->hash, index, &inserted);
    assert(inserted);

    p->len[loc] = len;
    p->ind[loc] = ind;
    p->val[loc] = val;

    p->count++;
}

/*--------------------------------------------------------------------------
 * StoredRowsGet - Return the row with index "index" through the pointers 
 * "lenp", "indp" and "valp" in the stored rows object "p".
 *--------------------------------------------------------------------------*/

void StoredRowsGet(StoredRows *p, int index, int *lenp, int **indp, 
  double **valp)
{
    if (p->mat->beg_row <= index && index <= p->mat->end_row)
    {
        MatrixGetRow(p->mat, index, lenp, indp, valp);
    }
    else
    {
        int loc = HashLookup(p->hash, index);

        if (loc == HASH_NOTFOUND)
        {
            printf("StoredRowsGet: index %d not found in hash table.\n", index);
            PARASAILS_EXIT;
        }

        *lenp = p->len[loc];
        *indp = p->ind[loc];
        *valp = p->val[loc];
    }
}

