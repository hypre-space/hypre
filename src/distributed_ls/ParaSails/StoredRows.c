/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * StoredRows - Local storage of rows from other processors.  Although only
 * off-processor rows are stored, if an on-processor row is requested, it
 * is returned by referring to the local matrix.  Local indexing is used to
 * access the stored rows.
 *
 *****************************************************************************/

#include <stdlib.h>
#include "Common.h"
#include "Mem.h"
#include "Matrix.h"
#include "StoredRows.h"

/*--------------------------------------------------------------------------
 * StoredRowsCreate - Return (a pointer to) a stored rows object.
 *
 * mat  - matrix used for returning on-processor rows (input)
 * size - the maximum number of (off-processor) rows that can be stored 
 *        (input).  See below for more a precise description.
 *
 * A slot is available for "size" off-processor rows.  The slot for the
 * row with local index i is (i - num_loc).  Therefore, if max_i is the
 * largest local index expected, then size should be set to 
 * (max_i - num_loc + 1).  StoredRows will automatically increase its 
 * size if a row with a larger local index needs to be put in StoredRows.
 *--------------------------------------------------------------------------*/

StoredRows *StoredRowsCreate(Matrix *mat, HYPRE_Int size)
{
    StoredRows *p = hypre_TAlloc(StoredRows, 1, HYPRE_MEMORY_HOST);

    p->mat  = mat;
    p->mem  = MemCreate();

    p->size = size;
    p->num_loc = mat->end_row - mat->beg_row + 1;

    p->len = hypre_CTAlloc(HYPRE_Int, size, HYPRE_MEMORY_HOST);
    p->ind = hypre_TAlloc(HYPRE_Int *, size , HYPRE_MEMORY_HOST);
    p->val = hypre_TAlloc(HYPRE_Real *, size , HYPRE_MEMORY_HOST);

    p->count = 0;

    return p;
}

/*--------------------------------------------------------------------------
 * StoredRowsDestroy - Destroy a stored rows object "p".
 *--------------------------------------------------------------------------*/

void StoredRowsDestroy(StoredRows *p)
{
    MemDestroy(p->mem);
    hypre_TFree(p->len,HYPRE_MEMORY_HOST);
    hypre_TFree(p->ind,HYPRE_MEMORY_HOST);
    hypre_TFree(p->val,HYPRE_MEMORY_HOST);
    hypre_TFree(p,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * StoredRowsAllocInd - Return space allocated for "len" indices in the
 * stored rows object "p".  The indices may span several rows.
 *--------------------------------------------------------------------------*/

HYPRE_Int *StoredRowsAllocInd(StoredRows *p, HYPRE_Int len)
{
    return (HYPRE_Int *) MemAlloc(p->mem, len*sizeof(HYPRE_Int));
}

/*--------------------------------------------------------------------------
 * StoredRowsAllocVal - Return space allocated for "len" values in the
 * stored rows object "p".  The values may span several rows.
 *--------------------------------------------------------------------------*/

HYPRE_Real *StoredRowsAllocVal(StoredRows *p, HYPRE_Int len)
{
    return (HYPRE_Real *) MemAlloc(p->mem, len*sizeof(HYPRE_Real));
}

/*--------------------------------------------------------------------------
 * StoredRowsPut - Given a row (len, ind, val), store it as row "index" in
 * the stored rows object "p".  Only nonlocal stored rows should be put using
 * this interface; the local stored rows are put using the create function.
 *--------------------------------------------------------------------------*/

void StoredRowsPut(StoredRows *p, HYPRE_Int index, HYPRE_Int len, HYPRE_Int *ind, HYPRE_Real *val)
{
    HYPRE_Int i = index - p->num_loc;

    /* Reallocate if necessary */
    if (i >= p->size)
    {
        HYPRE_Int j;
        HYPRE_Int newsize;

	newsize = i*2;
#ifdef PARASAILS_DEBUG
		    hypre_printf("StoredRows resize %d\n", newsize);
#endif
        p->len = hypre_TReAlloc(p->len,HYPRE_Int,  newsize , HYPRE_MEMORY_HOST);
        p->ind = hypre_TReAlloc(p->ind,HYPRE_Int *,  newsize , HYPRE_MEMORY_HOST);
        p->val = hypre_TReAlloc(p->val,HYPRE_Real *,  newsize , HYPRE_MEMORY_HOST);

	/* set lengths to zero */
        for (j=p->size; j<newsize; j++)
	    p->len[j] = 0;

        p->size = newsize;
    }

    /* check that row has not been put already */
    hypre_assert(p->len[i] == 0);

    p->len[i] = len;
    p->ind[i] = ind;
    p->val[i] = val;

    p->count++;
}

/*--------------------------------------------------------------------------
 * StoredRowsGet - Return the row with index "index" through the pointers 
 * "lenp", "indp" and "valp" in the stored rows object "p".
 *--------------------------------------------------------------------------*/

void StoredRowsGet(StoredRows *p, HYPRE_Int index, HYPRE_Int *lenp, HYPRE_Int **indp, 
  HYPRE_Real **valp)
{
    if (index < p->num_loc)
    {
        MatrixGetRow(p->mat, index, lenp, indp, valp);
    }
    else
    {
	index = index - p->num_loc;

        *lenp = p->len[index];
        *indp = p->ind[index];
        *valp = p->val[index];
    }
}
