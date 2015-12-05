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
 * RowPatt - Pattern of a row, and functions to manipulate the pattern of
 * a row, particularly merging a pattern with a set of nonzero indices.
 *
 * Implementation and Notes: a full-length array is used to mark nonzeros
 * in the pattern.  Indices must not equal -1, which is the "empty" marker
 * used in the full length array.  It is expected that RowPatt will only be 
 * presented with local indices, otherwise the full length array may be very 
 * large.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "RowPatt.h"

/*--------------------------------------------------------------------------
 * resize - local function for automatically increasing the size of RowPatt
 *--------------------------------------------------------------------------*/

static void resize(RowPatt *p, HYPRE_Int newlen)
{
    HYPRE_Int oldlen, i;

#ifdef PARASAILS_DEBUG
    hypre_printf("RowPatt resize %d\n", newlen);
#endif

    oldlen = p->maxlen;
    p->maxlen = newlen;

    p->ind  = (HYPRE_Int *) realloc(p->ind,  p->maxlen * sizeof(HYPRE_Int));
    p->mark = (HYPRE_Int *) realloc(p->mark, p->maxlen * sizeof(HYPRE_Int));

    /* initialize the new portion of the mark array */
    for (i=oldlen; i<p->maxlen; i++)
	p->mark[i] = -1;
}

/*--------------------------------------------------------------------------
 * RowPattCreate - Return (a pointer to) a pattern of a row with a maximum
 * of "maxlen" nonzeros.
 *--------------------------------------------------------------------------*/

RowPatt *RowPattCreate(HYPRE_Int maxlen)
{
    HYPRE_Int i;
    RowPatt *p = (RowPatt *) malloc(sizeof(RowPatt));

    p->maxlen   = maxlen;
    p->len      = 0;
    p->prev_len = 0;
    p->ind      = (HYPRE_Int *) malloc(maxlen * sizeof(HYPRE_Int));
    p->mark     = (HYPRE_Int *) malloc(maxlen * sizeof(HYPRE_Int));
    p->buffer   = NULL;
    p->buflen   = 0;

    for (i=0; i<maxlen; i++)
        p->mark[i] = -1;

    return p;
}

/*--------------------------------------------------------------------------
 * RowPattDestroy - Destroy a row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattDestroy(RowPatt *p)
{
    free(p->ind);
    free(p->mark);
    free(p->buffer);
    free(p);
}

/*--------------------------------------------------------------------------
 * RowPattReset - Empty the pattern of row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattReset(RowPatt *p)
{
    HYPRE_Int i;

    for (i=0; i<p->len; i++)
        p->mark[p->ind[i]] = -1;

    p->len      = 0;
    p->prev_len = 0;
}

/*--------------------------------------------------------------------------
 * RowPattMerge - Merge the "len" nonzeros in array "ind" with pattern "p".
 *--------------------------------------------------------------------------*/

void RowPattMerge(RowPatt *p, HYPRE_Int len, HYPRE_Int *ind)
{
    HYPRE_Int i;

    for (i=0; i<len; i++)
    {
	if (ind[i] >= p->maxlen)
	    resize(p, ind[i]*2);

	if (p->mark[ind[i]] == -1)
	{
	    assert(p->len < p->maxlen);

	    p->mark[ind[i]] = p->len;
            p->ind[p->len] = ind[i];
            p->len++;
	}
    }
}

/*--------------------------------------------------------------------------
 * RowPattMergeExt - Merge the external nonzeros in the array "ind" of 
 * length "len" with the pattern "p".  The external indices are those
 * that are less than "beg" or greater than "end".
 *--------------------------------------------------------------------------*/

void RowPattMergeExt(RowPatt *p, HYPRE_Int len, HYPRE_Int *ind, HYPRE_Int num_loc)
{
    HYPRE_Int i;

    for (i=0; i<len; i++)
    {
        if (ind[i] < num_loc)
	    continue;

	if (ind[i] >= p->maxlen)
	    resize(p, ind[i]*2);

	if (p->mark[ind[i]] == -1)
	{
	    assert(p->len < p->maxlen);

	    p->mark[ind[i]] = p->len;
            p->ind[p->len] = ind[i];
            p->len++;
	}
    }
}

/*--------------------------------------------------------------------------
 * RowPattGet - Return the pattern of "p".  The length and pointer to the
 * pattern indices are returned through the parameters "lenp" and "indp".
 * A copy of the indices is returned; this copy is destroyed on the next
 * call to RowPattGet or RowPattPrevLevel.
 *--------------------------------------------------------------------------*/

void RowPattGet(RowPatt *p, HYPRE_Int *lenp, HYPRE_Int **indp)
{
    HYPRE_Int len;

    len = p->len;

    if (len > p->buflen)
    {
	free(p->buffer);
	p->buflen = len + 100;
	p->buffer = (HYPRE_Int *) malloc(p->buflen * sizeof(HYPRE_Int));
    }

    memcpy(p->buffer, p->ind, len*sizeof(HYPRE_Int));

    *lenp = len;
    *indp = p->buffer;
}

/*--------------------------------------------------------------------------
 * RowPattPrevLevel - Return the new indices added to the pattern of "p"
 * since the last call to RowPattPrevLevel (or all the indices if never
 * called).  The length and pointer to the pattern indices are returned 
 * through the parameters "lenp" and "indp".
 * A copy of the indices is returned; this copy is destroyed on the next
 * call to RowPattGet or RowPattPrevLevel.
 *--------------------------------------------------------------------------*/

void RowPattPrevLevel(RowPatt *p, HYPRE_Int *lenp, HYPRE_Int **indp)
{
    HYPRE_Int len;

    len = p->len - p->prev_len;

    if (len > p->buflen)
    {
	free(p->buffer);
	p->buflen = len + 100;
	p->buffer = (HYPRE_Int *) malloc(p->buflen * sizeof(HYPRE_Int));
    }

    memcpy(p->buffer, &p->ind[p->prev_len], len*sizeof(HYPRE_Int));

    *lenp = len;
    *indp = p->buffer;

    p->prev_len = p->len;
}
