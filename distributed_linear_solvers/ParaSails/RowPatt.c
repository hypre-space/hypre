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
 * RowPatt - Pattern of a row, and functions to manipulate the pattern of
 * a row, particularly merging a pattern with a set of nonzero indices.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "RowPatt.h"

/*--------------------------------------------------------------------------
 * RowPattCreate - Return (a pointer to) a pattern of a row with a maximum
 * of "maxlen" nonzeros.  "maxlen" should actually be about double the 
 * maximum expected, since it is the size of a hash table.
 *
 * Merge uses local indices.
 *--------------------------------------------------------------------------*/

RowPatt *RowPattCreate(int maxlen)
{
    int i;
    RowPatt *p = (RowPatt *) malloc(sizeof(RowPatt));

    p->maxlen   = maxlen;
    p->len      = 0;
    p->prev_len = 0;
    p->ind      = (int *) malloc((maxlen) * sizeof(int));
    p->mark     = (int *) malloc((maxlen) * sizeof(int));

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
    free(p);
}

/*--------------------------------------------------------------------------
 * RowPattReset - Empty the pattern of row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattReset(RowPatt *p)
{
    int i;

    for (i=0; i<p->len; i++)
        p->mark[p->ind[i]] = -1;

    p->len      = 0;
    p->prev_len = 0;
}

/*--------------------------------------------------------------------------
 * RowPattMerge - Merge the "len" nonzeros in array "ind" with pattern "p".
 *--------------------------------------------------------------------------*/

void RowPattMerge(RowPatt *p, int len, int *ind)
{
    int i;

    for (i=0; i<len; i++)
    {
	assert(ind[i] < p->maxlen);

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

void RowPattMergeExt(RowPatt *p, int len, int *ind, int num_loc)
{
    int i, index, inserted;

    for (i=0; i<len; i++)
    {
        if (ind[i] < num_loc)
	    continue;

	assert(ind[i] < p->maxlen);

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
 *--------------------------------------------------------------------------*/

void RowPattGet(RowPatt *p, int *lenp, int **indp)
{
    *lenp = p->len;
    *indp = p->ind;
}

/*--------------------------------------------------------------------------
 * RowPattPrevLevel - Return the new indices added to the pattern of "p"
 * since the last call to RowPattPrevLevel (or all the indices if never
 * called).  The length and pointer to the pattern indices are returned 
 * through the parameters "lenp" and "indp".
 *--------------------------------------------------------------------------*/

void RowPattPrevLevel(RowPatt *p, int *lenp, int **indp)
{
    *lenp = p->len - p->prev_len;
    *indp = &p->ind[p->prev_len];

    p->prev_len = p->len;
}
