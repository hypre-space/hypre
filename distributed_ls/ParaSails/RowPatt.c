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
#include "mpi.h"
#include "Hash.h"
#include "RowPatt.h"

/*--------------------------------------------------------------------------
 * ROWPATT_EXIT - Print message, flush all output streams, return -1 to
 * operating system, and exit to operating system.  Used internally only.
 *--------------------------------------------------------------------------*/

#define ROWPATT_EXIT \
{  printf("Exiting...\n"); \
   fflush(NULL); \
   MPI_Abort(MPI_COMM_WORLD, -1); \
}

/*--------------------------------------------------------------------------
 * RowPattCreate - Return (a pointer to) a pattern of a row with a maximum
 * of "maxlen" nonzeros.  "maxlen" should actually be about double the 
 * maximum expected, since it is the size of a hash table.
 *--------------------------------------------------------------------------*/

RowPatt *RowPattCreate(int maxlen)
{
    RowPatt *p = (RowPatt *) malloc(sizeof(RowPatt));

    p->maxlen   = maxlen;
    p->len      = 0;
    p->prev_len = 0;
    p->ind      = (int *) malloc((maxlen) * sizeof(int));
    p->back     = (int *) malloc((maxlen) * sizeof(int));
    p->hash     = HashCreate(maxlen);

    return p;
}

/*--------------------------------------------------------------------------
 * RowPattDestroy - Destroy a row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattDestroy(RowPatt *p)
{
    free(p->ind);
    free(p->back);
    HashDestroy(p->hash);
    free(p);
}

/*--------------------------------------------------------------------------
 * RowPattReset - Empty the pattern of row pattern object "p".
 *--------------------------------------------------------------------------*/

void RowPattReset(RowPatt *p)
{
    HashReset(p->hash, p->len, p->back);
    p->len      = 0;
    p->prev_len = 0;
}

/*--------------------------------------------------------------------------
 * RowPattMerge - Merge the "len" nonzeros in array "ind" with pattern "p".
 *--------------------------------------------------------------------------*/

void RowPattMerge(RowPatt *p, int len, int *ind)
{
    int i, index, inserted;

    for (i=0; i<len; i++)
    {
        index = HashInsert(p->hash, ind[i], &inserted);

        if (inserted)
        {
	    p->back[p->len] = index;
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

void RowPattMergeExt(RowPatt *p, int len, int *ind, int beg, int end)
{
    int i, index, inserted;

    for (i=0; i<len; i++)
    {
        if (ind[i] < beg || ind[i] > end)
	{
            index = HashInsert(p->hash, ind[i], &inserted);

            if (inserted)
            {
	        p->back[p->len] = index;
                p->ind[p->len] = ind[i];
                p->len++;
            }
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


