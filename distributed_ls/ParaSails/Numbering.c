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
 * Numbering - An object that maintains a mapping to and from global indices
 * and local indices.  The local indices are numbered:
 *
 * 0 .. num_loc - 1         <--- locally owned indices
 * num_loc .. num_ind - 1   <--- external indices
 *
 * Implementation:  Mapping from a local index to a global index is performed
 * through an array.  Mapping from a global index to a local index is more
 * difficult.  If the global index is determined to be owned by the local
 * processor, then a conversion is performed; else the local index is 
 * looked up in a hash table.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "Common.h"
#include "Numbering.h"

/*--------------------------------------------------------------------------
 * NumberingCreate - Return (a pointer to) a numbering object
 * for a given matrix.  The "size" parameter is the initial number of 
 * external indices that can be stored, and will grow if necessary.
 * (Implementation note: the hash table size is kept approximately twice
 * this number.)
 *
 * The numbering is created such that the local indices from a given processor
 * are contiguous.  This is required by the mat-vec routine.
 *--------------------------------------------------------------------------*/

Numbering *NumberingCreate(Matrix *mat, int size)
{
    Numbering *numb = (Numbering *) malloc(sizeof(Numbering));
    int row, i, len, *ind;
    double *val;
    int num_external = 0;
    int *local_to_global; /* temp pointer */

    numb->size    = size;
    numb->beg_row = mat->beg_row;
    numb->end_row = mat->end_row;
    numb->num_loc = mat->end_row - mat->beg_row + 1;
    numb->num_ind = mat->end_row - mat->beg_row + 1;

    numb->local_to_global = (int *) malloc((numb->num_loc+size) * sizeof(int));
    numb->hash            = HashCreate(2*size+1);

    /* Set up the local part of local_to_global */
    for (i=0; i<numb->num_loc; i++)
        numb->local_to_global[i] = mat->beg_row + i;

    /* Set up pointer to external part of local_to_global array */
    local_to_global = &numb->local_to_global[numb->num_loc];

    /* Fill local_to_global array */
    for (row=0; row<=mat->end_row - mat->beg_row; row++)
    {
        MatrixGetRow(mat, row, &len, &ind, &val);

        for (i=0; i<len; i++)
        {
            /* Only interested in external indices */
	    if (ind[i] < mat->beg_row || ind[i] > mat->end_row)
            {
		if (HashLookup(numb->hash, ind[i]) == HASH_NOTFOUND)
		{
                    HashInsert(numb->hash, ind[i], num_external);
                    local_to_global[num_external] = ind[i];
		    num_external++;
		}
            }
        }
    }

    /* Sort the indices */
    shell_sort(num_external, local_to_global);

    /* Redo the hash table for the sorted indices */
    HashReset(numb->hash);

    for (i=0; i<num_external; i++)
        HashInsert(numb->hash, local_to_global[i], i + numb->num_loc);

    numb->num_ind += num_external;

    return numb;
}

/*--------------------------------------------------------------------------
 * NumberingCreateCopy 
 *--------------------------------------------------------------------------*/

Numbering *NumberingCreateCopy(Numbering *orig)
{
    Numbering *numb = (Numbering *) malloc(sizeof(Numbering));

    numb->size    = orig->size;
    numb->beg_row = orig->beg_row;
    numb->end_row = orig->end_row;
    numb->num_loc = orig->num_loc;
    numb->num_ind = orig->num_ind;

    numb->local_to_global = 
        (int *) malloc((numb->num_loc+numb->size) * sizeof(int));
    memcpy(numb->local_to_global, orig->local_to_global, 
         numb->num_ind*sizeof(int));

    numb->hash = HashCreate(2*numb->size+1);
    HashRehash(orig->hash, numb->hash);

    return numb;
}

/*--------------------------------------------------------------------------
 * NumberingDestroy - Destroy a numbering object.
 *--------------------------------------------------------------------------*/

void NumberingDestroy(Numbering *numb)
{
    free(numb->local_to_global);
    HashDestroy(numb->hash);

    free(numb);
}

/*--------------------------------------------------------------------------
 * NumberingLocalToGlobal -
 *--------------------------------------------------------------------------*/

void NumberingLocalToGlobal(Numbering *numb, int len, int *local, int *global)
{
    int i;

    for (i=0; i<len; i++)
        global[i] = numb->local_to_global[local[i]];
}

/*--------------------------------------------------------------------------
 * NumberingGlobalToLocal -
 * global indices are added this way...
 *** new : no longer need a global to local, since this is done by hash tab **
 *** ALLOW in place conversion
 *--------------------------------------------------------------------------*/

void NumberingGlobalToLocal(Numbering *numb, int len, int *global, int *local)
{
    int i, l;

    for (i=0; i<len; i++)
    {
        if (global[i] < numb->beg_row || global[i] > numb->end_row)
        {
	    l = HashLookup(numb->hash, global[i]);

	    if (l == HASH_NOTFOUND)
	    {
                if (numb->num_ind >= numb->num_loc + numb->size)
		{
		    Hash *new;

		    /* allocate more space for numbering */
		    numb->size *= 2;
		    numb->local_to_global = (int *) 
			realloc(numb->local_to_global, numb->size*sizeof(int));
                    new = HashCreate(2*numb->size+1);
		    HashRehash(numb->hash, new);
		    HashDestroy(numb->hash);
		    numb->hash = new;
		}

		HashInsert(numb->hash, global[i], numb->num_ind);
		numb->local_to_global[numb->num_ind] = global[i];
		local[i] = numb->num_ind;
		numb->num_ind++;
	    }
	    else
	    {
	        local[i] = l;
	    }
        }
        else
        {
            local[i] = global[i] - numb->beg_row;
        }
    }
}
