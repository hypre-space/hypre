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
 * Numbering - 
 *
 *****************************************************************************/

#include <stdlib.h>
#include <memory.h>
#include <assert.h>
#include "Common.h"
#include "Numbering.h"
#include "Matrix.h"

/* design options:
- should local indices be converted through the arrays as well?

global_to_local - no, since it would require a hash lookup anyway
local_to_global - yes -- would allow fast conversion using an array only

- global to local is simplified when we know that it is a local node.
   the index is simply: local=global-beg_row

- conversions from a to b, but can be done in place, i.e., a can be b

- how are the tables initialized?  [interface question]

* How is the numbering object created?  a maxtrix will originally have
* global indices.  Can the matrix be converted to local indices row by row?
* if local node, okay, if external node, then add to hash table if necessary.
* So, the matrix has a convert function, which must be passed a numbering
* object which is created by the user application.
*   matrix stores its own num_ext?  or, does matvec setup immediately
*    so it is not needed.
*/

/*--------------------------------------------------------------------------
 * NumberingCreate - Return (a pointer to) a numbering object.
 * size = maximum number of indices (size of local_to_global array)
 * size is also the maximum number of external indices (size will actually
 * be much larger than the number of external indices, but its hash table
 * needs to be this large.
 *
 * local indices start at 0
 * 0 .. num_loc-1, num_loc .. num_ind-1
 *
 *--------------------------------------------------------------------------*/

Numbering *NumberingCreate(int size, int beg_row, int end_row)
{
    int i;
    Numbering *numb = (Numbering *) malloc(sizeof(Numbering));

    numb->size    = size;
    numb->beg_row = beg_row;
    numb->end_row = end_row;
    numb->num_loc = end_row - beg_row + 1;
    numb->num_ind = end_row - beg_row + 1;
    numb->num_ext = -1; /* not set yet */

    numb->local_to_global = (int *) malloc(size * sizeof(int));
    numb->global_to_local = (int *) malloc(size * sizeof(int));
    numb->hash            = HashCreate(size);

    /* Set up the local part of local_to_global */
    for (i=0; i<numb->num_loc; i++)
        numb->local_to_global[i] = beg_row + i;

    return numb;
}

/*--------------------------------------------------------------------------
 * NumberingDestroy - Destroy a numbering object.
 *--------------------------------------------------------------------------*/

void NumberingDestroy(Numbering *numb)
{
    free(numb->local_to_global);
    free(numb->global_to_local);
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
 *--------------------------------------------------------------------------*/

void NumberingGlobalToLocal(Numbering *numb, int len, int *global, int *local)
{
    int i, index, inserted;

    for (i=0; i<len; i++)
    {
        if (global[i] < numb->beg_row || global[i] > numb->end_row)
        {
            index = HashInsert(numb->hash, global[i], &inserted);

	    if (inserted)
	    {
		/* stop if no more space */
                assert(numb->num_ind < numb->size); 

                numb->global_to_local[index] = numb->num_ind;
                numb->local_to_global[numb->num_ind] = global[i];
                numb->num_ind++;
	    }

            local[i] = numb->global_to_local[index];
        }
        else
        {
            local[i] = global[i] - numb->beg_row;
        }
    }
}
