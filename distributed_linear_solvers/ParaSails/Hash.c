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
 * Hash - Hash table.  This is an open addressing hash table with linear
 * probing.  The keys are nonnegative integers.  The lookup function 
 * returns the index in the hash table of the key, or HASH_NOTFOUND.  
 * The user may want to map a returned index to a pointer that contains 
 * the user's data associated with the key.  The modulus function is 
 * used as the hash function.
 *
 *****************************************************************************/

#include <stdlib.h>
#include <assert.h>
#include "Common.h"
#include "Hash.h"

/*--------------------------------------------------------------------------
 * HashCreate - Return (a pointer to) a hash table of size "size".
 * "size" should be prime, if possible.
 *--------------------------------------------------------------------------*/

Hash *HashCreate(int size)
{
    int i, *p;

    Hash *h = (Hash *) malloc(sizeof(Hash));

    h->size = size;
    h->keys = (int *) malloc(size * sizeof(int));

    /* Initialize the table to empty */
    p = h->keys;
    for (i=0; i<size; i++)
        *p++ = HASH_EMPTY;

    return h;
}

/*--------------------------------------------------------------------------
 * HashDestroy - Destroy a hash table object "h".
 *--------------------------------------------------------------------------*/

void HashDestroy(Hash *h)
{
    free(h->keys);
    free(h);
}

/*--------------------------------------------------------------------------
 * HashInsert - Insert "key" into hash table "h" and return the index
 * of the location where it was inserted.  Keys are nonnegative integers,
 * as are returned indices.  If the key is already in the hash table,
 * the key is not inserted but the index is still returned.  The returned
 * parameter "inserted" indicates whether or not the key was inserted.
 *--------------------------------------------------------------------------*/

int HashInsert(Hash *h, int key, int *inserted)
{
    int loc, initloc;

    initloc = key % h->size;
    loc = initloc;

    *inserted = 0;

    while (h->keys[loc] != key)
    {
        if (h->keys[loc] == HASH_EMPTY)
        {
            h->keys[loc] = key;  /* insert the key */
            *inserted = 1;
            break; /* break to return statement */
        }

        loc = (loc + 1) % h->size;

        if (loc == initloc)
        {
	    printf("HashInsert: hash table of size %d is full.\n", h->size);
	    PARASAILS_EXIT;
	}
    }

    return loc;
}

/*--------------------------------------------------------------------------
 * HashLookup - Look up the "key" in hash table "h" and return the index
 * of its location in the hash table, or return HASH_NOTFOUND.
 *--------------------------------------------------------------------------*/

int HashLookup(Hash *h, int key)
{
    int loc;

    loc = key % h->size;

    while (h->keys[loc] != key)
    {
        if (h->keys[loc] == HASH_EMPTY)
            return HASH_NOTFOUND;

        loc = (loc + 1) % h->size;
    }

    return loc;
}

/*--------------------------------------------------------------------------
 * HashReset - Empty the contents of the hash table "h" by reseting the 
 * "len" locations in the array "ind".  This is useful if the location
 * indices have been saved by the user and a hash table needs to be reused.
 *--------------------------------------------------------------------------*/

void HashReset(Hash *h, int len, int *ind)
{
    int i;

    for (i=0; i<len; i++)
        h->keys[ind[i]] = HASH_EMPTY;
}
