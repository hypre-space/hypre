/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * in the table (also known as a closed table).  Conflicts are resolved with
 *
 * We allow rehashing the data into a larger or smaller table, and thus
 * allow a data item (an integer, but a pointer would be more general)
 * to be stored with each key in the table.  (If we only return the
 * storage location of the key in the table (the implied index), then
 * rehashing would change the implied indices.)
 *
 * The modulus function is used as the hash function.
 * The keys must not equal HASH_EMPTY, which is -1.
 * The integer data associated with a key must not equal HASH_NOTFOUND,
 * which is -1.
 *
 *****************************************************************************/

#include <stdlib.h>
#include "Common.h"
#include "Hash.h"

/*--------------------------------------------------------------------------
 * HashCreate - Return (a pointer to) a hash table of size "size".
 * "size" should be prime, if possible.
 *--------------------------------------------------------------------------*/

Hash *HashCreate(HYPRE_Int size)
{
    HYPRE_Int i, *p;

    Hash *h = hypre_TAlloc(Hash, 1, HYPRE_MEMORY_HOST);

    h->size  = size;
    h->num   = 0;
    h->keys  = hypre_TAlloc(HYPRE_Int, size , HYPRE_MEMORY_HOST);
    h->table = hypre_TAlloc(HYPRE_Int, size , HYPRE_MEMORY_HOST);
    h->data  = hypre_TAlloc(HYPRE_Int, size , HYPRE_MEMORY_HOST);

    /* Initialize the table to empty */
    p = h->table;
    for (i=0; i<size; i++)
        *p++ = HASH_EMPTY;

    return h;
}

/*--------------------------------------------------------------------------
 * HashDestroy - Destroy a hash table object "h".
 *--------------------------------------------------------------------------*/

void HashDestroy(Hash *h)
{
    hypre_TFree(h->keys,HYPRE_MEMORY_HOST);
    hypre_TFree(h->table,HYPRE_MEMORY_HOST);
    hypre_TFree(h->data,HYPRE_MEMORY_HOST);
    hypre_TFree(h,HYPRE_MEMORY_HOST);
}

/*--------------------------------------------------------------------------
 * HashLookup - Look up the "key" in hash table "h" and return the data
 * associated with the key, or return HASH_NOTFOUND.
 *--------------------------------------------------------------------------*/

HYPRE_Int HashLookup(Hash *h, HYPRE_Int key)
{
    HYPRE_Int loc;

    /* loc = key % h->size; */
    HYPRE_Real keyd = key * 0.6180339887;
    loc = (HYPRE_Int) (h->size * (keyd - (HYPRE_Int) keyd));

    while (h->table[loc] != key)
    {
        if (h->table[loc] == HASH_EMPTY)
            return HASH_NOTFOUND;

        loc = (loc + 1) % h->size;
    }

    return h->data[loc];
}

/*--------------------------------------------------------------------------
 * HashInsert - Insert "key" with data "data" into hash table "h".
 * If the key is already in the hash table, the data item is replaced.
 *--------------------------------------------------------------------------*/

void HashInsert(Hash *h, HYPRE_Int key, HYPRE_Int data)
{
    HYPRE_Int loc;

    /* loc = key % h->size; */
    HYPRE_Real keyd = (HYPRE_Real) key * 0.6180339887;
    loc = (HYPRE_Int) ((HYPRE_Real) h->size * (keyd - (HYPRE_Int) keyd));

    while (h->table[loc] != key)
    {
        if (h->table[loc] == HASH_EMPTY)
        {
            hypre_assert(h->num < h->size);

	    h->keys[h->num++] = key;
            h->table[loc] = key;
            break;
        }

        loc = (loc + 1) % h->size;
    }

    h->data[loc] = data;
}

/*--------------------------------------------------------------------------
 * HashRehash - Given two hash tables, put the entries in one table into
 * the other.
 *--------------------------------------------------------------------------*/

void HashRehash(Hash *oldHash, Hash *newHash)
{
    HYPRE_Int i, data;

    for (i=0; i<oldHash->num; i++)
    {
	data = HashLookup(oldHash, oldHash->keys[i]);
	HashInsert(newHash, oldHash->keys[i], data);
    }
}

/*--------------------------------------------------------------------------
 * HashReset - Reset the hash table to all empty.
 *--------------------------------------------------------------------------*/

void HashReset(Hash *h)
{
    HYPRE_Int i, *p;

    h->num = 0;
    p = h->table;
    for (i=0; i<h->size; i++)
	*p++ = HASH_EMPTY;
}

/*--------------------------------------------------------------------------
 * HashPrint - Print hash table to stdout.
 *--------------------------------------------------------------------------*/

void HashPrint(Hash *h)
{
    HYPRE_Int i, j, *p;
    HYPRE_Int lines = h->size/38;

    hypre_printf("Hash size: %d\n", h->size);

    p = h->table;
    for (i=0; i<lines; i++)
    {
	for (j=0; j<38; j++)
	    hypre_printf("%d ", ((*p++ == HASH_EMPTY) ? 0 : 1));
	    /*hypre_printf("%d ", *p++);*/
	hypre_printf("\n");
    }
}

