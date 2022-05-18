/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Hash.h header file.
 *
 *****************************************************************************/

#include <stdio.h>

#ifndef _HASH_H
#define _HASH_H

#define HASH_EMPTY    -1 /* keys cannot equal HASH_EMPTY */
#define HASH_NOTFOUND -1 /* data items cannot equal HASH_NOTFOUND */

typedef struct
{
    HYPRE_Int  size;  /* size of hash table */
    HYPRE_Int  num;   /* number of entries in hash table */
    HYPRE_Int *keys;  /* list of keys, used for rehashing */
    HYPRE_Int *table; /* the hash table storing the keys */
    HYPRE_Int *data;  /* data associated with each entry in the table */
}
Hash;

Hash *HashCreate(HYPRE_Int size);
void  HashDestroy(Hash *h);
HYPRE_Int   HashLookup(Hash *h, HYPRE_Int key);
void  HashInsert(Hash *h, HYPRE_Int key, HYPRE_Int data);
void  HashRehash(Hash *old, Hash *);
void  HashReset(Hash *h);
void  HashPrint(Hash *h);

#endif /* _HASH_H */
