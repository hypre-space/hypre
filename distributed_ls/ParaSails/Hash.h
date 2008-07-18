/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/




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
    int  size;  /* size of hash table */
    int  num;   /* number of entries in hash table */
    int *keys;  /* list of keys, used for rehashing */
    int *table; /* the hash table storing the keys */
    int *data;  /* data associated with each entry in the table */
}
Hash;

Hash *HashCreate(int size);
void  HashDestroy(Hash *h);
int   HashLookup(Hash *h, int key);
void  HashInsert(Hash *h, int key, int data);
void  HashRehash(Hash *old, Hash *);
void  HashReset(Hash *h);
void  HashPrint(Hash *h);

#endif /* _HASH_H */
