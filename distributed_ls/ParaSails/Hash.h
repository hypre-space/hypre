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
 * Hash.h header file.
 *
 *****************************************************************************/

#include <stdio.h>

#ifndef _HASH_H
#define _HASH_H

/* keys cannot be equal to HASH_EMPTY */
#define HASH_EMPTY    -1
#define HASH_NOTFOUND -1

typedef struct
{
    int  size;
    int *keys;
}
Hash;

Hash *HashCreate(int size);
void  HashDestroy(Hash *h);
int   HashInsert(Hash *h, int key, int *inserted);
int   HashLookup(Hash *h, int key);
int   HashLookup2(Hash *h, int key, int *nhops);
void  HashReset(Hash *h, int len, int *ind);

#endif /* _HASH_H */
