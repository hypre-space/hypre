/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




#ifndef HASH_D_DH
#define HASH_D_DH

/* todo: rehash should be implemented in Hash_dhInsert();
         as of now, an error is set if the table overflows.
*/

#include "euclid_common.h"

/* This should be done with templates, if this were in C++;
   for now, a record contains every type of entry we might
   need; this is a waste of memory, when one is only intersted
   in hashing <key, HYPRE_Int> pairs!
*/
typedef struct _hash_node {
  HYPRE_Int     iData;      /* integer */
  double  fData;      /* float */
  HYPRE_Int     *iDataPtr;  /* pointer to integer */
  HYPRE_Int     *iDataPtr2; /* pointer to integer */
  double  *fDataPtr;  /* pointer to float */
} HashData;


typedef struct _hash_node_private HashRecord;

/* data structure for the hash table; do not directly access */
struct _hash_dh {
  HYPRE_Int         size;   /* total slots in table */
  HYPRE_Int         count;  /* number of insertions in table */
  HYPRE_Int         curMark;
  HashRecord  *data;
};


extern void Hash_dhCreate(Hash_dh *h, HYPRE_Int size);
extern void Hash_dhDestroy(Hash_dh h);
extern void Hash_dhInsert(Hash_dh h, HYPRE_Int key, HashData *data);
extern HashData *Hash_dhLookup(Hash_dh h, HYPRE_Int key);
         /* returns NULL if record isn't found. */

extern void Hash_dhReset(Hash_dh h);
extern void Hash_dhPrint(Hash_dh h, FILE *fp);


#define HASH_1(k,size,idxOut)    \
         {  *idxOut = k % size;  }

#define HASH_2(k,size,idxOut)      \
          {  \
            HYPRE_Int r = k % (size-13); \
            r = (r % 2) ? r : r+1; \
            *idxOut = r;           \
          }

#endif
