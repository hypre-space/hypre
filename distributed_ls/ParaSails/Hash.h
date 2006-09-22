/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
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
