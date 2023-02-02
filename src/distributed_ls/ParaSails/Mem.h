/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Mem.h header file.
 *
 *****************************************************************************/

#include <stdio.h>

#ifndef _MEM_H
#define _MEM_H

#define MEM_BLOCKSIZE (2*1024*1024)
#define MEM_MAXBLOCKS 1024

typedef struct
{
    HYPRE_Int   num_blocks;
    HYPRE_Int   bytes_left;

    hypre_longint  total_bytes;
    hypre_longint  bytes_alloc;
    HYPRE_Int   num_over;

    char *avail;
    char *blocks[MEM_MAXBLOCKS];
}
Mem;

Mem  *MemCreate(void);
void  MemDestroy(Mem *m);
char *MemAlloc(Mem *m, HYPRE_Int size);
void  MemStat(Mem *m, FILE *stream, char *msg);

#endif /* _MEM_H */
