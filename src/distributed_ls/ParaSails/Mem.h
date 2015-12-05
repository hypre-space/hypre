/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/




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
    int   num_blocks;
    int   bytes_left;

    long  total_bytes;
    long  bytes_alloc;
    int   num_over;

    char *avail;
    char *blocks[MEM_MAXBLOCKS];
}
Mem;

Mem  *MemCreate();
void  MemDestroy(Mem *m);
char *MemAlloc(Mem *m, int size);
void  MemStat(Mem *m, FILE *stream, char *msg);

#endif /* _MEM_H */
