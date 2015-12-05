/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * Mem - Memory pool for aggregate data with unknown total size at creation.
 * For example, a sparse matrix may be constructed one row at a time, which
 * do not need to be stored contiguously in memory.  MemAlloc may be called
 * for each row that needs to be stored, and space is allocated from the 
 * memory pool (individual requests are not made to the operating system).
 * Memory from the memory pool is freed entirely at once.
 *
 * Memory is requested from the operating system in blocks of 1 Mbyte
 * by default.  This default must be changed if requests of more than
 * 1 Mbyte will be made, or if large requests (e.g., 0.5 Mbytes) will
 * be made, in order to efficiently use the memory block.  Up to 1000 
 * blocks can be allocated, by default, giving a total of 1 Gbyte of 
 * memory.  Actual storage will be less, and this can be determined by
 * a call to MemStat.  
 *
 * If much less than 1 Mbyte is required or if the exact size of the 
 * aggregate data is known, this these routines should not be used.
 *
 * Note that the size requested will be rounded up to the nearest multiple
 * of a pointer size, i.e., the memory is pointer size aligned.
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include "Common.h"
#include "Mem.h"

/*--------------------------------------------------------------------------
 * MemCreate - Return (a pointer to) a memory pool object.
 *--------------------------------------------------------------------------*/

Mem *MemCreate()
{
    Mem *m = (Mem *) malloc(sizeof(Mem));

    m->num_blocks  = 0;  /* number of blocks allocated */
    m->bytes_left  = 0;  /* bytes left in current block */

    m->total_bytes = 0;  /* total number of bytes stored */
    m->bytes_alloc = 0;  /* total number of bytes allocated */
    m->num_over    = 0;  /* number of blocks larger than blocksize */

    return m;
}

/*--------------------------------------------------------------------------
 * MemDestroy - Destroy a memory pool object "m", and release all allocated 
 * memory to the operating system.
 *--------------------------------------------------------------------------*/

void MemDestroy(Mem *m)
{
    HYPRE_Int i;

    /* Free all blocks of memory */
    for (i=0; i<m->num_blocks; i++)
        free(m->blocks[i]);

    free(m);
}

/*--------------------------------------------------------------------------
 * MemAlloc - Return "size" bytes from the memory pool "m".  This function 
 * will return to the operating system on the following conditions:
 * 1) max block size exceeded, 2) max number of blocks exceeded,
 * 3) memory exhausted.
 *--------------------------------------------------------------------------*/

char *MemAlloc(Mem *m, HYPRE_Int size)
{
    HYPRE_Int req;
    char *p;

    /* Align on 16-byte boundary */
    size = ((size + 15) / 16) * 16;

    if (m->bytes_left < size)
    {
        /* Allocate a new block */
        if (m->num_blocks+1 > MEM_MAXBLOCKS)
        {
	    hypre_printf("MemAlloc: max number of blocks %d exceeded.\n",
	        MEM_MAXBLOCKS);
	    PARASAILS_EXIT;
        }

	/* Size of requested block */
	req = MAX(size, MEM_BLOCKSIZE);

        m->avail = (char *) malloc(req);

        if (m->avail == NULL)
        {
	    hypre_printf("MemAlloc: request for %d bytes failed.\n", req);
	    PARASAILS_EXIT;
        }

        m->blocks[m->num_blocks] = m->avail;
        m->num_blocks++;
        m->bytes_left = req;
	m->total_bytes += size;
	m->bytes_alloc += req;
	if (req > MEM_BLOCKSIZE)
	    m->num_over++;
    }

    p = m->avail;
    m->avail += size;
    m->bytes_left -= size;
    m->total_bytes += size;

    return p;
}

/*--------------------------------------------------------------------------
 * MemStat - Print statistics about memory pool "m" to stream "stream" with 
 * a descriptive message "msg".
 *--------------------------------------------------------------------------*/

void MemStat(Mem *m, FILE *stream, char *msg)
{
    hypre_fprintf(stream, "****** Mem: %s ******\n", msg);
    hypre_fprintf(stream, "num_blocks : %d\n", m->num_blocks);
    hypre_fprintf(stream, "num_over   : %d\n", m->num_over);
    hypre_fprintf(stream, "total_bytes: %ld\n", m->total_bytes);
    hypre_fprintf(stream, "bytes_alloc: %ld\n", m->bytes_alloc);
    if (m->bytes_alloc != 0)
        hypre_fprintf(stream, "efficiency : %f\n", m->total_bytes / 
	    (double) m->bytes_alloc);
    hypre_fprintf(stream, "*********************\n");
    fflush(stream);
}

