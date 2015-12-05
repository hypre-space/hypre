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



#ifdef HYPRE_USE_UMALLOC
#include "umalloc_local.h"


void *_uget_fn(Heap_t usrheap, size_t *length, HYPRE_Int *clean)
{
   void *p;
 
   *length = ((*length) / INITIAL_HEAP_SIZE) * INITIAL_HEAP_SIZE
              + INITIAL_HEAP_SIZE;
 
   *clean = _BLOCK_CLEAN;
   p = (void *) calloc(*length, 1);
   return p;
}
 
void _urelease_fn(Heap_t usrheap, void *p, size_t size)
{
   free (p);
   return;
}
#else
/* this is used only to eliminate compiler warnings */
double umalloc_empty;
#endif

