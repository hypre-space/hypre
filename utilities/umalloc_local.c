/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifdef HYPRE_USE_UMALLOC
#include "umalloc_local.h"


void *_uget_fn(Heap_t usrheap, size_t *length, int *clean)
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
int umalloc_empty;
#endif

