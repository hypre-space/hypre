/*BHEADER**********************************************************************
 * (c) 1999   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

#ifndef UMALLOC_LOCAL_HEADER
#define UMALLOC_LOCAL_HEADER

#ifdef HYPRE_USE_UMALLOC
#include <umalloc.h>

#define MAX_THREAD_COUNT 10 
#define INITIAL_HEAP_SIZE 500000
#define GET_MISS_COUNTS

struct upc_struct
{
	Heap_t myheap;
};

void *_uinitial_block[MAX_THREAD_COUNT];
struct upc_struct _uparam[MAX_THREAD_COUNT];

int _uheapReleasesCount=0;
int _uheapGetsCount=0;

void *_uget_fn(Heap_t usrheap, size_t *length, int *clean);
void _urelease_fn(Heap_t usrheap, void *p, size_t size);

#endif

#endif
