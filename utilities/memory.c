/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * Memory management utilities
 *
 *****************************************************************************/

#include "memory.h"
#include <stdio.h>

/******************************************************************************
 *
 * Standard routines
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/

int
hypre_OutOfMemory( int size )
{
   printf("Out of memory trying to allocate %d bytes\n", size);
   fflush(stdout);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_MAlloc( int size )
{
   char *ptr;

   if (size > 0)
   {
      ptr = malloc(size);

#if 0
      if (ptr == NULL)
        hypre_OutOfMemory(size);
#endif
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_CAlloc( int count,
              int elt_size )
{
   char *ptr;
   int   size = count*elt_size;

   if (size > 0)
   {
      ptr = calloc(count, elt_size);

#if 0
      if (ptr == NULL)
        hypre_OutOfMemory(count * elt_size);
#endif
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_ReAlloc( char *ptr,
               int   size )
{
   ptr = realloc(ptr, size);

#if 0
   if (ptr == NULL)
     hypre_OutOfMemory(size);
#endif

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

void
hypre_Free( char *ptr )
{
   if (ptr)
   {
      free(ptr);
   }
}


