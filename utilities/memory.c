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

/* used in Check routines */
static int mem_size = 0;

/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

char *
hypre_MAlloc( int size )
{
   char *ptr;

   if (size > 0)
      ptr = malloc(size);
   else
      ptr = NULL;

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

/*--------------------------------------------------------------------------
 * hypre_MAllocCheck
 *--------------------------------------------------------------------------*/

char *
hypre_MAllocCheck( int   size,
                   char *file,
                   int   line )
{
   char *ptr;

   if (size > 0)
   {
      ptr = malloc(size);
      mem_size += size;

      if (ptr == NULL)
      {
	 printf("Error: out of memory in %s at line %d\n", file, line);
      }

      if (size > HYPRE_MEMORY_CHECK_SIZE)
      {
	 printf("In %s at line %d, memory alloc = %d, total = %d\n",
                file, line, size, mem_size);
      }
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}


/*--------------------------------------------------------------------------
 * hypre_CAllocCheck
 *--------------------------------------------------------------------------*/

char *
hypre_CAllocCheck( int   count,
                   int   elt_size,
                   char *file,
                   int   line    )
{
   char *ptr;
   int   size = count*elt_size;

   if (size > 0)
   {
      ptr = calloc(count, elt_size);
      mem_size += size;

      if (ptr == NULL)
      {
	 printf("Error: out of memory in %s at line %d\n", file, line);
      }

      if (size > HYPRE_MEMORY_CHECK_SIZE)
      {
	 printf("In %s at line %d, memory alloc = %d, total = %d\n",
                file, line, size, mem_size);
      }
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_ReAllocCheck
 *--------------------------------------------------------------------------*/

char *
hypre_ReAllocCheck( char *ptr,
                    int   size,
                    char *file,
                    int   line )
{
   ptr = realloc(ptr, size);

   if ((ptr == NULL) && (size > 0))
   {
      printf("Error: memory problem in %s at line %d\n", file, line);
   }

   if (size > HYPRE_MEMORY_CHECK_SIZE)
   {
      printf("In %s at line %d, memory alloc = %d, total = %d\n",
             file, line, size, mem_size);
   }

   return ptr;
}

