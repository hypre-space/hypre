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
 * zzz_MAlloc
 *--------------------------------------------------------------------------*/

char *
zzz_MAlloc( int size )
{
   char *ptr;

   if (size > 0)
      ptr = malloc(size);
   else
      ptr = NULL;

   return ptr;
}

/*--------------------------------------------------------------------------
 * zzz_CAlloc
 *--------------------------------------------------------------------------*/

char *
zzz_CAlloc( int count,
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
 * zzz_ReAlloc
 *--------------------------------------------------------------------------*/

char *
zzz_ReAlloc( char *ptr,
             int   size )
{
   ptr = realloc(ptr, size);

   return ptr;
}

/*--------------------------------------------------------------------------
 * zzz_Free
 *--------------------------------------------------------------------------*/

void
zzz_Free( char *ptr )
{
   if (ptr)
   {
      free(ptr);
   }
}

/*--------------------------------------------------------------------------
 * zzz_MAllocCheck
 *--------------------------------------------------------------------------*/

char *
zzz_MAllocCheck( int   size,
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

      if (size > ZZZ_MEMORY_CHECK_SIZE)
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
 * zzz_CAllocCheck
 *--------------------------------------------------------------------------*/

char *
zzz_CAllocCheck( int   count,
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

      if (size > ZZZ_MEMORY_CHECK_SIZE)
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
 * zzz_ReAllocCheck
 *--------------------------------------------------------------------------*/

char *
zzz_ReAllocCheck( char *ptr,
                  int   size,
                  char *file,
                  int   line )
{
   ptr = realloc(ptr, size);

   if ((ptr == NULL) && (size > 0))
   {
      printf("Error: memory problem in %s at line %d\n", file, line);
   }

   if (size > ZZZ_MEMORY_CHECK_SIZE)
   {
      printf("In %s at line %d, memory alloc = %d, total = %d\n",
             file, line, size, mem_size);
   }

   return ptr;
}

