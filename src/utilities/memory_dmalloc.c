/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Memory management utilities
 *
 * Routines to use "Debug Malloc Library", dmalloc
 *
 *****************************************************************************/

#ifdef HYPRE_MEMORY_DMALLOC

#include "memory.h"
#include <dmalloc.h>

char dmalloc_logpath_memory[256];

/*--------------------------------------------------------------------------
 * hypre_InitMemoryDebugDML
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_InitMemoryDebugDML( HYPRE_Int id  )
{
   HYPRE_Int  *iptr;

   /* do this to get the Debug Malloc Library started/initialized */
   iptr = hypre_TAlloc(HYPRE_Int,  1, HYPRE_MEMORY_HOST);
   hypre_TFree(iptr, HYPRE_MEMORY_HOST);

   dmalloc_logpath = dmalloc_logpath_memory;
   hypre_sprintf(dmalloc_logpath, "dmalloc.log.%04d", id);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_FinalizeMemoryDebugDML
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_FinalizeMemoryDebugDML( )
{
   dmalloc_verify(NULL);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MAllocDML
 *--------------------------------------------------------------------------*/

char *
hypre_MAllocDML( HYPRE_Int   size,
                 char *file,
                 HYPRE_Int   line )
{
   char *ptr;

   if (size > 0)
   {
      ptr = _malloc_leap(file, line, size);
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAllocDML
 *--------------------------------------------------------------------------*/

char *
hypre_CAllocDML( HYPRE_Int   count,
                 HYPRE_Int   elt_size,
                 char *file,
                 HYPRE_Int   line    )
{
   char *ptr;
   HYPRE_Int   size = count * elt_size;

   if (size > 0)
   {
      ptr = _calloc_leap(file, line, count, elt_size);
   }
   else
   {
      ptr = NULL;
   }

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_ReAllocDML
 *--------------------------------------------------------------------------*/

char *
hypre_ReAllocDML( char *ptr,
                  HYPRE_Int   size,
                  char *file,
                  HYPRE_Int   line )
{
   ptr = _realloc_leap(file, line, ptr, size);

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_FreeDML
 *--------------------------------------------------------------------------*/

void
hypre_FreeDML( char *ptr,
               char *file,
               HYPRE_Int   line )
{
   if (ptr)
   {
      _free_leap(file, line, ptr);
   }
}

#else

/* this is used only to eliminate compiler warnings */
char hypre_memory_dmalloc_empty;

#endif
