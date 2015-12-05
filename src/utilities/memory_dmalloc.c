/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/



/******************************************************************************
 *
 * Memory management utilities
 *
 * Routines to use "Debug Malloc Library", dmalloc
 *
 *****************************************************************************/

#ifdef HYPRE_MEMORY_DMALLOC

#include "hypre_memory.h"
#include <dmalloc.h>

char dmalloc_logpath_memory[256];

/*--------------------------------------------------------------------------
 * hypre_InitMemoryDebugDML
 *--------------------------------------------------------------------------*/

int
hypre_InitMemoryDebugDML( int id  )
{
   int  *iptr;

   /* do this to get the Debug Malloc Library started/initialized */
   iptr = hypre_TAlloc(int, 1);
   hypre_TFree(iptr);

   dmalloc_logpath = dmalloc_logpath_memory;
   sprintf(dmalloc_logpath, "dmalloc.log.%04d", id);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_FinalizeMemoryDebugDML
 *--------------------------------------------------------------------------*/

int
hypre_FinalizeMemoryDebugDML( )
{
   dmalloc_verify(NULL);

   return 0;
}

/*--------------------------------------------------------------------------
 * hypre_MAllocDML
 *--------------------------------------------------------------------------*/

char *
hypre_MAllocDML( int   size,
                 char *file,
                 int   line )
{
   char *ptr;

   if (size > 0)
      ptr = _malloc_leap(file, line, size);
   else
      ptr = NULL;

   return ptr;
}

/*--------------------------------------------------------------------------
 * hypre_CAllocDML
 *--------------------------------------------------------------------------*/

char *
hypre_CAllocDML( int   count,
                 int   elt_size,
                 char *file,
                 int   line    )
{
   char *ptr;
   int   size = count*elt_size;

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
                  int   size,
                  char *file,
                  int   line )
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
               int   line )
{
   if (ptr)
   {
      _free_leap(file, line, ptr);
   }
}

#else

/* this is used only to eliminate compiler warnings */
int hypre_empty1;

#endif
