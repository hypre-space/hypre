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
 * Box allocation routines.  These hopefully increase efficiency
 * and reduce memory fragmentation.
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * Box memory data structure and static variables used to manage free
 * list and blocks to be freed by the finalization routine.
 *--------------------------------------------------------------------------*/

union box_memory
{
   union box_memory *d_next;
   hypre_Box         d_box;
};

static union box_memory *s_free      = NULL;
static union box_memory *s_finalize  = NULL;
static int               s_at_a_time = 1000;
static int               s_count     = 0;

/*--------------------------------------------------------------------------
 * Allocate a new block of memory and thread it into the free list.  The
 * first block will always be put on the finalize list to be freed by
 * the hypre_BoxFinalizeMemory() routine to remove memory leaks.
 *--------------------------------------------------------------------------*/

static int
hypre_AllocateBoxBlock()
{
   int               ierr = 0;
   union box_memory *ptr;
   int               i;

   ptr = hypre_TAlloc(union box_memory, s_at_a_time);
   ptr[0].d_next = s_finalize;
   s_finalize = &ptr[0];

   for (i = (s_at_a_time - 1); i > 0; i--)
   {
      ptr[i].d_next = s_free;
      s_free = &ptr[i];
   }

   return ierr;
}

/*--------------------------------------------------------------------------
 * Set up the allocation block size and allocate the first memory block.
 *--------------------------------------------------------------------------*/

int
hypre_BoxInitializeMemory( const int at_a_time )
{
   int ierr = 0;

   if (at_a_time > 0)
   {
      s_at_a_time = at_a_time;
   }

   return ierr;
}
   
/*--------------------------------------------------------------------------
 * Free all of the memory used to manage boxes.  This should only be
 * called at the end of the program to collect free memory.  The blocks
 * in the finalize list are freed.
 *--------------------------------------------------------------------------*/

int
hypre_BoxFinalizeMemory()
{
   int               ierr = 0;
   union box_memory *byebye;

   while (s_finalize)
   {
      byebye = s_finalize;
      s_finalize = (s_finalize -> d_next);
      hypre_TFree(byebye);
   }
   s_finalize = NULL;
   s_free = NULL;

   return ierr;
}
      
/*--------------------------------------------------------------------------
 * Allocate a box from the free list.  If no boxes exist on the free
 * list, then allocate a block of memory to repopulate the free list.
 *--------------------------------------------------------------------------*/

hypre_Box *
hypre_BoxAlloc()
{
   union box_memory *ptr = NULL;

   if (!s_free)
   {
      hypre_AllocateBoxBlock();
   }

   ptr = s_free;
   s_free = (s_free -> d_next);
   s_count++;
   return( &(ptr -> d_box) );
}

/*--------------------------------------------------------------------------
 * Put a box back on the free list.
 *--------------------------------------------------------------------------*/

int
hypre_BoxFree( hypre_Box *box )
{
   int               ierr = 0;
   union box_memory *ptr = (union box_memory *) box;

   (ptr -> d_next) = s_free;
   s_free = ptr;
   s_count--;

   if (!s_count)
   {
      hypre_BoxFinalizeMemory();
   }

   return ierr;
}

