/*
 * File:          bHYPRE_SStructStencil_Impl.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.9.8
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 * 
 */

#include "bHYPRE_SStructStencil_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__ctor"

void
impl_bHYPRE_SStructStencil__ctor(
  /*in*/ bHYPRE_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_SStructStencil__data * data;
   data = hypre_CTAlloc( struct bHYPRE_SStructStencil__data, 1 );
   data -> stencil = NULL;
   bHYPRE_SStructStencil__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__dtor"

void
impl_bHYPRE_SStructStencil__dtor(
  /*in*/ bHYPRE_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil stencil;
   data = bHYPRE_SStructStencil__get_data( self );
   stencil = data -> stencil;
   ierr += HYPRE_SStructStencilDestroy( stencil );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._dtor) */
}

/*
 * Set the number of spatial dimensions and stencil entries.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetNumDimSize"

int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  /*in*/ bHYPRE_SStructStencil self, /*in*/ int32_t ndim, /*in*/ int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetNumDimSize) */
  /* Insert the implementation of the SetNumDimSize method here... */
   /* note: StructStencil does this with two functions, SStruct with one.
      But StructStencil_SetElement and SStructStencil_SetEntry are inherently
      different, and no other stencil classes are expected to exist, so there
      is little point in reconciling this */
 
   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil stencil;

   data = bHYPRE_SStructStencil__get_data( self );
   stencil = data -> stencil;

   ierr += HYPRE_SStructStencilCreate( ndim, size, &stencil );
   data -> stencil = stencil;

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetNumDimSize) */
}

/*
 * Set a stencil entry.
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetEntry"

int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  /*in*/ bHYPRE_SStructStencil self, /*in*/ int32_t entry,
    /*in*/ struct sidl_int__array* offset, /*in*/ int32_t var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetEntry) */
  /* Insert the implementation of the SetEntry method here... */
 
   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil Hstencil;

   data = bHYPRE_SStructStencil__get_data( self );
   Hstencil = data -> stencil;

   ierr += HYPRE_SStructStencilSetEntry( Hstencil, entry,
                                         sidlArrayAddr1( offset, 0 ), var );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetEntry) */
}
