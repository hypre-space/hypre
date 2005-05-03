/*
 * File:          bHYPRE_StructStencil_Impl.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.9.8
 * Description:   Server-side implementation for bHYPRE.StructStencil
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
 * Symbol "bHYPRE.StructStencil" (version 1.0.0)
 * 
 * Define a structured stencil for a structured problem
 * description.  More than one implementation is not envisioned,
 * thus the decision has been made to make this a class rather than
 * an interface.
 * 
 */

#include "bHYPRE_StructStencil_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
#include "mpi.h"
#include "struct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__ctor"

void
impl_bHYPRE_StructStencil__ctor(
  /*in*/ bHYPRE_StructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */

   struct bHYPRE_StructStencil__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructStencil__data, 1 );
   data -> stencil = NULL;
   data -> dim = 0;
   data -> size = 0;
   bHYPRE_StructStencil__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__dtor"

void
impl_bHYPRE_StructStencil__dtor(
  /*in*/ bHYPRE_StructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;
   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   ierr += HYPRE_StructStencilDestroy( stencil );
   assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._dtor) */
}

/*
 * Method:  SetDimension[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetDimension"

int32_t
impl_bHYPRE_StructStencil_SetDimension(
  /*in*/ bHYPRE_StructStencil self, /*in*/ int32_t dim)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;
   int size;

   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   assert( stencil == NULL );  /* can't reset dimension */
   assert( dim > 0 );
   data -> dim = dim;
   size = data -> size;

   if ( size>0 )
   {
      ierr += HYPRE_StructStencilCreate( dim, size, &stencil );
      data -> stencil = stencil;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetDimension) */
}

/*
 * Method:  SetSize[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetSize"

int32_t
impl_bHYPRE_StructStencil_SetSize(
  /*in*/ bHYPRE_StructStencil self, /*in*/ int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetSize) */
  /* Insert the implementation of the SetSize method here... */
 
   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;
   int dim;

   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   assert( stencil == NULL );  /* can't reset size */
   assert( size>0 );
   data -> size = size;
   dim = data -> dim;

   if ( dim>0 )
   {
      ierr += HYPRE_StructStencilCreate( dim, size, &stencil );
      data -> stencil = stencil;
   }

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetSize) */
}

/*
 * Method:  SetElement[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetElement"

int32_t
impl_bHYPRE_StructStencil_SetElement(
  /*in*/ bHYPRE_StructStencil self, /*in*/ int32_t index,
    /*in*/ struct sidl_int__array* offset)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetElement) */
  /* Insert the implementation of the SetElement method here... */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;

   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   assert( stencil != NULL );

   ierr += HYPRE_StructStencilSetElement( stencil, index,
                                          sidlArrayAddr1( offset, 0 ) );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetElement) */
}
