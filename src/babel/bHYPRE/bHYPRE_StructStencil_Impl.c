/*
 * File:          bHYPRE_StructStencil_Impl.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
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
 */

#include "bHYPRE_StructStencil_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._includes) */
/* Put additional includes or other arbitrary code here... */



#include "hypre_babel_exception_handler.h"
/*#include "mpi.h"*/
#include "_hypre_struct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructStencil__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._load) */
  /* Insert-Code-Here {bHYPRE.StructStencil._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructStencil__ctor(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._ctor) */
  /* Insert the implementation of the constructor method here... */

   /* User calls of __create are DEPRECATED.  Instead, call _Create, which
      also calls this function. */

   struct bHYPRE_StructStencil__data * data;
   data = hypre_CTAlloc( struct bHYPRE_StructStencil__data, 1 );
   data -> stencil = NULL;
   data -> dim = 0;
   data -> size = 0;
   bHYPRE_StructStencil__set_data( self, data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructStencil__ctor2(
  /* in */ bHYPRE_StructStencil self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._ctor2) */
    /* Insert-Code-Here {bHYPRE.StructStencil._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructStencil__dtor(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;
   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   ierr += HYPRE_StructStencilDestroy( stencil );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._dtor) */
  }
}

/*
 *  This function is the preferred way to create a Struct Stencil.
 * You provide the number of spatial dimensions and the number of
 * stencil entries.  
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructStencil
impl_bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.Create) */
  /* Insert-Code-Here {bHYPRE.StructStencil.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructStencil stencil;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil Hstencil;

   stencil = bHYPRE_StructStencil__create(_ex); SIDL_CHECK(*_ex);
   data = bHYPRE_StructStencil__get_data( stencil );
   hypre_assert( ndim > 0 );
   hypre_assert( size > 0 );
   data->dim = ndim;
   data->size = size;

   ierr += HYPRE_StructStencilCreate( ndim, size, &Hstencil );
   hypre_assert( ierr==0 );
   data->stencil = Hstencil;

   return stencil;

   hypre_babel_exception_no_return(_ex);
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.Create) */
  }
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_Destroy"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructStencil_Destroy(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.Destroy) */
    /* Insert-Code-Here {bHYPRE.StructStencil.Destroy} (Destroy method) */
     bHYPRE_StructStencil_deleteRef(self,_ex);
     return;
    /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.Destroy) */
  }
}

/*
 *  Set the number of dimensions.  DEPRECATED, use StructStencilCreate 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetDimension"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetDimension) */
  /* Insert the implementation of the SetDimension method here... */

   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;
   int size;

   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   hypre_assert( stencil == NULL );  /* can't reset dimension */
   hypre_assert( dim > 0 );
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
}

/*
 *  Set the number of stencil entries.
 * DEPRECATED, use StructStencilCreate 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetSize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetSize) */
  /* Insert the implementation of the SetSize method here... */
 
   /* DEPRECATED   use _Create */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;
   int dim;

   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   hypre_assert( stencil == NULL );  /* can't reset size */
   hypre_assert( size>0 );
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
}

/*
 *  Set a stencil element.  Specify the stencil index, and an array of
 * offsets.  "offset" is an array of length "dim", the number of spatial
 * dimensions. 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetElement"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.SetElement) */
  /* Insert the implementation of the SetElement method here... */

   int ierr = 0;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil stencil;

   data = bHYPRE_StructStencil__get_data( self );
   stencil = data -> stencil;
   hypre_assert( stencil != NULL );

   ierr += HYPRE_StructStencilSetElement( stencil, index,
                                          offset );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.SetElement) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return bHYPRE_StructStencil__connectI(url, ar, _ex);
}
struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fcast_bHYPRE_StructStencil(void* bi,
  sidl_BaseInterface* _ex) {
  return bHYPRE_StructStencil__cast(bi, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
