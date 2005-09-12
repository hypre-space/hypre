/*
 * File:          bHYPRE_StructStencil_Impl.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
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
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_StructStencil__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil._load) */
  /* Insert-Code-Here {bHYPRE.StructStencil._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil._load) */
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
  /* in */ bHYPRE_StructStencil self)
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
  /* in */ bHYPRE_StructStencil self)
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

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_StructStencil
impl_bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.StructStencil.Create) */
  /* Insert-Code-Here {bHYPRE.StructStencil.Create} (Create method) */

   int ierr = 0;
   bHYPRE_StructStencil stencil;
   struct bHYPRE_StructStencil__data * data;
   HYPRE_StructStencil Hstencil;

   stencil = bHYPRE_StructStencil__create();
   data = bHYPRE_StructStencil__get_data( stencil );
   hypre_assert( ndim > 0 );
   hypre_assert( size > 0 );
   data->dim = ndim;
   data->size = size;

   ierr += HYPRE_StructStencilCreate( ndim, size, &Hstencil );
   hypre_assert( ierr==0 );
   data->stencil = Hstencil;

   return stencil;

  /* DO-NOT-DELETE splicer.end(bHYPRE.StructStencil.Create) */
}

/*
 * Method:  SetDimension[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetDimension"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim)
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

/*
 * Method:  SetSize[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_StructStencil_SetSize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size)
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

/*
 * Method:  SetElement[]
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
  /* in */ int32_t dim)
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
/* Babel internal methods, Users should not edit below this line. */
struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_StructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_StructStencil__connect(url, _ex);
}
char * impl_bHYPRE_StructStencil_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj) {
  return bHYPRE_StructStencil__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_StructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_StructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
