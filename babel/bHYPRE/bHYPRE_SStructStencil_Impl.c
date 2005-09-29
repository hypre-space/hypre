/*
 * File:          bHYPRE_SStructStencil_Impl.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for bHYPRE.SStructStencil
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
 * Symbol "bHYPRE.SStructStencil" (version 1.0.0)
 * 
 * The semi-structured grid stencil class.
 * 
 */

#include "bHYPRE_SStructStencil_Impl.h"

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional includes or other arbitrary code here... */
#include <assert.h>
/*#include "mpi.h"*/
#include "sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._load) */
  /* Insert-Code-Here {bHYPRE.SStructStencil._load} (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__ctor(
  /* in */ bHYPRE_SStructStencil self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_bHYPRE_SStructStencil__dtor(
  /* in */ bHYPRE_SStructStencil self)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._dtor) */
  /* Insert the implementation of the destructor method here... */

   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil stencil;
   data = bHYPRE_SStructStencil__get_data( self );
   stencil = data -> stencil;
   ierr += HYPRE_SStructStencilDestroy( stencil );
   hypre_assert( ierr==0 );
   hypre_TFree( data );

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._dtor) */
}

/*
 * Method:  Create[]
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_Create"

#ifdef __cplusplus
extern "C"
#endif
bHYPRE_SStructStencil
impl_bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.Create) */
  /* Insert-Code-Here {bHYPRE.SStructStencil.Create} (Create method) */

   int ierr = 0;
   bHYPRE_SStructStencil stencil;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil Hstencil;

   stencil = bHYPRE_SStructStencil__create();
   data = bHYPRE_SStructStencil__get_data( stencil );

   ierr += HYPRE_SStructStencilCreate( ndim, size, &Hstencil );
   data->stencil = Hstencil;

   return stencil;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.Create) */
}

/*
 * Set the number of spatial dimensions and stencil entries.
 * DEPRECATED, use Create:
 * 
 */

#undef __FUNC__
#define __FUNC__ "impl_bHYPRE_SStructStencil_SetNumDimSize"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetNumDimSize) */
  /* Insert the implementation of the SetNumDimSize method here... */
   /* note: StructStencil does this with two functions, SStruct with one.
      But StructStencil_SetElement and SStructStencil_SetEntry are inherently
      different, and no other stencil classes are expected to exist, so there
      is little point in reconciling this */
 
   /* DEPRECATED   use _Create */

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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var)
{
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil.SetEntry) */
  /* Insert the implementation of the SetEntry method here... */
 
   int ierr = 0;
   struct bHYPRE_SStructStencil__data * data;
   HYPRE_SStructStencil Hstencil;

   data = bHYPRE_SStructStencil__get_data( self );
   Hstencil = data -> stencil;

   ierr += HYPRE_SStructStencilSetEntry( Hstencil, entry,
                                         offset, var );

   return ierr;

  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil.SetEntry) */
}
/* Babel internal methods, Users should not edit below this line. */
struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex) {
  return bHYPRE_SStructStencil__connect(url, _ex);
}
char * impl_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj) {
  return bHYPRE_SStructStencil__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
