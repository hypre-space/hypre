/*
 * File:          bHYPRE_SStructStencil_Impl.h
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_bHYPRE_SStructStencil_Impl_h
#define included_bHYPRE_SStructStencil_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_bHYPRE_SStructStencil_h
#include "bHYPRE_SStructStencil.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif

/* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._includes) */
/* Put additional include files here... */


#include "_hypre_sstruct_mv.h"
/* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._includes) */

/*
 * Private data for class bHYPRE.SStructStencil
 */

struct bHYPRE_SStructStencil__data {
  /* DO-NOT-DELETE splicer.begin(bHYPRE.SStructStencil._data) */
  /* Put private data members here... */
   HYPRE_SStructStencil  stencil;
  /* DO-NOT-DELETE splicer.end(bHYPRE.SStructStencil._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct bHYPRE_SStructStencil__data*
bHYPRE_SStructStencil__get_data(
  bHYPRE_SStructStencil);

extern void
bHYPRE_SStructStencil__set_data(
  bHYPRE_SStructStencil,
  struct bHYPRE_SStructStencil__data*);

extern
void
impl_bHYPRE_SStructStencil__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructStencil__ctor(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructStencil__ctor2(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_SStructStencil__dtor(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern
bHYPRE_SStructStencil
impl_bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fcast_bHYPRE_SStructStencil(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_bHYPRE_SStructStencil_Destroy(
  /* in */ bHYPRE_SStructStencil self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fcast_bHYPRE_SStructStencil(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_SStructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
