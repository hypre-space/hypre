/*
 * File:          bHYPRE_StructStencil_Skel.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "bHYPRE_StructStencil_IOR.h"
#include "bHYPRE_StructStencil.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructStencil__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructStencil__ctor(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructStencil__ctor2(
  /* in */ bHYPRE_StructStencil self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_bHYPRE_StructStencil__dtor(
  /* in */ bHYPRE_StructStencil self,
  /* out */ sidl_BaseInterface *_ex);

extern
bHYPRE_StructStencil
impl_bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fcast_bHYPRE_StructStencil(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern
int32_t
impl_bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* out */ sidl_BaseInterface *_ex);

extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fcast_bHYPRE_StructStencil(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_bHYPRE_StructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
static int32_t
skel_bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in rarray[dim] */ struct sidl_int__array* offset,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_int__array* offset_proxy = sidl_int__array_ensure(offset, 1,
    sidl_column_major_order);
  int32_t* offset_tmp = offset_proxy->d_firstElement;
  int32_t dim = sidlLength(offset_proxy,0);
  _return =
    impl_bHYPRE_StructStencil_SetElement(
      self,
      index,
      offset_tmp,
      dim,
      _ex);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructStencil__set_epv(struct bHYPRE_StructStencil__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructStencil__ctor;
  epv->f__ctor2 = impl_bHYPRE_StructStencil__ctor2;
  epv->f__dtor = impl_bHYPRE_StructStencil__dtor;
  epv->f_SetDimension = impl_bHYPRE_StructStencil_SetDimension;
  epv->f_SetSize = impl_bHYPRE_StructStencil_SetSize;
  epv->f_SetElement = skel_bHYPRE_StructStencil_SetElement;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructStencil__set_sepv(struct bHYPRE_StructStencil__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_StructStencil_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_StructStencil__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_bHYPRE_StructStencil__load(&_throwaway_exception);
}
struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(url, ar, _ex);
}

struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructStencil_fcast_bHYPRE_StructStencil(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fcast_bHYPRE_StructStencil(bi, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructStencil_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructStencil_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructStencil_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_bHYPRE_StructStencil_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fcast_sidl_RuntimeException(bi, _ex);
}

struct bHYPRE_StructStencil__data*
bHYPRE_StructStencil__get_data(bHYPRE_StructStencil self)
{
  return (struct bHYPRE_StructStencil__data*)(self ? self->d_data : NULL);
}

void bHYPRE_StructStencil__set_data(
  bHYPRE_StructStencil self,
  struct bHYPRE_StructStencil__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
