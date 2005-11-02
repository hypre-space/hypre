/*
 * File:          bHYPRE_StructStencil_Skel.c
 * Symbol:        bHYPRE.StructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for bHYPRE.StructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "bHYPRE_StructStencil_IOR.h"
#include "bHYPRE_StructStencil.h"
#include <stddef.h>

extern
void
impl_bHYPRE_StructStencil__load(
  void);

extern
void
impl_bHYPRE_StructStencil__ctor(
  /* in */ bHYPRE_StructStencil self);

extern
void
impl_bHYPRE_StructStencil__dtor(
  /* in */ bHYPRE_StructStencil self);

extern
bHYPRE_StructStencil
impl_bHYPRE_StructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size);

extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_StructStencil_SetDimension(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t dim);

extern
int32_t
impl_bHYPRE_StructStencil_SetSize(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t size);

extern
int32_t
impl_bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim);

extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct bHYPRE_StructStencil__object* 
  impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_StructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_StructStencil_SetElement(
  /* in */ bHYPRE_StructStencil self,
  /* in */ int32_t index,
/* in rarray[dim] */ struct sidl_int__array* offset)
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
      dim);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_StructStencil__set_epv(struct bHYPRE_StructStencil__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_StructStencil__ctor;
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
  impl_bHYPRE_StructStencil__load();
}
struct sidl_ClassInfo__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_StructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_StructStencil_fgetURL_sidl_ClassInfo(obj);
}

struct bHYPRE_StructStencil__object* 
  skel_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_bHYPRE_StructStencil(url, _ex);
}

char* skel_bHYPRE_StructStencil_fgetURL_bHYPRE_StructStencil(struct 
  bHYPRE_StructStencil__object* obj) { 
  return impl_bHYPRE_StructStencil_fgetURL_bHYPRE_StructStencil(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_StructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_StructStencil_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_StructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_StructStencil_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_StructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_StructStencil_fgetURL_sidl_BaseClass(obj);
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
