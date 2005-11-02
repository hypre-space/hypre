/*
 * File:          bHYPRE_SStructStencil_Skel.c
 * Symbol:        bHYPRE.SStructStencil-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for bHYPRE.SStructStencil
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "bHYPRE_SStructStencil_IOR.h"
#include "bHYPRE_SStructStencil.h"
#include <stddef.h>

extern
void
impl_bHYPRE_SStructStencil__load(
  void);

extern
void
impl_bHYPRE_SStructStencil__ctor(
  /* in */ bHYPRE_SStructStencil self);

extern
void
impl_bHYPRE_SStructStencil__dtor(
  /* in */ bHYPRE_SStructStencil self);

extern
bHYPRE_SStructStencil
impl_bHYPRE_SStructStencil_Create(
  /* in */ int32_t ndim,
  /* in */ int32_t size);

extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_bHYPRE_SStructStencil_SetNumDimSize(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t ndim,
  /* in */ int32_t size);

extern
int32_t
impl_bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ int32_t* offset,
  /* in */ int32_t dim,
  /* in */ int32_t var);

extern struct bHYPRE_SStructStencil__object* 
  impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_bHYPRE_SStructStencil_SetEntry(
  /* in */ bHYPRE_SStructStencil self,
  /* in */ int32_t entry,
  /* in rarray[dim] */ struct sidl_int__array* offset,
/* in */ int32_t var)
{
  int32_t _return;
  struct sidl_int__array* offset_proxy = sidl_int__array_ensure(offset, 1,
    sidl_column_major_order);
  int32_t* offset_tmp = offset_proxy->d_firstElement;
  int32_t dim = sidlLength(offset_proxy,0);
  _return =
    impl_bHYPRE_SStructStencil_SetEntry(
      self,
      entry,
      offset_tmp,
      dim,
      var);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructStencil__set_epv(struct bHYPRE_SStructStencil__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_SStructStencil__ctor;
  epv->f__dtor = impl_bHYPRE_SStructStencil__dtor;
  epv->f_SetNumDimSize = impl_bHYPRE_SStructStencil_SetNumDimSize;
  epv->f_SetEntry = skel_bHYPRE_SStructStencil_SetEntry;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_SStructStencil__set_sepv(struct bHYPRE_SStructStencil__sepv *sepv)
{
  sepv->f_Create = impl_bHYPRE_SStructStencil_Create;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_SStructStencil__call_load(void) { 
  impl_bHYPRE_SStructStencil__load();
}
struct bHYPRE_SStructStencil__object* 
  skel_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructStencil_fconnect_bHYPRE_SStructStencil(url, _ex);
}

char* skel_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(struct 
  bHYPRE_SStructStencil__object* obj) { 
  return impl_bHYPRE_SStructStencil_fgetURL_bHYPRE_SStructStencil(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructStencil_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_SStructStencil_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructStencil_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_SStructStencil_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_SStructStencil_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_SStructStencil__data*
bHYPRE_SStructStencil__get_data(bHYPRE_SStructStencil self)
{
  return (struct bHYPRE_SStructStencil__data*)(self ? self->d_data : NULL);
}

void bHYPRE_SStructStencil__set_data(
  bHYPRE_SStructStencil self,
  struct bHYPRE_SStructStencil__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
