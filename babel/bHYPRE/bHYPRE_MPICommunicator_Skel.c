/*
 * File:          bHYPRE_MPICommunicator_Skel.c
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "bHYPRE_MPICommunicator_IOR.h"
#include "bHYPRE_MPICommunicator.h"
#include <stddef.h>

extern
void
impl_bHYPRE_MPICommunicator__load(
  void);

extern
void
impl_bHYPRE_MPICommunicator__ctor(
  /* in */ bHYPRE_MPICommunicator self);

extern
void
impl_bHYPRE_MPICommunicator__dtor(
  /* in */ bHYPRE_MPICommunicator self);

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateC(
  /* in */ void* mpi_comm);

extern
bHYPRE_MPICommunicator
impl_bHYPRE_MPICommunicator_CreateF(
  /* in */ void* mpi_comm);

extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct bHYPRE_MPICommunicator__object* 
  impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_MPICommunicator__set_epv(struct bHYPRE_MPICommunicator__epv *epv)
{
  epv->f__ctor = impl_bHYPRE_MPICommunicator__ctor;
  epv->f__dtor = impl_bHYPRE_MPICommunicator__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
bHYPRE_MPICommunicator__set_sepv(struct bHYPRE_MPICommunicator__sepv *sepv)
{
  sepv->f_CreateC = impl_bHYPRE_MPICommunicator_CreateC;
  sepv->f_CreateF = impl_bHYPRE_MPICommunicator_CreateF;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void bHYPRE_MPICommunicator__call_load(void) { 
  impl_bHYPRE_MPICommunicator__load();
}
struct bHYPRE_MPICommunicator__object* 
  skel_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_MPICommunicator_fconnect_bHYPRE_MPICommunicator(url, _ex);
}

char* skel_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(struct 
  bHYPRE_MPICommunicator__object* obj) { 
  return impl_bHYPRE_MPICommunicator_fgetURL_bHYPRE_MPICommunicator(obj);
}

struct sidl_ClassInfo__object* 
  skel_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_MPICommunicator_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_bHYPRE_MPICommunicator_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_bHYPRE_MPICommunicator_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_bHYPRE_MPICommunicator_fgetURL_sidl_BaseClass(obj);
}

struct bHYPRE_MPICommunicator__data*
bHYPRE_MPICommunicator__get_data(bHYPRE_MPICommunicator self)
{
  return (struct bHYPRE_MPICommunicator__data*)(self ? self->d_data : NULL);
}

void bHYPRE_MPICommunicator__set_data(
  bHYPRE_MPICommunicator self,
  struct bHYPRE_MPICommunicator__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
