/*
 * File:          sidlx_rmi_Common_Skel.c
 * Symbol:        sidlx.rmi.Common-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for sidlx.rmi.Common
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "sidlx_rmi_Common_IOR.h"
#include "sidlx_rmi_Common.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_Common__load(
  void);

extern
void
impl_sidlx_rmi_Common__ctor(
  /* in */ sidlx_rmi_Common self);

extern
void
impl_sidlx_rmi_Common__dtor(
  /* in */ sidlx_rmi_Common self);

extern
int32_t
impl_sidlx_rmi_Common_fork(
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_Common_gethostbyname(
  /* in */ const char* hostname,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_Common__set_epv(struct sidlx_rmi_Common__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_Common__ctor;
  epv->f__dtor = impl_sidlx_rmi_Common__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_Common__set_sepv(struct sidlx_rmi_Common__sepv *sepv)
{
  sepv->f_fork = impl_sidlx_rmi_Common_fork;
  sepv->f_gethostbyname = impl_sidlx_rmi_Common_gethostbyname;
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_Common__call_load(void) { 
  impl_sidlx_rmi_Common__load();
}
struct sidlx_rmi_Common__object* 
  skel_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(url, _ex);
}

char* skel_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj) { 
  return impl_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(url, _ex);
}

char* skel_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_Common_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_Common__data*
sidlx_rmi_Common__get_data(sidlx_rmi_Common self)
{
  return (struct sidlx_rmi_Common__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_Common__set_data(
  sidlx_rmi_Common self,
  struct sidlx_rmi_Common__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
