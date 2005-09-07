/*
 * File:          sidlx_rmi_SimpleServer_Skel.c
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.rmi.SimpleServer
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_rmi_SimpleServer_IOR.h"
#include "sidlx_rmi_SimpleServer.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimpleServer__load(
  void);

extern
void
impl_sidlx_rmi_SimpleServer__ctor(
  /* in */ sidlx_rmi_SimpleServer self);

extern
void
impl_sidlx_rmi_SimpleServer__dtor(
  /* in */ sidlx_rmi_SimpleServer self);

extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_SimpleServer_setPort(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleServer_run(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimpleServer__set_epv(struct sidlx_rmi_SimpleServer__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimpleServer__ctor;
  epv->f__dtor = impl_sidlx_rmi_SimpleServer__dtor;
  epv->f_setPort = impl_sidlx_rmi_SimpleServer_setPort;
  epv->f_run = impl_sidlx_rmi_SimpleServer_run;
  epv->f_serviceRequest = NULL;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimpleServer__call_load(void) { 
  impl_sidlx_rmi_SimpleServer__load();
}
struct sidl_SIDLException__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_SIDLException(url, _ex);
}

char* skel_sidlx_rmi_SimpleServer_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) { 
  return impl_sidlx_rmi_SimpleServer_fgetURL_sidl_SIDLException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_SimpleServer_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_SimpleServer_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(url, _ex);
}

char* skel_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj) { 
  return impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_SimpleServer(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_SimpleServer__data*
sidlx_rmi_SimpleServer__get_data(sidlx_rmi_SimpleServer self)
{
  return (struct sidlx_rmi_SimpleServer__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimpleServer__set_data(
  sidlx_rmi_SimpleServer self,
  struct sidlx_rmi_SimpleServer__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
