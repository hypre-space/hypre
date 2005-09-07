/*
 * File:          sidlx_rmi_SimHandle_Skel.c
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_rmi_SimHandle_IOR.h"
#include "sidlx_rmi_SimHandle.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimHandle__load(
  void);

extern
void
impl_sidlx_rmi_SimHandle__ctor(
  /* in */ sidlx_rmi_SimHandle self);

extern
void
impl_sidlx_rmi_SimHandle__dtor(
  /* in */ sidlx_rmi_SimHandle self);

extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
sidl_bool
impl_sidlx_rmi_SimHandle_initCreate(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimHandle_initConnect(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getProtocol(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getObjectID(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getURL(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Invocation
impl_sidlx_rmi_SimHandle_createInvocation(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* methodName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimHandle_close(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimHandle__set_epv(struct sidlx_rmi_SimHandle__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimHandle__ctor;
  epv->f__dtor = impl_sidlx_rmi_SimHandle__dtor;
  epv->f_initCreate = impl_sidlx_rmi_SimHandle_initCreate;
  epv->f_initConnect = impl_sidlx_rmi_SimHandle_initConnect;
  epv->f_getProtocol = impl_sidlx_rmi_SimHandle_getProtocol;
  epv->f_getObjectID = impl_sidlx_rmi_SimHandle_getObjectID;
  epv->f_getURL = impl_sidlx_rmi_SimHandle_getURL;
  epv->f_createInvocation = impl_sidlx_rmi_SimHandle_createInvocation;
  epv->f_close = impl_sidlx_rmi_SimHandle_close;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimHandle__call_load(void) { 
  impl_sidlx_rmi_SimHandle__load();
}
struct sidl_rmi_InstanceHandle__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(obj);
}

struct sidlx_rmi_SimHandle__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_SimHandle__data*
sidlx_rmi_SimHandle__get_data(sidlx_rmi_SimHandle self)
{
  return (struct sidlx_rmi_SimHandle__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimHandle__set_data(
  sidlx_rmi_SimHandle self,
  struct sidlx_rmi_SimHandle__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
