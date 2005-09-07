/*
 * File:          sidlx_rmi_ClientSocket_Skel.c
 * Symbol:        sidlx.rmi.ClientSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.rmi.ClientSocket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_rmi_ClientSocket_IOR.h"
#include "sidlx_rmi_ClientSocket.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_ClientSocket__load(
  void);

extern
void
impl_sidlx_rmi_ClientSocket__ctor(
  /* in */ sidlx_rmi_ClientSocket self);

extern
void
impl_sidlx_rmi_ClientSocket__dtor(
  /* in */ sidlx_rmi_ClientSocket self);

extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_sidlx_rmi_ClientSocket_init(
  /* in */ sidlx_rmi_ClientSocket self,
  /* in */ const char* hostname,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_ClientSocket__set_epv(struct sidlx_rmi_ClientSocket__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_ClientSocket__ctor;
  epv->f__dtor = impl_sidlx_rmi_ClientSocket__dtor;
  epv->f_init = impl_sidlx_rmi_ClientSocket_init;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_ClientSocket__call_load(void) { 
  impl_sidlx_rmi_ClientSocket__load();
}
struct sidlx_rmi_ClientSocket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(url, _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(url,
    _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(obj);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(url, _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_ClientSocket__data*
sidlx_rmi_ClientSocket__get_data(sidlx_rmi_ClientSocket self)
{
  return (struct sidlx_rmi_ClientSocket__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_ClientSocket__set_data(
  sidlx_rmi_ClientSocket self,
  struct sidlx_rmi_ClientSocket__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
