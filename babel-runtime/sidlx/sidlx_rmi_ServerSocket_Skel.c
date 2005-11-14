/*
 * File:          sidlx_rmi_ServerSocket_Skel.c
 * Symbol:        sidlx.rmi.ServerSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for sidlx.rmi.ServerSocket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_rmi_ServerSocket_IOR.h"
#include "sidlx_rmi_ServerSocket.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_ServerSocket__load(
  void);

extern
void
impl_sidlx_rmi_ServerSocket__ctor(
  /* in */ sidlx_rmi_ServerSocket self);

extern
void
impl_sidlx_rmi_ServerSocket__dtor(
  /* in */ sidlx_rmi_ServerSocket self);

extern struct sidlx_rmi_ServerSocket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_ServerSocket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_ServerSocket(struct 
  sidlx_rmi_ServerSocket__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_ServerSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_sidlx_rmi_ServerSocket_init(
  /* in */ sidlx_rmi_ServerSocket self,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex);

extern
sidlx_rmi_Socket
impl_sidlx_rmi_ServerSocket_accept(
  /* in */ sidlx_rmi_ServerSocket self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_rmi_ServerSocket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_ServerSocket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_ServerSocket(struct 
  sidlx_rmi_ServerSocket__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_ServerSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_ServerSocket__set_epv(struct sidlx_rmi_ServerSocket__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_ServerSocket__ctor;
  epv->f__dtor = impl_sidlx_rmi_ServerSocket__dtor;
  epv->f_init = impl_sidlx_rmi_ServerSocket_init;
  epv->f_accept = impl_sidlx_rmi_ServerSocket_accept;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_ServerSocket__call_load(void) { 
  impl_sidlx_rmi_ServerSocket__load();
}
struct sidlx_rmi_ServerSocket__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_ServerSocket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_ServerSocket(url, _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_ServerSocket(struct 
  sidlx_rmi_ServerSocket__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_ServerSocket(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidl_rmi_NetworkException(url,
    _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseInterface(obj);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_IPv4Socket(url, _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_IPv4Socket(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_ServerSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_ServerSocket__data*
sidlx_rmi_ServerSocket__get_data(sidlx_rmi_ServerSocket self)
{
  return (struct sidlx_rmi_ServerSocket__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_ServerSocket__set_data(
  sidlx_rmi_ServerSocket self,
  struct sidlx_rmi_ServerSocket__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
