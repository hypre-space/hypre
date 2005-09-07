/*
 * File:          sidlx_rmi_IPv4Socket_Skel.c
 * Symbol:        sidlx.rmi.IPv4Socket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.rmi.IPv4Socket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_rmi_IPv4Socket_IOR.h"
#include "sidlx_rmi_IPv4Socket.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_IPv4Socket__load(
  void);

extern
void
impl_sidlx_rmi_IPv4Socket__ctor(
  /* in */ sidlx_rmi_IPv4Socket self);

extern
void
impl_sidlx_rmi_IPv4Socket__dtor(
  /* in */ sidlx_rmi_IPv4Socket self);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
int32_t
impl_sidlx_rmi_IPv4Socket_getsockname(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_getpeername(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* address,
  /* inout */ int32_t* port,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_close(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readn(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readline(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readstring(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readstring_alloc(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout array<char> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_readint(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* inout */ int32_t* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_writen(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_writestring(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t nbytes,
  /* in array<char> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_writeint(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_IPv4Socket_setFileDescriptor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t fd,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_IPv4Socket_getFileDescriptor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_IPv4Socket__set_epv(struct sidlx_rmi_IPv4Socket__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_IPv4Socket__ctor;
  epv->f__dtor = impl_sidlx_rmi_IPv4Socket__dtor;
  epv->f_getsockname = impl_sidlx_rmi_IPv4Socket_getsockname;
  epv->f_getpeername = impl_sidlx_rmi_IPv4Socket_getpeername;
  epv->f_close = impl_sidlx_rmi_IPv4Socket_close;
  epv->f_readn = impl_sidlx_rmi_IPv4Socket_readn;
  epv->f_readline = impl_sidlx_rmi_IPv4Socket_readline;
  epv->f_readstring = impl_sidlx_rmi_IPv4Socket_readstring;
  epv->f_readstring_alloc = impl_sidlx_rmi_IPv4Socket_readstring_alloc;
  epv->f_readint = impl_sidlx_rmi_IPv4Socket_readint;
  epv->f_writen = impl_sidlx_rmi_IPv4Socket_writen;
  epv->f_writestring = impl_sidlx_rmi_IPv4Socket_writestring;
  epv->f_writeint = impl_sidlx_rmi_IPv4Socket_writeint;
  epv->f_setFileDescriptor = impl_sidlx_rmi_IPv4Socket_setFileDescriptor;
  epv->f_getFileDescriptor = impl_sidlx_rmi_IPv4Socket_getFileDescriptor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_IPv4Socket__call_load(void) { 
  impl_sidlx_rmi_IPv4Socket__load();
}
struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_rmi_NetworkException(url, _ex);
}

char* skel_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseInterface(obj);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(url, _ex);
}

char* skel_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) { 
  return impl_sidlx_rmi_IPv4Socket_fgetURL_sidlx_rmi_IPv4Socket(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_IPv4Socket_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_IPv4Socket__data*
sidlx_rmi_IPv4Socket__get_data(sidlx_rmi_IPv4Socket self)
{
  return (struct sidlx_rmi_IPv4Socket__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_IPv4Socket__set_data(
  sidlx_rmi_IPv4Socket self,
  struct sidlx_rmi_IPv4Socket__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
