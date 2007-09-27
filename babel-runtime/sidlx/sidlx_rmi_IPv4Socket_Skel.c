/*
 * File:          sidlx_rmi_IPv4Socket_Skel.c
 * Symbol:        sidlx.rmi.IPv4Socket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side glue code for sidlx.rmi.IPv4Socket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_IPv4Socket_IOR.h"
#include "sidlx_rmi_IPv4Socket.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_IPv4Socket__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_IPv4Socket__ctor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_IPv4Socket__ctor2(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_IPv4Socket__dtor(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
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

extern
sidl_bool
impl_sidlx_rmi_IPv4Socket_test(
  /* in */ sidlx_rmi_IPv4Socket self,
  /* in */ int32_t secs,
  /* in */ int32_t usecs,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_IPv4Socket__set_epv(struct sidlx_rmi_IPv4Socket__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_IPv4Socket__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_IPv4Socket__ctor2;
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
  epv->f_test = impl_sidlx_rmi_IPv4Socket_test;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_IPv4Socket__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_IPv4Socket__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* skel_sidlx_rmi_IPv4Socket_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_IPv4Socket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* skel_sidlx_rmi_IPv4Socket_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_IPv4Socket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_IPv4Socket(url, ar, _ex);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_IPv4Socket(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_Socket(void* bi, sidl_BaseInterface 
  *_ex) { 
  return impl_sidlx_rmi_IPv4Socket_fcast_sidlx_rmi_Socket(bi, _ex);
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
