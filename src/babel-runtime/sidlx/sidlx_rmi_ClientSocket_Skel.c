/*
 * File:          sidlx_rmi_ClientSocket_Skel.c
 * Symbol:        sidlx.rmi.ClientSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side glue code for sidlx.rmi.ClientSocket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_ClientSocket_IOR.h"
#include "sidlx_rmi_ClientSocket.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_ClientSocket__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_ClientSocket__ctor(
  /* in */ sidlx_rmi_ClientSocket self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_ClientSocket__ctor2(
  /* in */ sidlx_rmi_ClientSocket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_ClientSocket__dtor(
  /* in */ sidlx_rmi_ClientSocket self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_ClientSocket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern
int32_t
impl_sidlx_rmi_ClientSocket_init(
  /* in */ sidlx_rmi_ClientSocket self,
  /* in */ const char* hostname,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_ClientSocket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_ClientSocket__set_epv(struct sidlx_rmi_ClientSocket__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_ClientSocket__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_ClientSocket__ctor2;
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
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_ClientSocket__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* skel_sidlx_rmi_ClientSocket_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_ClientSocket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* skel_sidlx_rmi_ClientSocket_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidl_RuntimeException(url, ar, 
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_ClientSocket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidlx_rmi_ClientSocket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(url, ar, 
    _ex);
}

struct sidlx_rmi_ClientSocket__object* 
  skel_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_ClientSocket(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_ClientSocket(bi, _ex);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(url, ar, 
    _ex);
}

struct sidlx_rmi_IPv4Socket__object* 
  skel_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_IPv4Socket(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_Socket(bi, _ex);
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
