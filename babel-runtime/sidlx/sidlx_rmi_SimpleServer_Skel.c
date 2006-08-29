/*
 * File:          sidlx_rmi_SimpleServer_Skel.c
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for sidlx.rmi.SimpleServer
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_SimpleServer_IOR.h"
#include "sidlx_rmi_SimpleServer.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimpleServer__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleServer__ctor(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleServer__ctor2(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleServer__dtor(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_SimpleServer_setMaxThreadPool(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t max,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimpleServer_requestPort(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimpleServer_requestPortInRange(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t minport,
  /* in */ int32_t maxport,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_rmi_SimpleServer_getPort(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimpleServer_getServerName(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimpleServer_getServerURL(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex);

extern
int64_t
impl_sidlx_rmi_SimpleServer_run(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleServer_shutdown(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimpleServer__set_epv(struct sidlx_rmi_SimpleServer__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimpleServer__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_SimpleServer__ctor2;
  epv->f__dtor = impl_sidlx_rmi_SimpleServer__dtor;
  epv->f_setMaxThreadPool = impl_sidlx_rmi_SimpleServer_setMaxThreadPool;
  epv->f_requestPort = impl_sidlx_rmi_SimpleServer_requestPort;
  epv->f_requestPortInRange = impl_sidlx_rmi_SimpleServer_requestPortInRange;
  epv->f_getPort = impl_sidlx_rmi_SimpleServer_getPort;
  epv->f_getServerName = impl_sidlx_rmi_SimpleServer_getServerName;
  epv->f_getServerURL = impl_sidlx_rmi_SimpleServer_getServerURL;
  epv->f_run = impl_sidlx_rmi_SimpleServer_run;
  epv->f_shutdown = impl_sidlx_rmi_SimpleServer_shutdown;
  epv->f_serviceRequest = NULL;
  epv->f_getExceptions = NULL;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimpleServer__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_SimpleServer__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_io_Serializable(url, ar,
    _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_rmi_ServerInfo__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidl_rmi_ServerInfo(url, ar, _ex);
}

struct sidl_rmi_ServerInfo__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidl_rmi_ServerInfo(bi, _ex);
}

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(url, ar,
    _ex);
}

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_SimpleServer(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleServer_fcast_sidlx_rmi_Socket(bi, _ex);
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
