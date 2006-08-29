/*
 * File:          sidlx_rmi_SimpleOrb_Skel.c
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for sidlx.rmi.SimpleOrb
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_SimpleOrb_IOR.h"
#include "sidlx_rmi_SimpleOrb.h"
#include <stddef.h>
#include "sidlx_rmi_SimpleOrb_IOR.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.h"
#endif

extern void sidlx_rmi_SimpleOrb__superEPV(
struct sidlx_rmi_SimpleServer__epv*);
/*
 * Hold pointer to IOR functions.
 */

static const struct sidlx_rmi_SimpleOrb__external *_externals = NULL;
/*
 * Lookup the symbol to get the IOR functions.
 */

static const struct sidlx_rmi_SimpleOrb__external* _loadIOR(void)
/*
 * Return pointer to internal IOR functions.
 */

{
#ifdef SIDL_STATIC_LIBRARY
  _externals = sidlx_rmi_SimpleOrb__externals();
#else
  _externals = (struct 
    sidlx_rmi_SimpleOrb__external*)sidl_dynamicLoadIOR("sidlx.rmi.SimpleOrb",
    "sidlx_rmi_SimpleOrb__externals") ;
  sidl_checkIORVersion("sidlx.rmi.SimpleOrb", _externals->d_ior_major_version,
    _externals->d_ior_minor_version, 0, 10);
#endif
  return _externals;
}

#define _getExternals() (_externals ? _externals : _loadIOR())


extern void sidlx_rmi_SimpleOrb__superEPV(
struct sidlx_rmi_SimpleServer__epv*);

extern
void
impl_sidlx_rmi_SimpleOrb__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleOrb__ctor(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleOrb__ctor2(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimpleOrb__dtor(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleOrb(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_SimpleOrb_serviceRequest(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimpleOrb_getServerURL(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimpleOrb_isLocalObject(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimpleOrb_getProtocol(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex);

extern
struct sidl_io_Serializable__array*
impl_sidlx_rmi_SimpleOrb_getExceptions(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleOrb(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimpleOrb__set_epv(struct sidlx_rmi_SimpleOrb__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimpleOrb__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_SimpleOrb__ctor2;
  epv->f__dtor = impl_sidlx_rmi_SimpleOrb__dtor;
  epv->f_serviceRequest = impl_sidlx_rmi_SimpleOrb_serviceRequest;
  epv->f_getServerURL = impl_sidlx_rmi_SimpleOrb_getServerURL;
  epv->f_isLocalObject = impl_sidlx_rmi_SimpleOrb_isLocalObject;
  epv->f_getProtocol = impl_sidlx_rmi_SimpleOrb_getProtocol;
  epv->f_getExceptions = impl_sidlx_rmi_SimpleOrb_getExceptions;

  sidlx_rmi_SimpleOrb__superEPV(_getExternals()->getSuperEPV());
}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimpleOrb__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_SimpleOrb__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_io_Serializable(url, ar, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_rmi_ServerInfo__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidl_rmi_ServerInfo(url, ar, _ex);
}

struct sidl_rmi_ServerInfo__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidl_rmi_ServerInfo(bi, _ex);
}

struct sidlx_rmi_SimpleOrb__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(url, ar, _ex);
}

struct sidlx_rmi_SimpleOrb__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleOrb(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleOrb(bi, _ex);
}

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(url, ar, _ex);
}

struct sidlx_rmi_SimpleServer__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleServer(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_Socket(bi, _ex);
}

struct sidlx_rmi_SimpleOrb__data*
sidlx_rmi_SimpleOrb__get_data(sidlx_rmi_SimpleOrb self)
{
  return (struct sidlx_rmi_SimpleOrb__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimpleOrb__set_data(
  sidlx_rmi_SimpleOrb self,
  struct sidlx_rmi_SimpleOrb__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
