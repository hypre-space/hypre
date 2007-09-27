/*
 * File:          sidlx_rmi_JimEchoServer_Impl.h
 * Symbol:        sidlx.rmi.JimEchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.JimEchoServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidlx_rmi_JimEchoServer_Impl_h
#define included_sidlx_rmi_JimEchoServer_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_JimEchoServer_IOR_h
#include "sidlx_rmi_JimEchoServer_IOR.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_rmi_ServerInfo_h
#include "sidl_rmi_ServerInfo.h"
#endif
#ifndef included_sidlx_rmi_JimEchoServer_h
#include "sidlx_rmi_JimEchoServer.h"
#endif
#ifndef included_sidlx_rmi_SimpleServer_h
#include "sidlx_rmi_SimpleServer.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._includes) */
/* insert implementation here: sidlx.rmi.JimEchoServer._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._includes) */

/*
 * Private data for class sidlx.rmi.JimEchoServer
 */

struct sidlx_rmi_JimEchoServer__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._data) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._data (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_JimEchoServer__data*
sidlx_rmi_JimEchoServer__get_data(
  sidlx_rmi_JimEchoServer);

extern void
sidlx_rmi_JimEchoServer__set_data(
  sidlx_rmi_JimEchoServer,
  struct sidlx_rmi_JimEchoServer__data*);

extern void sidlx_rmi_JimEchoServer__superEPV(
struct sidlx_rmi_SimpleServer__epv*);

extern
void
impl_sidlx_rmi_JimEchoServer__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_JimEchoServer__ctor(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_JimEchoServer__ctor2(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_JimEchoServer__dtor(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_io_Serializable(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_rmi_ServerInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_rmi_ServerInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_JimEchoServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_JimEchoServer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_SimpleServer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_SimpleServer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_JimEchoServer_serviceRequest(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_JimEchoServer_getServerURL(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_JimEchoServer_isLocalObject(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern
struct sidl_io_Serializable__array*
impl_sidlx_rmi_JimEchoServer_getExceptions(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_io_Serializable(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_rmi_ServerInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_rmi_ServerInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_JimEchoServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_JimEchoServer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_SimpleServer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_SimpleServer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
