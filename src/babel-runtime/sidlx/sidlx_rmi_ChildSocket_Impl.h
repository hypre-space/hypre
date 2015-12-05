/*
 * File:          sidlx_rmi_ChildSocket_Impl.h
 * Symbol:        sidlx.rmi.ChildSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.ChildSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidlx_rmi_ChildSocket_Impl_h
#define included_sidlx_rmi_ChildSocket_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
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
#ifndef included_sidlx_rmi_ChildSocket_h
#include "sidlx_rmi_ChildSocket.h"
#endif
#ifndef included_sidlx_rmi_IPv4Socket_h
#include "sidlx_rmi_IPv4Socket.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._includes) */
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "sidl_rmi_NetworkException.h"
#include "sidl_String.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._includes) */

/*
 * Private data for class sidlx.rmi.ChildSocket
 */

struct sidlx_rmi_ChildSocket__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._data) */
  /* insert implementation here: sidlx.rmi.ChildSocket._data (private data members) */
  int port;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_ChildSocket__data*
sidlx_rmi_ChildSocket__get_data(
  sidlx_rmi_ChildSocket);

extern void
sidlx_rmi_ChildSocket__set_data(
  sidlx_rmi_ChildSocket,
  struct sidlx_rmi_ChildSocket__data*);

extern
void
impl_sidlx_rmi_ChildSocket__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_ChildSocket__ctor(
  /* in */ sidlx_rmi_ChildSocket self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_ChildSocket__ctor2(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_ChildSocket__dtor(
  /* in */ sidlx_rmi_ChildSocket self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_ChildSocket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_ChildSocket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_ChildSocket_init(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t fileDes,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseClass(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_BaseClass(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_ClassInfo(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_ClassInfo(void* bi, sidl_BaseInterface* 
  _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_ChildSocket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_ChildSocket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_IPv4Socket(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_Socket(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_Socket(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
