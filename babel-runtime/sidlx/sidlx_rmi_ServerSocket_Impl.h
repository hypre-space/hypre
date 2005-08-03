/*
 * File:          sidlx_rmi_ServerSocket_Impl.h
 * Symbol:        sidlx.rmi.ServerSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for sidlx.rmi.ServerSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_rmi_ServerSocket_Impl_h
#define included_sidlx_rmi_ServerSocket_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_rmi_ServerSocket_h
#include "sidlx_rmi_ServerSocket.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidlx_rmi_IPv4Socket_h
#include "sidlx_rmi_IPv4Socket.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 41 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._includes) */
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "sidlx_rmi_GenNetworkException.h"
#include "sidlx_rmi_Socket.h"
#include "sidl_String.h"
#include "sidl_Exception.h"
/* insert implementation here: sidlx.rmi.ServerSocket._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._includes) */
#line 58 "sidlx_rmi_ServerSocket_Impl.h"

/*
 * Private data for class sidlx.rmi.ServerSocket
 */

struct sidlx_rmi_ServerSocket__data {
#line 63 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._data) */
  /* insert implementation here: sidlx.rmi.ServerSocket._data (private data members) */
  int addrlen;
  struct sockaddr_in d_serv_addr;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._data) */
#line 71 "sidlx_rmi_ServerSocket_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_ServerSocket__data*
sidlx_rmi_ServerSocket__get_data(
  sidlx_rmi_ServerSocket);

extern void
sidlx_rmi_ServerSocket__set_data(
  sidlx_rmi_ServerSocket,
  struct sidlx_rmi_ServerSocket__data*);

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

/*
 * User-defined object methods
 */

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
}
#endif
#endif
