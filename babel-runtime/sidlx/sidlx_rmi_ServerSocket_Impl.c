/*
 * File:          sidlx_rmi_ServerSocket_Impl.c
 * Symbol:        sidlx.rmi.ServerSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.ServerSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.ServerSocket" (version 0.1)
 * 
 * Automatically sets up a port for listening for new connections
 */

#include "sidlx_rmi_ServerSocket_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._includes) */
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include "sidlx_rmi_ChildSocket.h"
#define LISTENQ 1024
#define MAXLINE 1023
/* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._includes) */
#line 35 "sidlx_rmi_ServerSocket_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ServerSocket__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ServerSocket__load(
  void)
{
#line 49 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._load) */
  /* insert implementation here: sidlx.rmi.ServerSocket._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._load) */
#line 55 "sidlx_rmi_ServerSocket_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ServerSocket__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ServerSocket__ctor(
  /* in */ sidlx_rmi_ServerSocket self)
{
#line 67 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._ctor) */
  /* insert implementation here: sidlx.rmi.ServerSocket._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._ctor) */
#line 75 "sidlx_rmi_ServerSocket_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ServerSocket__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ServerSocket__dtor(
  /* in */ sidlx_rmi_ServerSocket self)
{
#line 86 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._dtor) */
  /* insert implementation here: sidlx.rmi.ServerSocket._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._dtor) */
#line 96 "sidlx_rmi_ServerSocket_Impl.c"
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ServerSocket_init"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_ServerSocket_init(
  /* in */ sidlx_rmi_ServerSocket self,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex)
{
#line 107 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket.init) */
  int n = -1;
  int addrlen;
  int tempfd;

  struct sidlx_rmi_ServerSocket__data *dptr;
  dptr = malloc(sizeof(struct sidlx_rmi_ServerSocket__data));
  dptr->d_serv_addr.sin_family = AF_INET;
  dptr->d_serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  dptr->d_serv_addr.sin_port = htons( port );
  dptr->addrlen = sizeof(struct sockaddr_in);
  
  if((tempfd = s_socket( AF_INET, SOCK_STREAM, 0, _ex)) < 0) {
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "socket() error");
  }

  if ((n=bind(tempfd, (struct sockaddr*) &(dptr->d_serv_addr), dptr->addrlen)) < 0 ) { 
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "bind() error");
  }

  if ((n=listen(tempfd,10))<0) { 
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "listen() error");
  }
  
  sidlx_rmi_ServerSocket_setFileDescriptor(self, tempfd, _ex);
  sidlx_rmi_ServerSocket__set_data(self, dptr);
 EXIT:
  return n;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket.init) */
#line 145 "sidlx_rmi_ServerSocket_Impl.c"
}

/*
 * Method:  accept[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ServerSocket_accept"

#ifdef __cplusplus
extern "C"
#endif
sidlx_rmi_Socket
impl_sidlx_rmi_ServerSocket_accept(
  /* in */ sidlx_rmi_ServerSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 153 "../../../babel/runtime/sidlx/sidlx_rmi_ServerSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket.accept) */
  sidlx_rmi_ChildSocket cSock = NULL;
  sidlx_rmi_Socket retSock = NULL;
  struct sockaddr_in cliaddr;
  int clilen = sizeof(struct sockaddr_in);
  int cfd;
  int n = -1;
  struct sidlx_rmi_ServerSocket__data *dptr = sidlx_rmi_ServerSocket__get_data( self );

  cfd = sidlx_rmi_ServerSocket_getFileDescriptor(self, _ex); SIDL_CHECK(*_ex);
  
  if (dptr) {
    if ((n=accept(cfd, (struct sockaddr*) &cliaddr, &clilen)) < 0 ) { 
      SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "accept() error");
    }
    cSock = sidlx_rmi_ChildSocket__create();
    sidlx_rmi_ChildSocket_init(cSock, n, _ex);
    retSock = sidlx_rmi_Socket__cast(cSock); 

    /* This is getting really lame.  Re-listen on socket*/
    if ((n=listen(cfd,10))<0) { 
      SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "listen() error");
    }
    

    return retSock;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "Server Socket has not been initialized!");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket.accept) */
#line 195 "sidlx_rmi_ServerSocket_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_rmi_ServerSocket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_ServerSocket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_ServerSocket__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_ServerSocket(struct 
  sidlx_rmi_ServerSocket__object* obj) {
  return sidlx_rmi_ServerSocket__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) {
  return sidlx_rmi_IPv4Socket__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_ServerSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
