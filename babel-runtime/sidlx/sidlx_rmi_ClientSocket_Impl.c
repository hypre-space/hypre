/*
 * File:          sidlx_rmi_ClientSocket_Impl.c
 * Symbol:        sidlx.rmi.ClientSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.ClientSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.ClientSocket" (version 0.1)
 * 
 * Automatically sets up a port for listening for new connections
 */

#include "sidlx_rmi_ClientSocket_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._includes) */
#include "sidlx_rmi_Socket.h"
#include "sidlx_rmi_Common.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._includes) */
#line 31 "sidlx_rmi_ClientSocket_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ClientSocket__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ClientSocket__load(
  void)
{
#line 45 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._load) */
  /* insert implementation here: sidlx.rmi.ClientSocket._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._load) */
#line 51 "sidlx_rmi_ClientSocket_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ClientSocket__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ClientSocket__ctor(
  /* in */ sidlx_rmi_ClientSocket self)
{
#line 63 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._ctor) */
  /* insert implementation here: sidlx.rmi.ClientSocket._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._ctor) */
#line 71 "sidlx_rmi_ClientSocket_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ClientSocket__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ClientSocket__dtor(
  /* in */ sidlx_rmi_ClientSocket self)
{
#line 82 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._dtor) */
  /* insert implementation here: sidlx.rmi.ClientSocket._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._dtor) */
#line 92 "sidlx_rmi_ClientSocket_Impl.c"
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ClientSocket_init"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_ClientSocket_init(
  /* in */ sidlx_rmi_ClientSocket self,
  /* in */ const char* hostname,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex)
{
#line 104 "../../../babel/runtime/sidlx/sidlx_rmi_ClientSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket.init) */
  int sockfd;
  int n;
  int port_number,address;
  int32_t data;
  struct sockaddr_in servaddr;
  struct sidl_char__array * sendline = NULL;
  struct sidl_char__array * recvline = NULL;
  sidlx_rmi_Socket osock = NULL;/* sidlx_rmi_IPv4Socket__create(); */
  struct sidlx_rmi_ClientSocket__data *dptr;

  address = sidlx_rmi_Common_gethostbyname(hostname, _ex); SIDL_CHECK(*_ex);

  dptr = malloc(sizeof(struct sidlx_rmi_ClientSocket__data));
  dptr->d_serv_addr.sin_family = AF_INET;
  dptr->d_serv_addr.sin_addr.s_addr = htonl(address);
  dptr->d_serv_addr.sin_port = htons( port );
  dptr->addrlen = sizeof(struct sockaddr_in);

  if ( (sockfd = socket(AF_INET, SOCK_STREAM,0)) < 0 ) { 
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "socket() error");
  }

  if ( connect( sockfd, (struct sockaddr *) &(dptr->d_serv_addr), dptr->addrlen) < 0) {
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "connect() error");
  }

  sidlx_rmi_ClientSocket_setFileDescriptor(self, sockfd, _ex); SIDL_CHECK(*_ex);
  return 0;
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket.init) */
#line 145 "sidlx_rmi_ClientSocket_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_ClientSocket__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_ClientSocket(struct 
  sidlx_rmi_ClientSocket__object* obj) {
  return sidlx_rmi_ClientSocket__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) {
  return sidlx_rmi_IPv4Socket__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_ClientSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
