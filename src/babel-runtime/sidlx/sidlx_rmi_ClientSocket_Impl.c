/*
 * File:          sidlx_rmi_ClientSocket_Impl.c
 * Symbol:        sidlx.rmi.ClientSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.ClientSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
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
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._includes) */
#include "sidlx_rmi_Socket.h"
#include "sidlx_rmi_Common.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._load) */
  /* insert implementation here: sidlx.rmi.ClientSocket._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._load) */
  }
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
  /* in */ sidlx_rmi_ClientSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._ctor) */
  /* insert implementation here: sidlx.rmi.ClientSocket._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ClientSocket__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ClientSocket__ctor2(
  /* in */ sidlx_rmi_ClientSocket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.ClientSocket._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._ctor2) */
  }
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
  /* in */ sidlx_rmi_ClientSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket._dtor) */
  /* insert implementation here: sidlx.rmi.ClientSocket._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket._dtor) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ClientSocket.init) */
  int sockfd;
  int address;
  struct sidlx_rmi_ClientSocket__data client_data;
  sidl_BaseInterface _throwaway_exception = NULL;
  address = sidlx_rmi_Common_getHostIP(hostname, _ex); SIDL_CHECK(*_ex);

  client_data.d_serv_addr.sin_family = AF_INET;
  client_data.d_serv_addr.sin_addr.s_addr = htonl(address);
  client_data.d_serv_addr.sin_port = htons( port );
  client_data.addrlen = sizeof(struct sockaddr_in);

  if ( (sockfd = socket(AF_INET, SOCK_STREAM,0)) < 0 ) { 
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "socket() error");
  }

  if ( connect( sockfd, (struct sockaddr *) &(client_data.d_serv_addr), client_data.addrlen) < 0) {
    sidlx_rmi_ClientSocket_setFileDescriptor(self, sockfd, &_throwaway_exception); 
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "connect() error");
  }

  sidlx_rmi_ClientSocket_setFileDescriptor(self, sockfd, _ex); SIDL_CHECK(*_ex);
  return 0;
 EXIT:
  return -1;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ClientSocket.init) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_ClientSocket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_ClientSocket__connectI(url, ar, _ex);
}
struct sidlx_rmi_ClientSocket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_ClientSocket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_ClientSocket__cast(bi, _ex);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_IPv4Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_IPv4Socket__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ClientSocket_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
