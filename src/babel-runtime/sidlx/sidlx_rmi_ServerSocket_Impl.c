/*
 * File:          sidlx_rmi_ServerSocket_Impl.c
 * Symbol:        sidlx.rmi.ServerSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.ServerSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
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
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._includes) */
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include "sidlx_rmi_ChildSocket.h"
#include "sidlx_common.h"
#define LISTENQ 1024
#define MAXLINE 1023
#include <sys/types.h>
#include <sys/socket.h>

/* I'd rather use strerror_r (threadsafe version) but it
   appears to be broken on ingot -- gkk */
#define SIDL_THROW_PERROR( ex, type, msg ) { \
   char stp_buf[1024]; \
   int my_err = errno; \
   const int msg_len = strlen(msg); \
   strncpy(stp_buf,msg,sizeof(stp_buf)); \
   strncpy(stp_buf+msg_len, strerror(my_err),  sizeof(stp_buf)-msg_len-1); \
   SIDL_THROW(ex, type, stp_buf); \
 }

/* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._load) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._load) */
  }
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
  /* in */ sidlx_rmi_ServerSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._ctor) */
  struct sidlx_rmi_ServerSocket__data *dptr = 
    (struct sidlx_rmi_ServerSocket__data *)
    malloc(sizeof(struct sidlx_rmi_ServerSocket__data));
  sidlx_rmi_ServerSocket__set_data(self,dptr);
  /* initialize entire struct to zeros */
  memset( dptr, 0, sizeof(struct sidlx_rmi_ServerSocket__data));
  dptr->d_serv_addr.sin_family = AF_INET;
  dptr->d_serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  /* nonsensical port number for now */
  dptr->d_serv_addr.sin_port = htons(0); 
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ServerSocket__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ServerSocket__ctor2(
  /* in */ sidlx_rmi_ServerSocket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.ServerSocket._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._ctor2) */
  }
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
  /* in */ sidlx_rmi_ServerSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket._dtor) */
  struct sidlx_rmi_ServerSocket__data *dptr = 
    sidlx_rmi_ServerSocket__get_data(self);
  sidlx_rmi_ServerSocket__set_data(self,NULL);
  if (dptr) { 
    if ( ntohs( dptr->d_serv_addr.sin_port ) != 0 ) { 
      sidlx_rmi_ServerSocket_close(self,_ex); SIDL_CLEAR(*_ex);
      dptr->d_serv_addr.sin_port = htons( 0 );
    }
  }
  free((void*) dptr);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket._dtor) */
  }
}

/*
 *  if successful, returns 0 
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket.init) */
  int n = -1;
  int fd;

  struct sidlx_rmi_ServerSocket__data *dptr =  
    sidlx_rmi_ServerSocket__get_data(self);
  sidlx_rmi_ServerSocket__set_data(self, dptr);

  /* test if the port is assigned */
  if ( ntohs( dptr->d_serv_addr.sin_port ) != 0 ) { 
    SIDL_THROW_PERROR( *_ex, sidl_rmi_NetworkException, "cannot init() an active sidlx.rmi.ServerSocket");
  } 

  /* set the requested port */
  dptr->d_serv_addr.sin_port = htons( port );
  
  if((fd = s_socket( AF_INET, SOCK_STREAM, 0, _ex)) < 0) {
    SIDL_THROW_PERROR( *_ex, sidl_rmi_NetworkException, "socket() error: ");
  }
  sidlx_rmi_ServerSocket_setFileDescriptor(self, fd, _ex);

  if ((n=bind(fd, (struct sockaddr*) &(dptr->d_serv_addr), 
	      sizeof(struct sockaddr_in))) < 0 ) { 
    /* reset port number on failure */
    dptr->d_serv_addr.sin_port = htons( 0 );
    sidlx_rmi_ServerSocket_close(self, _ex); SIDL_CLEAR(*_ex);
    SIDL_THROW_PERROR( *_ex, sidl_rmi_NetworkException, "bind() error: ");
  }

  if ((n=listen(fd,10))<0) { 
    /* reset port number on failure */
    dptr->d_serv_addr.sin_port = htons( 0 );
    sidlx_rmi_ServerSocket_close(self, _ex); SIDL_CLEAR(*_ex);
    SIDL_THROW_PERROR( *_ex, sidl_rmi_NetworkException, "listen() error: ");
  }
  
 EXIT:
  return n;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket.init) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ServerSocket.accept) */
  sidlx_rmi_ChildSocket cSock = NULL;
  sidlx_rmi_Socket retSock = NULL;
  struct sockaddr_in cliaddr;
  socklen_t clilen = sizeof(struct sockaddr_in);
  int cfd;
  int n = -1;
  struct sidlx_rmi_ServerSocket__data *dptr = sidlx_rmi_ServerSocket__get_data( self );

  cfd = sidlx_rmi_ServerSocket_getFileDescriptor(self, _ex); SIDL_CHECK(*_ex);
  
  if (dptr) {
    if ((n=accept(cfd, (struct sockaddr*) &cliaddr, &clilen)) < 0 ) { 
      SIDL_THROW_PERROR( *_ex, sidl_rmi_NetworkException, "accept() error: ");
    }
    cSock = sidlx_rmi_ChildSocket__create(_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ChildSocket_init(cSock, n, _ex); SIDL_CHECK(*_ex);
    retSock = sidlx_rmi_Socket__cast(cSock,_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ChildSocket_deleteRef(cSock,_ex); SIDL_CHECK(*_ex);

    /* Re-listen on socket*/
    if ((n=listen(cfd,10))<0) { 
      SIDL_THROW_PERROR( *_ex, sidl_rmi_NetworkException, "listen() error: ");
    }
    return retSock;
  }
  SIDL_THROW( *_ex, sidl_rmi_NetworkException, "Server Socket has not been initialized!");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ServerSocket.accept) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidlx_rmi_IPv4Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_IPv4Socket__cast(bi, _ex);
}
struct sidlx_rmi_ServerSocket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_ServerSocket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_ServerSocket__connectI(url, ar, _ex);
}
struct sidlx_rmi_ServerSocket__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidlx_rmi_ServerSocket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_ServerSocket__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ServerSocket_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ServerSocket_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
