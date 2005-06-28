/*
 * File:          sidlx_rmi_SimpleServer_Impl.c
 * Symbol:        sidlx.rmi.SimpleServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.rmi.SimpleServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimpleServer" (version 0.1)
 * 
 * A multi-threaded base class for simple network servers.
 */

#include "sidlx_rmi_SimpleServer_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._includes) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include "sidlx_common.h"
#define LISTENQ 1024
#define MAXLINE 1023
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__ctor(
  /* in */ sidlx_rmi_SimpleServer self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._ctor) */
  struct sidlx_rmi_SimpleServer__data *dptr;
  dptr = malloc(sizeof(struct sidlx_rmi_SimpleServer__data));
  dptr->s_sock = sidlx_rmi_ServerSocket__create();
  sidlx_rmi_SimpleServer__set_data(self, dptr);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer__dtor(
  /* in */ sidlx_rmi_SimpleServer self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer._dtor) */
  struct sidlx_rmi_SimpleServer__data * data = sidlx_rmi_SimpleServer__get_data( self );
  if (data) {
    sidlx_rmi_ServerSocket_deleteRef(data->s_sock);
    free((void*) data);
  }
  sidlx_rmi_SimpleServer__set_data( self, NULL );
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer._dtor) */
}

/*
 * set which port number to bind to
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_setPort"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer_setPort(
  /* in */ sidlx_rmi_SimpleServer self,
  /* in */ int32_t port,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.setPort) */
  struct sidlx_rmi_SimpleServer__data *data=sidlx_rmi_SimpleServer__get_data(self);
  
  sidlx_rmi_ServerSocket_init(data->s_sock, port, _ex); SIDL_CHECK(*_ex);
  return;
 EXIT:
  printf("Exception caught in impl_sidlx_rmi_SimpleServer_setPort\n\n");
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.setPort) */
}

/*
 * run the server (must have port specified first)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleServer_run"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleServer_run(
  /* in */ sidlx_rmi_SimpleServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleServer.run) */
  int i;
  int len;
  int connection_socket;
  int pid;
  char buff[MAXLINE];
  sidlx_rmi_Socket ac_sock = NULL;
  struct sidlx_rmi_SimpleServer__data *dptr=sidlx_rmi_SimpleServer__get_data(self);
  if(dptr) {
    /*
      data=sidlx_rmi_SimpleServer__get_data(self);
      
      data->d_listen_socket = 
      s_socket( AF_INET, SOCK_STREAM, 0, _ex); SIDL_CHECK(*_ex);
      
      s_bind( data->d_listen_socket, (struct sockaddr*) &(data->d_serv_addr), 
      sizeof (data->d_serv_addr), _ex); SIDL_CHECK(*_ex);
      
      s_listen( data->d_listen_socket, LISTENQ, _ex); SIDL_CHECK(*_ex);
    */
    /*for(;;) {*/ 
      /* len = sizeof(cliaddr); */
      ac_sock = sidlx_rmi_ServerSocket_accept(dptr->s_sock, _ex); SIDL_CHECK(*_ex);
      
      /*Basically, I need our ORB to be single process
      #define GARY_K
      #ifdef GARY_K
            pid = s_fork(_ex); SIDL_CHECK(*_ex);
      if ( pid == 0 ) {*/ 
      /* child closes listening socket 
      sidlx_rmi_ServerSocket_close( dptr->s_sock, _ex); SIDL_CHECK(*_ex);
      sidlx_rmi_Socket_deleteRef(dptr->s_sock);*/
      /* SHOULD clean up here */
      /*#endif*/
	/* process request */
      printf("SimpleServer: connection\n");
      /*
	from %s port %d\n",
	inet_ntop(AF_INET, &cliaddr.sin_addr, buff, sizeof(buff)),
	ntohs(cliaddr.sin_port ));
      */
      sidlx_rmi_SimpleServer_serviceRequest( self, ac_sock,_ex );SIDL_CHECK(*_ex);
      /*s_close(connection_socket, _ex); SIDL_CHECK(*_ex);
      exit(0);
      #ifdef GARY_K
      } 
      sidlx_rmi_Socket_close( ac_sock, _ex); SIDL_CHECK(*_ex);
      sidlx_rmi_Socket_deleteRef(ac_sock);
      s_close(connection_socket, _ex); SIDL_CHECK(*_ex); 
      #endif
      }*/
      return;
  }
  SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "Simple Server not initialized");

 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleServer.run) */
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_SIDLException__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleServer_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) {
  return sidl_SIDLException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleServer_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleServer__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleServer_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj) {
  return sidlx_rmi_SimpleServer__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleServer_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleServer_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
