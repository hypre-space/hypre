/*
 * File:          sidlx_rmi_JimEchoServer_Impl.c
 * Symbol:        sidlx.rmi.JimEchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.JimEchoServer
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
 * Symbol "sidlx.rmi.JimEchoServer" (version 0.1)
 * 
 * Echos the string back to the client using Jim's test protocol
 */

#include "sidlx_rmi_JimEchoServer_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_JimEchoServer_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._includes) */
#include "sidlx_common.h"
#include "sidlx_rmi_Socket.h"
#include "sidlx_rmi_ServerSocket.h"
#include "sidlx_rmi_ChildSocket.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._includes) */
#line 33 "sidlx_rmi_JimEchoServer_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_JimEchoServer__load(
  void)
{
#line 47 "../../../babel/runtime/sidlx/sidlx_rmi_JimEchoServer_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._load) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._load) */
#line 53 "sidlx_rmi_JimEchoServer_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_JimEchoServer__ctor(
  /* in */ sidlx_rmi_JimEchoServer self)
{
#line 65 "../../../babel/runtime/sidlx/sidlx_rmi_JimEchoServer_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._ctor) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._ctor) */
#line 73 "sidlx_rmi_JimEchoServer_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_JimEchoServer__dtor(
  /* in */ sidlx_rmi_JimEchoServer self)
{
#line 84 "../../../babel/runtime/sidlx/sidlx_rmi_JimEchoServer_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._dtor) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._dtor) */
#line 94 "sidlx_rmi_JimEchoServer_Impl.c"
}

/*
 * Method:  serviceRequest[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer_serviceRequest"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_JimEchoServer_serviceRequest(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
#line 105 "../../../babel/runtime/sidlx/sidlx_rmi_JimEchoServer_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer.serviceRequest) */
  anysockaddr un;
  socklen_t len;
  char buff[MAXLINE];
  struct sidl_char__array * data=NULL; 
  int i, n;
  int32_t num;
  sidl_BaseException s_b_e = NULL;
  char * str = NULL;
  /*  
   len=MAXSOCKADDR;
   s_getsockname(client_fd, (struct sockaddr  * ) un.data, &len, 
	 	_ex);
   if (*_ex) { printf("exception in s_getsockname\n"); }     SIDL_CHECK(*_ex);

   if (un.sa.sa_family == AF_INET ) { 
     struct sockaddr_in * cliaddr = (struct sockaddr_in *) &(un.sa);
     printf("EchoServer: connection to %s port %d\n",
 	   inet_ntop(AF_INET, &cliaddr->sin_addr, buff, sizeof(buff)),
 	   ntohs(cliaddr->sin_port )); 
   } else { 
     printf("connection using family %d\n", un.sa.sa_family );
   }
  */
  printf("about to start Maxline=%d\n",MAXLINE);
  for ( i=0; i>-1; i++ ) { 
    printf("%d. ready to read...\n", i);
    n = sidlx_rmi_Socket_readstring_alloc(sock, &data, _ex);
    /*n = s_read_string_alloc(client_fd, &data, _ex);*/
    if (*_ex) { printf("exception in s_readstring\n"); }     SIDL_CHECK(*_ex);
    if (n == 0) {
     break;
    }
    printf("got %d bytes\n", n );
 
    sidlx_rmi_Socket_writestring(sock, n,data, _ex);
    /*s_write_string(client_fd, n, data, _ex);*/
    sidl_char__array_deleteRef(data);
    data = NULL;
    if (*_ex) { printf("exception in s_writen\n"); }     SIDL_CHECK(*_ex);
  }
 EXIT:
  s_b_e = sidl_BaseException__cast(*_ex);
  str = sidl_BaseException_getNote(s_b_e);

  printf("Exiting %s \n", str);
  if (data) { 
    sidl_char__array_deleteRef(data);
    data=NULL;
  }
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer.serviceRequest) */
#line 167 "sidlx_rmi_JimEchoServer_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_SIDLException__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) {
  return sidl_SIDLException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_JimEchoServer(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_JimEchoServer__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidlx_rmi_JimEchoServer(struct 
  sidlx_rmi_JimEchoServer__object* obj) {
  return sidlx_rmi_JimEchoServer__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleServer__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj) {
  return sidlx_rmi_SimpleServer__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_JimEchoServer_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
