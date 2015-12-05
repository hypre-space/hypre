/*
 * File:          sidlx_rmi_JimEchoServer_Impl.c
 * Symbol:        sidlx.rmi.JimEchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.JimEchoServer
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.JimEchoServer" (version 0.1)
 * 
 *  
 * Echos strings back to the client using Jim's echo protocol
 */

#include "sidlx_rmi_JimEchoServer_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._includes) */
#include "sidl_String.h"
#include "sidlx_common.h"
#include "sidlx_rmi_Socket.h"
#include "sidlx_rmi_ServerSocket.h"
#include "sidlx_rmi_ChildSocket.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
static const struct sidlx_rmi_SimpleServer__epv* superEPV = NULL;

void sidlx_rmi_JimEchoServer__superEPV(
struct sidlx_rmi_SimpleServer__epv* parentEPV){
  superEPV = parentEPV;
}
/*
 * Get the full URL for exporting objects (Doesn't do anything in this server)
 */

static char*
super_getServerURL(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*superEPV->f_getServerURL)((struct sidlx_rmi_SimpleServer__object*)
    self,
    objID,
    _ex);
}

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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._load) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._load) */
  }
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
  /* in */ sidlx_rmi_JimEchoServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._ctor) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_JimEchoServer__ctor2(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.JimEchoServer._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._ctor2) */
  }
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
  /* in */ sidlx_rmi_JimEchoServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer._dtor) */
  /* insert implementation here: sidlx.rmi.JimEchoServer._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer._dtor) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer.serviceRequest) */
  struct sidl_char__array * data=NULL; 
  int i, n;
  for (;;) { 
    i = sidlx_rmi_Socket_getFileDescriptor(sock, _ex); SIDL_CHECK(*_ex);
    n = sidlx_rmi_Socket_readstring_alloc(sock, &data, _ex);   SIDL_CHECK(*_ex);
    if (n == 0) {
     break;
    }
    sidlx_rmi_Socket_writestring(sock, n,data, _ex); SIDL_CHECK(*_ex);
    sidl_char__array_deleteRef(data);
  }
 EXIT:
  if (data) { 
    sidl_char__array_deleteRef(data);
    data=NULL;
  }
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer.serviceRequest) */
  }
}

/*
 * Get the full URL for exporting objects (Doesn't do anything in this server)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer_getServerURL"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_JimEchoServer_getServerURL(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer.getServerURL) */
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer.getServerURL) */
  }
}

/*
 * Method:  isLocalObject[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer_isLocalObject"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_JimEchoServer_isLocalObject(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer.isLocalObject) */
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer.isLocalObject) */
  }
}

/*
 * This gets an array of logged exceptions.  If an exception can
 * not be thrown back to the caller, we log it with the Server.  This 
 * gets the array of all those exceptions.
 * THIS IS SOMETHING OF A TEST! THIS MAY CHANGE!
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_JimEchoServer_getExceptions"

#ifdef __cplusplus
extern "C"
#endif
struct sidl_io_Serializable__array*
impl_sidlx_rmi_JimEchoServer_getExceptions(
  /* in */ sidlx_rmi_JimEchoServer self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.JimEchoServer.getExceptions) */
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.JimEchoServer.getExceptions) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidl_rmi_ServerInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_ServerInfo__connectI(url, ar, _ex);
}
struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidl_rmi_ServerInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_ServerInfo__cast(bi, _ex);
}
struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_JimEchoServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_JimEchoServer__connectI(url, ar, _ex);
}
struct sidlx_rmi_JimEchoServer__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_JimEchoServer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_JimEchoServer__cast(bi, _ex);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_SimpleServer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleServer__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_SimpleServer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimpleServer__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_JimEchoServer_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
