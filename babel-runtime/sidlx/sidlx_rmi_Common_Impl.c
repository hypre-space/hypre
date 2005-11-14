/*
 * File:          sidlx_rmi_Common_Impl.c
 * Symbol:        sidlx.rmi.Common-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.Common
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
 * Symbol "sidlx.rmi.Common" (version 0.1)
 * 
 * Some basic useful functions
 */

#include "sidlx_rmi_Common_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._includes) */

#include "sidlx_rmi_GenNetworkException.h"
#include "sidl_Exception.h"
#include <stdio.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "sidlx_rmi_Socket.h"
#include "sidl_String.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._includes) */
#line 44 "sidlx_rmi_Common_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Common__load(
  void)
{
#line 58 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._load) */
  /* insert implementation here: sidlx.rmi.Common._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._load) */
#line 64 "sidlx_rmi_Common_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Common__ctor(
  /* in */ sidlx_rmi_Common self)
{
#line 76 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._ctor) */
  /* insert implementation here: sidlx.rmi.Common._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._ctor) */
#line 84 "sidlx_rmi_Common_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Common__dtor(
  /* in */ sidlx_rmi_Common self)
{
#line 95 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._dtor) */
  /* insert implementation here: sidlx.rmi.Common._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._dtor) */
#line 105 "sidlx_rmi_Common_Impl.c"
}

/*
 * Method:  fork[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common_fork"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_Common_fork(
  /* out */ sidl_BaseInterface *_ex)
{
#line 114 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common.fork) */
  int32_t pid;
  if ((pid=fork()) < 0) { 
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "fork() error");
  }
 EXIT:
  return pid;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common.fork) */
#line 131 "sidlx_rmi_Common_Impl.c"
}

/*
 * Method:  gethostbyname[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common_gethostbyname"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_Common_gethostbyname(
  /* in */ const char* hostname,
  /* out */ sidl_BaseInterface *_ex)
{
#line 139 "../../../babel/runtime/sidlx/sidlx_rmi_Common_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common.gethostbyname) */
  /* TODO: Reimplement using gethostbyname */
  
  /* int32_t ret = 0;
  int32_t *temp = NULL;
  struct hostent *h = NULL;
  if ((h=gethostbyname2(hostname, AF_INET)) == NULL) { 
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "gethostbyname() error");
  }
  if (h->h_addrtype != AF_INET) {
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "gethostbyname() returned AF_INET6 addr!");
  }
  if (*(h->h_addr_list) == NULL) {
    SIDL_THROW( *_ex, sidlx_rmi_GenNetworkException, "gethostbyname() returned no data!");
  }
  temp = (int32_t*)(*(h->h_addr_list));
  ret = ntohl(*temp);
  return ret;*/
  /* free((void*)h); */
 EXIT:
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common.gethostbyname) */
#line 172 "sidlx_rmi_Common_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Common__connect(url, _ex);
}
char * impl_sidlx_rmi_Common_fgetURL_sidlx_rmi_Common(struct 
  sidlx_rmi_Common__object* obj) {
  return sidlx_rmi_Common__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_Common_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_Common_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_Common_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_Common_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
