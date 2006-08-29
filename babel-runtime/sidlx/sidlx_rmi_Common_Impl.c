/*
 * File:          sidlx_rmi_Common_Impl.c
 * Symbol:        sidlx.rmi.Common-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.Common
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
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
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._includes) */

#include "sidl_rmi_NetworkException.h"
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

#ifdef HAVE_PTHREAD
#include <pthread.h>
/*lock for the hashtables */
static pthread_mutex_t host_mutex = PTHREAD_MUTEX_INITIALIZER; 
#endif /* HAVE_PTHREAD */

/* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._load) */
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_init(&(host_mutex), NULL);
#endif /* HAVE_PTHREAD */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._load) */
  }
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
  /* in */ sidlx_rmi_Common self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._ctor) */
  /* insert implementation here: sidlx.rmi.Common._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_Common__ctor2(
  /* in */ sidlx_rmi_Common self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.Common._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._ctor2) */
  }
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
  /* in */ sidlx_rmi_Common self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common._dtor) */
  /* insert implementation here: sidlx.rmi.Common._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common._dtor) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common.fork) */
  int32_t pid;
  if ((pid=fork()) < 0) { 
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "fork() error");
  }
 EXIT:
  return pid;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common.fork) */
  }
}

/*
 * Method:  getHostIP[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common_getHostIP"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_rmi_Common_getHostIP(
  /* in */ const char* hostname,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common.getHostIP) */
  int32_t ret = 0;
  int32_t *temp = NULL;
  struct hostent *h = NULL;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(host_mutex));
#endif /* HAVE_PTHREAD */
  if ((h=gethostbyname(hostname)) == NULL) { 
    char buf[512];
    snprintf(buf,512,"gethostbyname(\"%s\") failed",hostname);
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, buf);
  }
  if (h->h_addrtype != AF_INET) {
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "gethostbyname() returned AF_INET6 addr!");
  }
  if (*(h->h_addr_list) == NULL) {
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "gethostbyname() returned no data!");
  }
  temp = (int32_t*)(*(h->h_addr_list));
  ret = ntohl(*temp);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(host_mutex));
#endif /* HAVE_PTHREAD */
  return ret;
 EXIT:
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(host_mutex));
#endif /* HAVE_PTHREAD */
  return 0;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common.getHostIP) */
  }
}

/*
 * Method:  getCanonicalName[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_Common_getCanonicalName"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_Common_getCanonicalName(
  /* in */ const char* hostname,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Common.getCanonicalName) */
  char* ret = NULL;
  struct hostent *h = NULL;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(host_mutex));
#endif /* HAVE_PTHREAD */
  if ((h=gethostbyname(hostname)) == NULL) { 
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "gethostbyname() error");
  }
  if (*(h->h_addr_list) == NULL) {
    SIDL_THROW( *_ex, sidl_rmi_NetworkException, "gethostbyname() returned no data!");
  }
  ret = sidl_String_strdup(h->h_name);
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(host_mutex));
#endif /* HAVE_PTHREAD */
  return ret;
 EXIT:
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(host_mutex));
#endif /* HAVE_PTHREAD */
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Common.getCanonicalName) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidlx_rmi_Common_fcast_sidl_BaseClass(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Common_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_Common_fcast_sidl_ClassInfo(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Common_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Common_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fconnect_sidlx_rmi_Common(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Common__connectI(url, ar, _ex);
}
struct sidlx_rmi_Common__object* 
  impl_sidlx_rmi_Common_fcast_sidlx_rmi_Common(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Common__cast(bi, _ex);
}
