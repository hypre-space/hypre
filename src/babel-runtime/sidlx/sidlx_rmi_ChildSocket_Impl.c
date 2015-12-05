/*
 * File:          sidlx_rmi_ChildSocket_Impl.c
 * Symbol:        sidlx.rmi.ChildSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.ChildSocket
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.ChildSocket" (version 0.1)
 * 
 * Simple socket passed back by accept
 */

#include "sidlx_rmi_ChildSocket_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._includes) */
/* insert implementation here: sidlx.rmi.ChildSocket._includes (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ChildSocket__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ChildSocket__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._load) */
  /* insert implementation here: sidlx.rmi.ChildSocket._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ChildSocket__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ChildSocket__ctor(
  /* in */ sidlx_rmi_ChildSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._ctor) */
  /* insert implementation here: sidlx.rmi.ChildSocket._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ChildSocket__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ChildSocket__ctor2(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.ChildSocket._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ChildSocket__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ChildSocket__dtor(
  /* in */ sidlx_rmi_ChildSocket self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._dtor) */
  /* insert implementation here: sidlx.rmi.ChildSocket._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._dtor) */
  }
}

/*
 * Method:  init[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_ChildSocket_init"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_ChildSocket_init(
  /* in */ sidlx_rmi_ChildSocket self,
  /* in */ int32_t fileDes,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket.init) */
  /*  struct sidlx_rmi_ChildSocket__data *dptr =
     sidlx_rmi_ChildSocket__get_data(self);
   if (dptr) {
     dptr->port = port;
   } else {
     dptr = malloc(sizeof(struct sidlx_rmi_ChildSocket__data));
     dptr->port = port;
     }
    sidlx_rmi_ChildSocket__set_data(self, dptr);*/
  sidlx_rmi_ChildSocket_setFileDescriptor(self, fileDes, _ex);
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket.init) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_ChildSocket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_ChildSocket__connectI(url, ar, _ex);
}
struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_ChildSocket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_ChildSocket__cast(bi, _ex);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_IPv4Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_IPv4Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_IPv4Socket__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
