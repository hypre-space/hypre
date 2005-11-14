/*
 * File:          sidlx_rmi_ChildSocket_Impl.c
 * Symbol:        sidlx.rmi.ChildSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.ChildSocket
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
 * Symbol "sidlx.rmi.ChildSocket" (version 0.1)
 * 
 * Simple socket passed back by accept
 */

#include "sidlx_rmi_ChildSocket_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_ChildSocket_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._includes) */
/* insert implementation here: sidlx.rmi.ChildSocket._includes (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._includes) */
#line 30 "sidlx_rmi_ChildSocket_Impl.c"

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
  void)
{
#line 44 "../../../babel/runtime/sidlx/sidlx_rmi_ChildSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._load) */
  /* insert implementation here: sidlx.rmi.ChildSocket._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._load) */
#line 50 "sidlx_rmi_ChildSocket_Impl.c"
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
  /* in */ sidlx_rmi_ChildSocket self)
{
#line 62 "../../../babel/runtime/sidlx/sidlx_rmi_ChildSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._ctor) */
  /* insert implementation here: sidlx.rmi.ChildSocket._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._ctor) */
#line 70 "sidlx_rmi_ChildSocket_Impl.c"
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
  /* in */ sidlx_rmi_ChildSocket self)
{
#line 81 "../../../babel/runtime/sidlx/sidlx_rmi_ChildSocket_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.ChildSocket._dtor) */
  /* insert implementation here: sidlx.rmi.ChildSocket._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.ChildSocket._dtor) */
#line 91 "sidlx_rmi_ChildSocket_Impl.c"
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
#line 102 "../../../babel/runtime/sidlx/sidlx_rmi_ChildSocket_Impl.c"
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
#line 123 "sidlx_rmi_ChildSocket_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidlx_rmi_ChildSocket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_ChildSocket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_ChildSocket__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidlx_rmi_ChildSocket(struct 
  sidlx_rmi_ChildSocket__object* obj) {
  return sidlx_rmi_ChildSocket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_rmi_IPv4Socket__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidlx_rmi_IPv4Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_IPv4Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidlx_rmi_IPv4Socket(struct 
  sidlx_rmi_IPv4Socket__object* obj) {
  return sidlx_rmi_IPv4Socket__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_ChildSocket_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_ChildSocket_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
