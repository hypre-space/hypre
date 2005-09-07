/*
 * File:          sidlx_rmi_GenNetworkException_Impl.c
 * Symbol:        sidlx.rmi.GenNetworkException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.GenNetworkException
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
 * Symbol "sidlx.rmi.GenNetworkException" (version 0.1)
 * 
 * Generic Network Exception
 */

#include "sidlx_rmi_GenNetworkException_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_GenNetworkException_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._includes) */
/* insert implementation here: sidlx.rmi.GenNetworkException._includes (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._includes) */
#line 30 "sidlx_rmi_GenNetworkException_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_GenNetworkException__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_GenNetworkException__load(
  void)
{
#line 44 "../../../babel/runtime/sidlx/sidlx_rmi_GenNetworkException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._load) */
  /* insert implementation here: sidlx.rmi.GenNetworkException._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._load) */
#line 50 "sidlx_rmi_GenNetworkException_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_GenNetworkException__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_GenNetworkException__ctor(
  /* in */ sidlx_rmi_GenNetworkException self)
{
#line 62 "../../../babel/runtime/sidlx/sidlx_rmi_GenNetworkException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._ctor) */
  /* insert implementation here: sidlx.rmi.GenNetworkException._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._ctor) */
#line 70 "sidlx_rmi_GenNetworkException_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_GenNetworkException__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_GenNetworkException__dtor(
  /* in */ sidlx_rmi_GenNetworkException self)
{
#line 81 "../../../babel/runtime/sidlx/sidlx_rmi_GenNetworkException_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._dtor) */
  /* insert implementation here: sidlx.rmi.GenNetworkException._dtor (destructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._dtor) */
#line 91 "sidlx_rmi_GenNetworkException_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, sidl_BaseInterface *_ex) {
  return sidlx_rmi_GenNetworkException__connect(url, _ex);
}
char * 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj) {
  return sidlx_rmi_GenNetworkException__getURL(obj);
}
struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connect(url, _ex);
}
char * impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) {
  return sidl_SIDLException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_io_IOException__connect(url, _ex);
}
char * impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) {
  return sidl_io_IOException__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseException__connect(url, _ex);
}
char * impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) {
  return sidl_BaseException__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
