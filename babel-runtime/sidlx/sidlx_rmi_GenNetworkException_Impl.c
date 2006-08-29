/*
 * File:          sidlx_rmi_GenNetworkException_Impl.c
 * Symbol:        sidlx.rmi.GenNetworkException-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.GenNetworkException
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
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
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._includes) */
/* Insert-Code-Here {sidlx.rmi.GenNetworkException._includes} (includes and arbitrary code) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._includes) */

#define SIDL_IOR_MAJOR_VERSION 0
#define SIDL_IOR_MINOR_VERSION 10
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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._load) */
    /* Insert-Code-Here {sidlx.rmi.GenNetworkException._load} (static class initializer method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._load) */
  }
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
  /* in */ sidlx_rmi_GenNetworkException self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._ctor) */
    /* Insert-Code-Here {sidlx.rmi.GenNetworkException._ctor} (constructor method) */
    /*
     * // boilerplate constructor
     * struct sidlx_rmi_GenNetworkException__data *dptr = (struct sidlx_rmi_GenNetworkException__data*)malloc(sizeof(struct sidlx_rmi_GenNetworkException__data));
     * if (dptr) {
     *   memset(dptr, 0, sizeof(struct sidlx_rmi_GenNetworkException__data));
     *   // initialize elements of dptr here
     * }
     * sidlx_rmi_GenNetworkException__set_data(self, dptr);
     */

    /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_GenNetworkException__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_GenNetworkException__ctor2(
  /* in */ sidlx_rmi_GenNetworkException self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._ctor2) */
    /* Insert-Code-Here {sidlx.rmi.GenNetworkException._ctor2} (special constructor method) */
    /*
     * This method has not been implemented
     */

    SIDL_THROW(*_ex, sidl_NotImplementedException,     "This method has not been implemented");
  EXIT:;
    /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._ctor2) */
  }
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
  /* in */ sidlx_rmi_GenNetworkException self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
    /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._dtor) */
    /* Insert-Code-Here {sidlx.rmi.GenNetworkException._dtor} (destructor method) */
    /*
     * // boilerplate destructor
     * struct sidlx_rmi_GenNetworkException__data *dptr = sidlx_rmi_GenNetworkException__get_data(self);
     * if (dptr) {
     *   // free contained in dtor before next line
     *   free(dptr);
     *   sidlx_rmi_GenNetworkException__set_data(self, NULL);
     * }
     */

    /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._dtor) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseException__connectI(url, ar, _ex);
}
struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseException__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connectI(url, ar, _ex);
}
struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_SIDLException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_SIDLException__cast(bi, _ex);
}
struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Deserializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Deserializer__connectI(url, ar, _ex);
}
struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Deserializer__cast(bi, _ex);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_IOException__connectI(url, ar, _ex);
}
struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_IOException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_IOException__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Serializable(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Serializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializer__connectI(url, ar, _ex);
}
struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializer__cast(bi, _ex);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connectI(url, ar, _ex);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_rmi_NetworkException(void* bi,
  sidl_BaseInterface* _ex) {
  return sidl_rmi_NetworkException__cast(bi, _ex);
}
struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_GenNetworkException__connectI(url, ar, _ex);
}
struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidlx_rmi_GenNetworkException(void* 
  bi, sidl_BaseInterface* _ex) {
  return sidlx_rmi_GenNetworkException__cast(bi, _ex);
}
