/*
 * File:          sidlx_rmi_GenNetworkException_Impl.h
 * Symbol:        sidlx.rmi.GenNetworkException-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.GenNetworkException
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidlx_rmi_GenNetworkException_Impl_h
#define included_sidlx_rmi_GenNetworkException_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_RuntimeException_h
#include "sidl_RuntimeException.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifndef included_sidl_io_IOException_h
#include "sidl_io_IOException.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidlx_rmi_GenNetworkException_h
#include "sidlx_rmi_GenNetworkException.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._includes) */
/* Insert-Code-Here {sidlx.rmi.GenNetworkException._includes} (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._includes) */

/*
 * Private data for class sidlx.rmi.GenNetworkException
 */

struct sidlx_rmi_GenNetworkException__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.GenNetworkException._data) */
  /* Insert-Code-Here {sidlx.rmi.GenNetworkException._data} (private data members) */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.GenNetworkException._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_GenNetworkException__data*
sidlx_rmi_GenNetworkException__get_data(
  sidlx_rmi_GenNetworkException);

extern void
sidlx_rmi_GenNetworkException__set_data(
  sidlx_rmi_GenNetworkException,
  struct sidlx_rmi_GenNetworkException__data*);

extern
void
impl_sidlx_rmi_GenNetworkException__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_GenNetworkException__ctor(
  /* in */ sidlx_rmi_GenNetworkException self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_GenNetworkException__ctor2(
  /* in */ sidlx_rmi_GenNetworkException self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_GenNetworkException__dtor(
  /* in */ sidlx_rmi_GenNetworkException self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_SIDLException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Deserializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Deserializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_IOException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Serializable(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Serializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Serializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_rmi_NetworkException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidlx_rmi_GenNetworkException(void* 
  bi, sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_SIDLException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Deserializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Deserializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_IOException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Serializable(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_Serializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_io_Serializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidl_rmi_NetworkException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fcast_sidlx_rmi_GenNetworkException(void* 
  bi, sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
