/*
 * File:          sidlx_rmi_Simsponse_Impl.h
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side implementation for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

#ifndef included_sidlx_rmi_Simsponse_Impl_h
#define included_sidlx_rmi_Simsponse_Impl_h

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
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidlx_rmi_Simsponse_h
#include "sidlx_rmi_Simsponse.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._includes) */
/* insert implementation here: sidlx.rmi.Simsponse._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._includes) */

/*
 * Private data for class sidlx.rmi.Simsponse
 */

struct sidlx_rmi_Simsponse__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._data) */
  /* insert implementation here: sidlx.rmi.Simsponse._data (private data members) */
  /*  int d_len; */
  /* char *d_buf; */
  struct sidl_char__array *d_carray;
  sidlx_rmi_Socket d_sock;
  char *d_methodName;
  char *d_className;
  char *d_objectID;
  int d_current;
  sidl_BaseException d_exception;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_Simsponse__data*
sidlx_rmi_Simsponse__get_data(
  sidlx_rmi_Simsponse);

extern void
sidlx_rmi_Simsponse__set_data(
  sidlx_rmi_Simsponse,
  struct sidlx_rmi_Simsponse__data*);

extern
void
impl_sidlx_rmi_Simsponse__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse__ctor(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse__ctor2(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse__dtor(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

/*
 * User-defined object methods
 */

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidlx_rmi_Simsponse(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_Simsponse_init(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* methodName,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_Simsponse_test(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ int32_t secs,
  /* in */ int32_t usecs,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_pullData(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getMethodName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getObjectID(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseException
impl_sidlx_rmi_Simsponse_getExceptionThrown(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackBool(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackChar(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackInt(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackLong(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackOpaque(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ void** value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackFloat(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackDouble(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackFcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackDcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackString(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackSerializable(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_io_Serializable* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackBoolArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<bool> */ struct sidl_bool__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackCharArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<char> */ struct sidl_char__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackIntArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<int> */ struct sidl_int__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackLongArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<long> */ struct sidl_long__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackOpaqueArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<opaque> */ struct sidl_opaque__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackFloatArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<float> */ struct sidl_float__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackDoubleArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<double> */ struct sidl_double__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackFcomplexArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackDcomplexArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackStringArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<string> */ struct sidl_string__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackGenericArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<> */ struct sidl__array** value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackSerializableArray(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidlx_rmi_Simsponse(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
}
#endif
#endif
