/*
 * File:          sidlx_rmi_SimCall_Skel.c
 * Symbol:        sidlx.rmi.SimCall-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for sidlx.rmi.SimCall
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_SimCall_IOR.h"
#include "sidlx_rmi_SimCall.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimCall__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall__ctor(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall__ctor2(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall__dtor(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Call__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_Call(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_rmi_Call__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_rmi_Call(void* bi, sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fcast_sidlx_rmi_SimCall(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_SimCall_init(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimCall_getMethodName(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimCall_getObjectID(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
enum sidlx_rmi_CallType__enum
impl_sidlx_rmi_SimCall_getCallType(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackBool(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackChar(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackInt(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackLong(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackOpaque(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ void** value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackFloat(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackDouble(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackFcomplex(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackDcomplex(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackString(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackSerializable(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ sidl_io_Serializable* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackBoolArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<bool> */ struct sidl_bool__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackCharArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<char> */ struct sidl_char__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackIntArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<int> */ struct sidl_int__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackLongArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<long> */ struct sidl_long__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackOpaqueArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<opaque> */ struct sidl_opaque__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackFloatArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<float> */ struct sidl_float__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackDoubleArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<double> */ struct sidl_double__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackFcomplexArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackDcomplexArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackStringArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<string> */ struct sidl_string__array** value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackGenericArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<> */ struct sidl__array** value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackSerializableArray(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool isRarray,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Call__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_Call(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex);
extern struct sidl_rmi_Call__object* 
  impl_sidlx_rmi_SimCall_fcast_sidl_rmi_Call(void* bi, sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fcast_sidlx_rmi_SimCall(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimCall__set_epv(struct sidlx_rmi_SimCall__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimCall__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_SimCall__ctor2;
  epv->f__dtor = impl_sidlx_rmi_SimCall__dtor;
  epv->f_init = impl_sidlx_rmi_SimCall_init;
  epv->f_getMethodName = impl_sidlx_rmi_SimCall_getMethodName;
  epv->f_getObjectID = impl_sidlx_rmi_SimCall_getObjectID;
  epv->f_getCallType = impl_sidlx_rmi_SimCall_getCallType;
  epv->f_unpackBool = impl_sidlx_rmi_SimCall_unpackBool;
  epv->f_unpackChar = impl_sidlx_rmi_SimCall_unpackChar;
  epv->f_unpackInt = impl_sidlx_rmi_SimCall_unpackInt;
  epv->f_unpackLong = impl_sidlx_rmi_SimCall_unpackLong;
  epv->f_unpackOpaque = impl_sidlx_rmi_SimCall_unpackOpaque;
  epv->f_unpackFloat = impl_sidlx_rmi_SimCall_unpackFloat;
  epv->f_unpackDouble = impl_sidlx_rmi_SimCall_unpackDouble;
  epv->f_unpackFcomplex = impl_sidlx_rmi_SimCall_unpackFcomplex;
  epv->f_unpackDcomplex = impl_sidlx_rmi_SimCall_unpackDcomplex;
  epv->f_unpackString = impl_sidlx_rmi_SimCall_unpackString;
  epv->f_unpackSerializable = impl_sidlx_rmi_SimCall_unpackSerializable;
  epv->f_unpackBoolArray = impl_sidlx_rmi_SimCall_unpackBoolArray;
  epv->f_unpackCharArray = impl_sidlx_rmi_SimCall_unpackCharArray;
  epv->f_unpackIntArray = impl_sidlx_rmi_SimCall_unpackIntArray;
  epv->f_unpackLongArray = impl_sidlx_rmi_SimCall_unpackLongArray;
  epv->f_unpackOpaqueArray = impl_sidlx_rmi_SimCall_unpackOpaqueArray;
  epv->f_unpackFloatArray = impl_sidlx_rmi_SimCall_unpackFloatArray;
  epv->f_unpackDoubleArray = impl_sidlx_rmi_SimCall_unpackDoubleArray;
  epv->f_unpackFcomplexArray = impl_sidlx_rmi_SimCall_unpackFcomplexArray;
  epv->f_unpackDcomplexArray = impl_sidlx_rmi_SimCall_unpackDcomplexArray;
  epv->f_unpackStringArray = impl_sidlx_rmi_SimCall_unpackStringArray;
  epv->f_unpackGenericArray = impl_sidlx_rmi_SimCall_unpackGenericArray;
  epv->f_unpackSerializableArray = 
    impl_sidlx_rmi_SimCall_unpackSerializableArray;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimCall__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_SimCall__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimCall_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimCall_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimCall_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimCall_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(url, ar, _ex);
}

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_SimCall_fcast_sidl_io_Deserializer(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_io_Deserializer(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_io_Serializable(url, ar, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimCall_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_rmi_Call__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_rmi_Call(const char* url, sidl_bool ar,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_Call(url, ar, _ex);
}

struct sidl_rmi_Call__object* skel_sidlx_rmi_SimCall_fcast_sidl_rmi_Call(void* 
  bi, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidl_rmi_Call(bi, _ex);
}

struct sidlx_rmi_SimCall__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(url, ar, _ex);
}

struct sidlx_rmi_SimCall__object* 
  skel_sidlx_rmi_SimCall_fcast_sidlx_rmi_SimCall(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidlx_rmi_SimCall(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimCall_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fcast_sidlx_rmi_Socket(bi, _ex);
}

struct sidlx_rmi_SimCall__data*
sidlx_rmi_SimCall__get_data(sidlx_rmi_SimCall self)
{
  return (struct sidlx_rmi_SimCall__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimCall__set_data(
  sidlx_rmi_SimCall self,
  struct sidlx_rmi_SimCall__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
