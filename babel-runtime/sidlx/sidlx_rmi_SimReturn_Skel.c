/*
 * File:          sidlx_rmi_SimReturn_Skel.c
 * Symbol:        sidlx.rmi.SimReturn-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for sidlx.rmi.SimReturn
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_SimReturn_IOR.h"
#include "sidlx_rmi_SimReturn.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimReturn__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn__ctor(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn__ctor2(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn__dtor(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Return__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_Return(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Return__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_rmi_Return(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_SimReturn(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_SimReturn_init(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* methodName,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimReturn_getMethodName(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_SendReturn(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_throwException(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ sidl_BaseException ex_to_throw,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packBool(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packChar(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packInt(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packLong(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packOpaque(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ void* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packFloat(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packDouble(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packFcomplex(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packDcomplex(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packString(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packSerializable(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ sidl_io_Serializable value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packBoolArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<bool> */ struct sidl_bool__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packCharArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<char> */ struct sidl_char__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packIntArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<int> */ struct sidl_int__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packLongArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<long> */ struct sidl_long__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packOpaqueArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<opaque> */ struct sidl_opaque__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packFloatArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<float> */ struct sidl_float__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packDoubleArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<double> */ struct sidl_double__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packFcomplexArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<fcomplex> */ struct sidl_fcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packDcomplexArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<dcomplex> */ struct sidl_dcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packStringArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<string> */ struct sidl_string__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packGenericArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<> */ struct sidl__array* value,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packSerializableArray(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in array<sidl.io.Serializable> */ struct sidl_io_Serializable__array* 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Return__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_Return(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Return__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidl_rmi_Return(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_SimReturn(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimReturn__set_epv(struct sidlx_rmi_SimReturn__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimReturn__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_SimReturn__ctor2;
  epv->f__dtor = impl_sidlx_rmi_SimReturn__dtor;
  epv->f_init = impl_sidlx_rmi_SimReturn_init;
  epv->f_getMethodName = impl_sidlx_rmi_SimReturn_getMethodName;
  epv->f_SendReturn = impl_sidlx_rmi_SimReturn_SendReturn;
  epv->f_throwException = impl_sidlx_rmi_SimReturn_throwException;
  epv->f_packBool = impl_sidlx_rmi_SimReturn_packBool;
  epv->f_packChar = impl_sidlx_rmi_SimReturn_packChar;
  epv->f_packInt = impl_sidlx_rmi_SimReturn_packInt;
  epv->f_packLong = impl_sidlx_rmi_SimReturn_packLong;
  epv->f_packOpaque = impl_sidlx_rmi_SimReturn_packOpaque;
  epv->f_packFloat = impl_sidlx_rmi_SimReturn_packFloat;
  epv->f_packDouble = impl_sidlx_rmi_SimReturn_packDouble;
  epv->f_packFcomplex = impl_sidlx_rmi_SimReturn_packFcomplex;
  epv->f_packDcomplex = impl_sidlx_rmi_SimReturn_packDcomplex;
  epv->f_packString = impl_sidlx_rmi_SimReturn_packString;
  epv->f_packSerializable = impl_sidlx_rmi_SimReturn_packSerializable;
  epv->f_packBoolArray = impl_sidlx_rmi_SimReturn_packBoolArray;
  epv->f_packCharArray = impl_sidlx_rmi_SimReturn_packCharArray;
  epv->f_packIntArray = impl_sidlx_rmi_SimReturn_packIntArray;
  epv->f_packLongArray = impl_sidlx_rmi_SimReturn_packLongArray;
  epv->f_packOpaqueArray = impl_sidlx_rmi_SimReturn_packOpaqueArray;
  epv->f_packFloatArray = impl_sidlx_rmi_SimReturn_packFloatArray;
  epv->f_packDoubleArray = impl_sidlx_rmi_SimReturn_packDoubleArray;
  epv->f_packFcomplexArray = impl_sidlx_rmi_SimReturn_packFcomplexArray;
  epv->f_packDcomplexArray = impl_sidlx_rmi_SimReturn_packDcomplexArray;
  epv->f_packStringArray = impl_sidlx_rmi_SimReturn_packStringArray;
  epv->f_packGenericArray = impl_sidlx_rmi_SimReturn_packGenericArray;
  epv->f_packSerializableArray = impl_sidlx_rmi_SimReturn_packSerializableArray;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimReturn__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_SimReturn__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseException__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_BaseException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseException(url, ar, _ex);
}

struct sidl_BaseException__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_BaseException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_BaseException(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializable(url, ar, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(url, ar, _ex);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_io_Serializer(bi, _ex);
}

struct sidl_rmi_Return__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_rmi_Return(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_Return(url, ar, _ex);
}

struct sidl_rmi_Return__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidl_rmi_Return(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidl_rmi_Return(bi, _ex);
}

struct sidlx_rmi_SimReturn__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(url, ar, _ex);
}

struct sidlx_rmi_SimReturn__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidlx_rmi_SimReturn(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_SimReturn(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimReturn_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fcast_sidlx_rmi_Socket(bi, _ex);
}

struct sidlx_rmi_SimReturn__data*
sidlx_rmi_SimReturn__get_data(sidlx_rmi_SimReturn self)
{
  return (struct sidlx_rmi_SimReturn__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimReturn__set_data(
  sidlx_rmi_SimReturn self,
  struct sidlx_rmi_SimReturn__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
