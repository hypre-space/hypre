/*
 * File:          sidlx_rmi_Simvocation_Skel.c
 * Symbol:        sidlx.rmi.Simvocation-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for sidlx.rmi.Simvocation
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_Simvocation_IOR.h"
#include "sidlx_rmi_Simvocation.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_Simvocation__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation__ctor(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation__ctor2(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation__dtor(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Invocation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Ticket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Ticket(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Simvocation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
extern
void
impl_sidlx_rmi_Simvocation_init(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* methodName,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simvocation_getMethodName(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Response
impl_sidlx_rmi_Simvocation_invokeMethod(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Ticket
impl_sidlx_rmi_Simvocation_invokeNonblocking(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_invokeOneWay(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packBool(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packChar(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packInt(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packLong(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packOpaque(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ void* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packFloat(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packDouble(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packFcomplex(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packDcomplex(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packString(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packSerializable(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ sidl_io_Serializable value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packBoolArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<bool> */ struct sidl_bool__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packCharArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<char> */ struct sidl_char__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packIntArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<int> */ struct sidl_int__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packLongArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<long> */ struct sidl_long__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packOpaqueArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<opaque> */ struct sidl_opaque__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packFloatArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<float> */ struct sidl_float__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packDoubleArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<double> */ struct sidl_double__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packFcomplexArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<fcomplex> */ struct sidl_fcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packDcomplexArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<dcomplex> */ struct sidl_dcomplex__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packStringArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<string> */ struct sidl_string__array* value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packGenericArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<> */ struct sidl__array* value,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packSerializableArray(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in array<sidl.io.Serializable> */ struct sidl_io_Serializable__array* 
    value,
  /* in */ int32_t ordering,
  /* in */ int32_t dimen,
  /* in */ sidl_bool reuse_array,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Invocation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Ticket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Ticket__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Ticket(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Simvocation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_Simvocation__set_epv(struct sidlx_rmi_Simvocation__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_Simvocation__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_Simvocation__ctor2;
  epv->f__dtor = impl_sidlx_rmi_Simvocation__dtor;
  epv->f_init = impl_sidlx_rmi_Simvocation_init;
  epv->f_getMethodName = impl_sidlx_rmi_Simvocation_getMethodName;
  epv->f_invokeMethod = impl_sidlx_rmi_Simvocation_invokeMethod;
  epv->f_invokeNonblocking = impl_sidlx_rmi_Simvocation_invokeNonblocking;
  epv->f_invokeOneWay = impl_sidlx_rmi_Simvocation_invokeOneWay;
  epv->f_packBool = impl_sidlx_rmi_Simvocation_packBool;
  epv->f_packChar = impl_sidlx_rmi_Simvocation_packChar;
  epv->f_packInt = impl_sidlx_rmi_Simvocation_packInt;
  epv->f_packLong = impl_sidlx_rmi_Simvocation_packLong;
  epv->f_packOpaque = impl_sidlx_rmi_Simvocation_packOpaque;
  epv->f_packFloat = impl_sidlx_rmi_Simvocation_packFloat;
  epv->f_packDouble = impl_sidlx_rmi_Simvocation_packDouble;
  epv->f_packFcomplex = impl_sidlx_rmi_Simvocation_packFcomplex;
  epv->f_packDcomplex = impl_sidlx_rmi_Simvocation_packDcomplex;
  epv->f_packString = impl_sidlx_rmi_Simvocation_packString;
  epv->f_packSerializable = impl_sidlx_rmi_Simvocation_packSerializable;
  epv->f_packBoolArray = impl_sidlx_rmi_Simvocation_packBoolArray;
  epv->f_packCharArray = impl_sidlx_rmi_Simvocation_packCharArray;
  epv->f_packIntArray = impl_sidlx_rmi_Simvocation_packIntArray;
  epv->f_packLongArray = impl_sidlx_rmi_Simvocation_packLongArray;
  epv->f_packOpaqueArray = impl_sidlx_rmi_Simvocation_packOpaqueArray;
  epv->f_packFloatArray = impl_sidlx_rmi_Simvocation_packFloatArray;
  epv->f_packDoubleArray = impl_sidlx_rmi_Simvocation_packDoubleArray;
  epv->f_packFcomplexArray = impl_sidlx_rmi_Simvocation_packFcomplexArray;
  epv->f_packDcomplexArray = impl_sidlx_rmi_Simvocation_packDcomplexArray;
  epv->f_packStringArray = impl_sidlx_rmi_Simvocation_packStringArray;
  epv->f_packGenericArray = impl_sidlx_rmi_Simvocation_packGenericArray;
  epv->f_packSerializableArray = 
    impl_sidlx_rmi_Simvocation_packSerializableArray;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_Simvocation__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_Simvocation__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_RuntimeException(url, ar,
    _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializable(url, ar, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(url, ar, _ex);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_io_Serializer(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_io_Serializer(bi, _ex);
}

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(url, ar, _ex);
}

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_rmi_Invocation(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Invocation(bi, _ex);
}

struct sidl_rmi_Response__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(url, ar, _ex);
}

struct sidl_rmi_Response__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_rmi_Response(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Response(bi, _ex);
}

struct sidl_rmi_Ticket__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Ticket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Ticket(url, ar, _ex);
}

struct sidl_rmi_Ticket__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidl_rmi_Ticket(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidl_rmi_Ticket(bi, _ex);
}

struct sidlx_rmi_Simvocation__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(url, ar,
    _ex);
}

struct sidlx_rmi_Simvocation__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Simvocation(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Simvocation(bi, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(url, ar, _ex);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Socket(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fcast_sidlx_rmi_Socket(bi, _ex);
}

struct sidlx_rmi_Simvocation__data*
sidlx_rmi_Simvocation__get_data(sidlx_rmi_Simvocation self)
{
  return (struct sidlx_rmi_Simvocation__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_Simvocation__set_data(
  sidlx_rmi_Simvocation self,
  struct sidlx_rmi_Simvocation__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
