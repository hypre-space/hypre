/*
 * File:          sidlx_rmi_Simvocation_Skel.c
 * Symbol:        sidlx.rmi.Simvocation-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side glue code for sidlx.rmi.Simvocation
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidlx_rmi_Simvocation_IOR.h"
#include "sidlx_rmi_Simvocation.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_Simvocation__load(
  void);

extern
void
impl_sidlx_rmi_Simvocation__ctor(
  /* in */ sidlx_rmi_Simvocation self);

extern
void
impl_sidlx_rmi_Simvocation__dtor(
  /* in */ sidlx_rmi_Simvocation self);

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(struct 
  sidlx_rmi_Simvocation__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_Simvocation_init(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simvocation_getMethodName(
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
sidl_rmi_Response
impl_sidlx_rmi_Simvocation_invokeMethod(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(struct 
  sidlx_rmi_Simvocation__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_Simvocation__set_epv(struct sidlx_rmi_Simvocation__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_Simvocation__ctor;
  epv->f__dtor = impl_sidlx_rmi_Simvocation__dtor;
  epv->f_init = impl_sidlx_rmi_Simvocation_init;
  epv->f_getMethodName = impl_sidlx_rmi_Simvocation_getMethodName;
  epv->f_packBool = impl_sidlx_rmi_Simvocation_packBool;
  epv->f_packChar = impl_sidlx_rmi_Simvocation_packChar;
  epv->f_packInt = impl_sidlx_rmi_Simvocation_packInt;
  epv->f_packLong = impl_sidlx_rmi_Simvocation_packLong;
  epv->f_packFloat = impl_sidlx_rmi_Simvocation_packFloat;
  epv->f_packDouble = impl_sidlx_rmi_Simvocation_packDouble;
  epv->f_packFcomplex = impl_sidlx_rmi_Simvocation_packFcomplex;
  epv->f_packDcomplex = impl_sidlx_rmi_Simvocation_packDcomplex;
  epv->f_packString = impl_sidlx_rmi_Simvocation_packString;
  epv->f_invokeMethod = impl_sidlx_rmi_Simvocation_invokeMethod;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_Simvocation__call_load(void) { 
  impl_sidlx_rmi_Simvocation__load();
}
struct sidl_rmi_Response__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(obj);
}

struct sidlx_rmi_Simvocation__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(struct 
  sidlx_rmi_Simvocation__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(obj);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(url,
    _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(obj);
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
