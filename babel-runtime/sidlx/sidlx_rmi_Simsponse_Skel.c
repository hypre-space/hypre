/*
 * File:          sidlx_rmi_Simsponse_Skel.c
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_rmi_Simsponse_IOR.h"
#include "sidlx_rmi_Simsponse.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_Simsponse__load(
  void);

extern
void
impl_sidlx_rmi_Simsponse__ctor(
  /* in */ sidlx_rmi_Simsponse self);

extern
void
impl_sidlx_rmi_Simsponse__dtor(
  /* in */ sidlx_rmi_Simsponse self);

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_Simsponse_init(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getMethodName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getClassName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getObjectID(
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
sidl_BaseException
impl_sidlx_rmi_Simsponse_getExceptionThrown(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_Simsponse_done(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_Simsponse__set_epv(struct sidlx_rmi_Simsponse__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_Simsponse__ctor;
  epv->f__dtor = impl_sidlx_rmi_Simsponse__dtor;
  epv->f_init = impl_sidlx_rmi_Simsponse_init;
  epv->f_getMethodName = impl_sidlx_rmi_Simsponse_getMethodName;
  epv->f_getClassName = impl_sidlx_rmi_Simsponse_getClassName;
  epv->f_getObjectID = impl_sidlx_rmi_Simsponse_getObjectID;
  epv->f_unpackBool = impl_sidlx_rmi_Simsponse_unpackBool;
  epv->f_unpackChar = impl_sidlx_rmi_Simsponse_unpackChar;
  epv->f_unpackInt = impl_sidlx_rmi_Simsponse_unpackInt;
  epv->f_unpackLong = impl_sidlx_rmi_Simsponse_unpackLong;
  epv->f_unpackFloat = impl_sidlx_rmi_Simsponse_unpackFloat;
  epv->f_unpackDouble = impl_sidlx_rmi_Simsponse_unpackDouble;
  epv->f_unpackFcomplex = impl_sidlx_rmi_Simsponse_unpackFcomplex;
  epv->f_unpackDcomplex = impl_sidlx_rmi_Simsponse_unpackDcomplex;
  epv->f_unpackString = impl_sidlx_rmi_Simsponse_unpackString;
  epv->f_getExceptionThrown = impl_sidlx_rmi_Simsponse_getExceptionThrown;
  epv->f_done = impl_sidlx_rmi_Simsponse_done;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_Simsponse__call_load(void) { 
  impl_sidlx_rmi_Simsponse__load();
}
struct sidl_rmi_Response__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(obj);
}

struct sidlx_rmi_Simsponse__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(obj);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseException__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_Simsponse__data*
sidlx_rmi_Simsponse__get_data(sidlx_rmi_Simsponse self)
{
  return (struct sidlx_rmi_Simsponse__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_Simsponse__set_data(
  sidlx_rmi_Simsponse self,
  struct sidlx_rmi_Simsponse__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
