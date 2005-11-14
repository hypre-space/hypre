/*
 * File:          sidlx_rmi_SimCall_Skel.c
 * Symbol:        sidlx.rmi.SimCall-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for sidlx.rmi.SimCall
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_rmi_SimCall_IOR.h"
#include "sidlx_rmi_SimCall.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimCall__load(
  void);

extern
void
impl_sidlx_rmi_SimCall__ctor(
  /* in */ sidlx_rmi_SimCall self);

extern
void
impl_sidlx_rmi_SimCall__dtor(
  /* in */ sidlx_rmi_SimCall self);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(struct 
  sidlx_rmi_SimCall__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
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
char*
impl_sidlx_rmi_SimCall_getClassName(
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

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(struct 
  sidlx_rmi_SimCall__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimCall__set_epv(struct sidlx_rmi_SimCall__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimCall__ctor;
  epv->f__dtor = impl_sidlx_rmi_SimCall__dtor;
  epv->f_init = impl_sidlx_rmi_SimCall_init;
  epv->f_getMethodName = impl_sidlx_rmi_SimCall_getMethodName;
  epv->f_getObjectID = impl_sidlx_rmi_SimCall_getObjectID;
  epv->f_getClassName = impl_sidlx_rmi_SimCall_getClassName;
  epv->f_getCallType = impl_sidlx_rmi_SimCall_getCallType;
  epv->f_unpackBool = impl_sidlx_rmi_SimCall_unpackBool;
  epv->f_unpackChar = impl_sidlx_rmi_SimCall_unpackChar;
  epv->f_unpackInt = impl_sidlx_rmi_SimCall_unpackInt;
  epv->f_unpackLong = impl_sidlx_rmi_SimCall_unpackLong;
  epv->f_unpackFloat = impl_sidlx_rmi_SimCall_unpackFloat;
  epv->f_unpackDouble = impl_sidlx_rmi_SimCall_unpackDouble;
  epv->f_unpackFcomplex = impl_sidlx_rmi_SimCall_unpackFcomplex;
  epv->f_unpackDcomplex = impl_sidlx_rmi_SimCall_unpackDcomplex;
  epv->f_unpackString = impl_sidlx_rmi_SimCall_unpackString;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimCall__call_load(void) { 
  impl_sidlx_rmi_SimCall__load();
}
struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(obj);
}

struct sidlx_rmi_SimCall__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(struct 
  sidlx_rmi_SimCall__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(obj);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(obj);
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
