/*
 * File:          sidlx_rmi_SimReturn_Skel.c
 * Symbol:        sidlx.rmi.SimReturn-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.rmi.SimReturn
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_rmi_SimReturn_IOR.h"
#include "sidlx_rmi_SimReturn.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimReturn__load(
  void);

extern
void
impl_sidlx_rmi_SimReturn__ctor(
  /* in */ sidlx_rmi_SimReturn self);

extern
void
impl_sidlx_rmi_SimReturn__dtor(
  /* in */ sidlx_rmi_SimReturn self);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(struct 
  sidlx_rmi_SimReturn__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_SimReturn_init(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* methodName,
  /* in */ const char* className,
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

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(struct 
  sidlx_rmi_SimReturn__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimReturn__set_epv(struct sidlx_rmi_SimReturn__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimReturn__ctor;
  epv->f__dtor = impl_sidlx_rmi_SimReturn__dtor;
  epv->f_init = impl_sidlx_rmi_SimReturn_init;
  epv->f_getMethodName = impl_sidlx_rmi_SimReturn_getMethodName;
  epv->f_SendReturn = impl_sidlx_rmi_SimReturn_SendReturn;
  epv->f_packBool = impl_sidlx_rmi_SimReturn_packBool;
  epv->f_packChar = impl_sidlx_rmi_SimReturn_packChar;
  epv->f_packInt = impl_sidlx_rmi_SimReturn_packInt;
  epv->f_packLong = impl_sidlx_rmi_SimReturn_packLong;
  epv->f_packFloat = impl_sidlx_rmi_SimReturn_packFloat;
  epv->f_packDouble = impl_sidlx_rmi_SimReturn_packDouble;
  epv->f_packFcomplex = impl_sidlx_rmi_SimReturn_packFcomplex;
  epv->f_packDcomplex = impl_sidlx_rmi_SimReturn_packDcomplex;
  epv->f_packString = impl_sidlx_rmi_SimReturn_packString;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimReturn__call_load(void) { 
  impl_sidlx_rmi_SimReturn__load();
}
struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_rmi_SimReturn__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(struct 
  sidlx_rmi_SimReturn__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(obj);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidlx_rmi_Socket__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(obj);
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
