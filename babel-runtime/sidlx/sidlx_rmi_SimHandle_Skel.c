/*
 * File:          sidlx_rmi_SimHandle_Skel.c
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.0
 * Description:   Server-side glue code for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_SimHandle_IOR.h"
#include "sidlx_rmi_SimHandle.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_SimHandle__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimHandle__ctor(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimHandle__ctor2(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimHandle__dtor(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_InstanceHandle(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_Invocation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidlx_rmi_SimHandle(void* bi,
  sidl_BaseInterface* _ex);
extern
sidl_bool
impl_sidlx_rmi_SimHandle_initCreate(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimHandle_initConnect(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* in */ sidl_bool ar,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_io_Serializable
impl_sidlx_rmi_SimHandle_initUnserialize(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getProtocol(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getObjectID(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getObjectURL(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Invocation
impl_sidlx_rmi_SimHandle_createInvocation(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* methodName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimHandle_close(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_InstanceHandle(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_Invocation(void* bi,
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidlx_rmi_SimHandle(void* bi,
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_SimHandle__set_epv(struct sidlx_rmi_SimHandle__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_SimHandle__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_SimHandle__ctor2;
  epv->f__dtor = impl_sidlx_rmi_SimHandle__dtor;
  epv->f_initCreate = impl_sidlx_rmi_SimHandle_initCreate;
  epv->f_initConnect = impl_sidlx_rmi_SimHandle_initConnect;
  epv->f_initUnserialize = impl_sidlx_rmi_SimHandle_initUnserialize;
  epv->f_getProtocol = impl_sidlx_rmi_SimHandle_getProtocol;
  epv->f_getObjectID = impl_sidlx_rmi_SimHandle_getObjectID;
  epv->f_getObjectURL = impl_sidlx_rmi_SimHandle_getObjectURL;
  epv->f_createInvocation = impl_sidlx_rmi_SimHandle_createInvocation;
  epv->f_close = impl_sidlx_rmi_SimHandle_close;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_SimHandle__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_SimHandle__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_BaseClass(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(url, ar, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_BaseInterface(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_ClassInfo(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_RuntimeException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_RuntimeException(url, ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_RuntimeException(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_io_Serializable(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_io_Serializable(url, ar, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_io_Serializable(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_rmi_InstanceHandle__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(url, ar,
    _ex);
}

struct sidl_rmi_InstanceHandle__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_rmi_InstanceHandle(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_InstanceHandle(bi, _ex);
}

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(url, ar, _ex);
}

struct sidl_rmi_Invocation__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidl_rmi_Invocation(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_Invocation(bi, _ex);
}

struct sidlx_rmi_SimHandle__object* 
  skel_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(url, ar, _ex);
}

struct sidlx_rmi_SimHandle__object* 
  skel_sidlx_rmi_SimHandle_fcast_sidlx_rmi_SimHandle(void* bi,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_SimHandle_fcast_sidlx_rmi_SimHandle(bi, _ex);
}

struct sidlx_rmi_SimHandle__data*
sidlx_rmi_SimHandle__get_data(sidlx_rmi_SimHandle self)
{
  return (struct sidlx_rmi_SimHandle__data*)(self ? self->d_data : NULL);
}

void sidlx_rmi_SimHandle__set_data(
  sidlx_rmi_SimHandle self,
  struct sidlx_rmi_SimHandle__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
