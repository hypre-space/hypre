/*
 * File:          sidlx_rmi_NoServerException_Skel.c
 * Symbol:        sidlx.rmi.NoServerException-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side glue code for sidlx.rmi.NoServerException
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidlx_rmi_NoServerException_IOR.h"
#include "sidlx_rmi_NoServerException.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_NoServerException__load(
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_NoServerException__ctor(
  /* in */ sidlx_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_NoServerException__ctor2(
  /* in */ sidlx_rmi_NoServerException self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_NoServerException__dtor(
  /* in */ sidlx_rmi_NoServerException self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_BaseException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_SIDLException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_SIDLException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Deserializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_Deserializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_IOException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_IOException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializable(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_Serializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_rmi_NetworkException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_rmi_NetworkException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_NoServerException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidlx_rmi_NoServerException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_NoServerException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidlx_rmi_NoServerException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_BaseException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_SIDLException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_SIDLException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Deserializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_Deserializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_IOException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_IOException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializable(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_io_Serializer(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidl_rmi_NetworkException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidl_rmi_NetworkException(void* bi, 
  sidl_BaseInterface* _ex);
extern struct sidlx_rmi_NoServerException__object* 
  impl_sidlx_rmi_NoServerException_fconnect_sidlx_rmi_NoServerException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex);
extern struct sidlx_rmi_NoServerException__object* 
  impl_sidlx_rmi_NoServerException_fcast_sidlx_rmi_NoServerException(void* bi, 
  sidl_BaseInterface* _ex);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_NoServerException__set_epv(struct sidlx_rmi_NoServerException__epv 
  *epv)
{
  epv->f__ctor = impl_sidlx_rmi_NoServerException__ctor;
  epv->f__ctor2 = impl_sidlx_rmi_NoServerException__ctor2;
  epv->f__dtor = impl_sidlx_rmi_NoServerException__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_NoServerException__call_load(void) { 
  sidl_BaseInterface _throwaway_exception = NULL;
  impl_sidlx_rmi_NoServerException__load(&_throwaway_exception);
}
struct sidl_BaseClass__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_BaseClass(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseClass(url, ar, _ex);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_BaseClass(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_BaseClass(bi, _ex);
}

struct sidl_BaseException__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_BaseException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseException(url, ar, 
    _ex);
}

struct sidl_BaseException__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_BaseException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_BaseException(bi, _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_BaseInterface(url, ar, 
    _ex);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_BaseInterface(bi, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_ClassInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_ClassInfo(url, ar, _ex);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_ClassInfo(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_ClassInfo(bi, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_RuntimeException(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_RuntimeException(url, 
    ar, _ex);
}

struct sidl_RuntimeException__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_RuntimeException(bi, _ex);
}

struct sidl_SIDLException__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_SIDLException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_SIDLException(url, ar, 
    _ex);
}

struct sidl_SIDLException__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_SIDLException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_SIDLException(bi, _ex);
}

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_io_Deserializer(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Deserializer(url, ar,
    _ex);
}

struct sidl_io_Deserializer__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_io_Deserializer(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_io_Deserializer(bi, _ex);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_io_IOException(const char* url,
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_io_IOException(url, ar, 
    _ex);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_io_IOException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_io_IOException(bi, _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializable(const char* 
  url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializable(url, ar,
    _ex);
}

struct sidl_io_Serializable__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_io_Serializable(bi, _ex);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_io_Serializer(url, ar, 
    _ex);
}

struct sidl_io_Serializer__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_io_Serializer(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_io_Serializer(bi, _ex);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidl_rmi_NetworkException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidl_rmi_NetworkException(
    url, ar, _ex);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidl_rmi_NetworkException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidl_rmi_NetworkException(bi, 
    _ex);
}

struct sidlx_rmi_NoServerException__object* 
  skel_sidlx_rmi_NoServerException_fconnect_sidlx_rmi_NoServerException(const 
  char* url, sidl_bool ar, sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fconnect_sidlx_rmi_NoServerException(
    url, ar, _ex);
}

struct sidlx_rmi_NoServerException__object* 
  skel_sidlx_rmi_NoServerException_fcast_sidlx_rmi_NoServerException(void* bi, 
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_NoServerException_fcast_sidlx_rmi_NoServerException(bi, 
    _ex);
}

struct sidlx_rmi_NoServerException__data*
sidlx_rmi_NoServerException__get_data(sidlx_rmi_NoServerException self)
{
  return (struct sidlx_rmi_NoServerException__data*)(self ? self->d_data : 
    NULL);
}

void sidlx_rmi_NoServerException__set_data(
  sidlx_rmi_NoServerException self,
  struct sidlx_rmi_NoServerException__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
