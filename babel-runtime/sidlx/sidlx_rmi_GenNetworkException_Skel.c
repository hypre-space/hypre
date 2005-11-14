/*
 * File:          sidlx_rmi_GenNetworkException_Skel.c
 * Symbol:        sidlx.rmi.GenNetworkException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side glue code for sidlx.rmi.GenNetworkException
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidlx_rmi_GenNetworkException_IOR.h"
#include "sidlx_rmi_GenNetworkException.h"
#include <stddef.h>

extern
void
impl_sidlx_rmi_GenNetworkException__load(
  void);

extern
void
impl_sidlx_rmi_GenNetworkException__ctor(
  /* in */ sidlx_rmi_GenNetworkException self);

extern
void
impl_sidlx_rmi_GenNetworkException__dtor(
  /* in */ sidlx_rmi_GenNetworkException self);

extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_rmi_GenNetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_rmi_GenNetworkException__set_epv(struct 
  sidlx_rmi_GenNetworkException__epv *epv)
{
  epv->f__ctor = impl_sidlx_rmi_GenNetworkException__ctor;
  epv->f__dtor = impl_sidlx_rmi_GenNetworkException__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_rmi_GenNetworkException__call_load(void) { 
  impl_sidlx_rmi_GenNetworkException__load();
}
struct sidlx_rmi_GenNetworkException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
  char* url, sidl_BaseInterface *_ex) { 
  return 
    impl_sidlx_rmi_GenNetworkException_fconnect_sidlx_rmi_GenNetworkException(
    url, _ex);
}

char* 
  skel_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
  struct sidlx_rmi_GenNetworkException__object* obj) { 
  return 
    impl_sidlx_rmi_GenNetworkException_fgetURL_sidlx_rmi_GenNetworkException(
    obj);
}

struct sidl_SIDLException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_GenNetworkException_fconnect_sidl_SIDLException(url,
    _ex);
}

char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) { 
  return impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_SIDLException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_GenNetworkException_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_io_IOException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_GenNetworkException_fconnect_sidl_io_IOException(url,
    _ex);
}

char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj) { 
  return impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_io_IOException(obj);
}

struct sidl_rmi_NetworkException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(char* 
  url, sidl_BaseInterface *_ex) { 
  return 
    impl_sidlx_rmi_GenNetworkException_fconnect_sidl_rmi_NetworkException(url,
    _ex);
}

char* 
  skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) { 
  return 
    impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_rmi_NetworkException(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseInterface(url,
    _ex);
}

char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseException__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseException(url,
    _ex);
}

char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) { 
  return impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseException(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_rmi_GenNetworkException_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_rmi_GenNetworkException_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_rmi_GenNetworkException__data*
sidlx_rmi_GenNetworkException__get_data(sidlx_rmi_GenNetworkException self)
{
  return (struct sidlx_rmi_GenNetworkException__data*)(self ? self->d_data : 
    NULL);
}

void sidlx_rmi_GenNetworkException__set_data(
  sidlx_rmi_GenNetworkException self,
  struct sidlx_rmi_GenNetworkException__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
