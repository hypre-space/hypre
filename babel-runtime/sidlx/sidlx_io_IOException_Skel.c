/*
 * File:          sidlx_io_IOException_Skel.c
 * Symbol:        sidlx.io.IOException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side glue code for sidlx.io.IOException
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.10
 */

#include "sidlx_io_IOException_IOR.h"
#include "sidlx_io_IOException.h"
#include <stddef.h>

extern
void
impl_sidlx_io_IOException__load(
  void);

extern
void
impl_sidlx_io_IOException__ctor(
  /* in */ sidlx_io_IOException self);

extern
void
impl_sidlx_io_IOException__dtor(
  /* in */ sidlx_io_IOException self);

extern struct sidl_SIDLException__object* 
  impl_sidlx_io_IOException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_IOException_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_IOException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_IOException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_io_IOException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_IOException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidl_SIDLException__object* 
  impl_sidlx_io_IOException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj);
extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_IOException_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_IOException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_IOException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_io_IOException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_IOException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_IOException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_io_IOException__set_epv(struct sidlx_io_IOException__epv *epv)
{
  epv->f__ctor = impl_sidlx_io_IOException__ctor;
  epv->f__dtor = impl_sidlx_io_IOException__dtor;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_io_IOException__call_load(void) { 
  impl_sidlx_io_IOException__load();
}
struct sidl_SIDLException__object* 
  skel_sidlx_io_IOException_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_IOException_fconnect_sidl_SIDLException(url, _ex);
}

char* skel_sidlx_io_IOException_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) { 
  return impl_sidlx_io_IOException_fgetURL_sidl_SIDLException(obj);
}

struct sidlx_io_IOException__object* 
  skel_sidlx_io_IOException_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_IOException_fconnect_sidlx_io_IOException(url, _ex);
}

char* skel_sidlx_io_IOException_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) { 
  return impl_sidlx_io_IOException_fgetURL_sidlx_io_IOException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_io_IOException_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_IOException_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_io_IOException_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_io_IOException_fgetURL_sidl_ClassInfo(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_io_IOException_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_IOException_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_io_IOException_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_io_IOException_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseException__object* 
  skel_sidlx_io_IOException_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_IOException_fconnect_sidl_BaseException(url, _ex);
}

char* skel_sidlx_io_IOException_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj) { 
  return impl_sidlx_io_IOException_fgetURL_sidl_BaseException(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_io_IOException_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_IOException_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_io_IOException_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_io_IOException_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_io_IOException__data*
sidlx_io_IOException__get_data(sidlx_io_IOException self)
{
  return (struct sidlx_io_IOException__data*)(self ? self->d_data : NULL);
}

void sidlx_io_IOException__set_data(
  sidlx_io_IOException self,
  struct sidlx_io_IOException__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
