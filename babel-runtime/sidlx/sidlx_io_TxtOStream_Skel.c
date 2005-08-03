/*
 * File:          sidlx_io_TxtOStream_Skel.c
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side glue code for sidlx.io.TxtOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidlx_io_TxtOStream_IOR.h"
#include "sidlx_io_TxtOStream.h"
#include <stddef.h>

extern
void
impl_sidlx_io_TxtOStream__load(
  void);

extern
void
impl_sidlx_io_TxtOStream__ctor(
  /* in */ sidlx_io_TxtOStream self);

extern
void
impl_sidlx_io_TxtOStream__dtor(
  /* in */ sidlx_io_TxtOStream self);

extern struct sidlx_io_TxtOStream__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_TxtOStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_TxtOStream(struct 
  sidlx_io_TxtOStream__object* obj);
extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_io_OStream__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_io_TxtOStream_setFD(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int32_t fd);

extern
void
impl_sidlx_io_TxtOStream_flush(
  /* in */ sidlx_io_TxtOStream self);

extern
int32_t
impl_sidlx_io_TxtOStream_write(
  /* in */ sidlx_io_TxtOStream self,
  /* in array<char,row-major> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putBool(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putChar(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putInt(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putLong(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putFloat(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putDouble(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putFcomplex(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putDcomplex(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtOStream_putString(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_io_TxtOStream__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_TxtOStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_TxtOStream(struct 
  sidlx_io_TxtOStream__object* obj);
extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_io_OStream__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
static int32_t
skel_sidlx_io_TxtOStream_write(
  /* in */ sidlx_io_TxtOStream self,
  /* in array<char,row-major> */ struct sidl_char__array* data,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_char__array* data_proxy = sidl_char__array_ensure(data, 1,
    sidl_row_major_order);
  _return =
    impl_sidlx_io_TxtOStream_write(
      self,
      data_proxy,
      _ex);
  sidl_char__array_deleteRef(data_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_io_TxtOStream__set_epv(struct sidlx_io_TxtOStream__epv *epv)
{
  epv->f__ctor = impl_sidlx_io_TxtOStream__ctor;
  epv->f__dtor = impl_sidlx_io_TxtOStream__dtor;
  epv->f_setFD = impl_sidlx_io_TxtOStream_setFD;
  epv->f_flush = impl_sidlx_io_TxtOStream_flush;
  epv->f_write = skel_sidlx_io_TxtOStream_write;
  epv->f_putBool = impl_sidlx_io_TxtOStream_putBool;
  epv->f_putChar = impl_sidlx_io_TxtOStream_putChar;
  epv->f_putInt = impl_sidlx_io_TxtOStream_putInt;
  epv->f_putLong = impl_sidlx_io_TxtOStream_putLong;
  epv->f_putFloat = impl_sidlx_io_TxtOStream_putFloat;
  epv->f_putDouble = impl_sidlx_io_TxtOStream_putDouble;
  epv->f_putFcomplex = impl_sidlx_io_TxtOStream_putFcomplex;
  epv->f_putDcomplex = impl_sidlx_io_TxtOStream_putDcomplex;
  epv->f_putString = impl_sidlx_io_TxtOStream_putString;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_io_TxtOStream__call_load(void) { 
  impl_sidlx_io_TxtOStream__load();
}
struct sidlx_io_TxtOStream__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidlx_io_TxtOStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtOStream_fconnect_sidlx_io_TxtOStream(url, _ex);
}

char* skel_sidlx_io_TxtOStream_fgetURL_sidlx_io_TxtOStream(struct 
  sidlx_io_TxtOStream__object* obj) { 
  return impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_TxtOStream(obj);
}

struct sidlx_io_IOException__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtOStream_fconnect_sidlx_io_IOException(url, _ex);
}

char* skel_sidlx_io_TxtOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) { 
  return impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_IOException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtOStream_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_io_TxtOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_io_TxtOStream_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_io_OStream__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtOStream_fconnect_sidlx_io_OStream(url, _ex);
}

char* skel_sidlx_io_TxtOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj) { 
  return impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_OStream(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtOStream_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_io_TxtOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_io_TxtOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtOStream_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_io_TxtOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_io_TxtOStream__data*
sidlx_io_TxtOStream__get_data(sidlx_io_TxtOStream self)
{
  return (struct sidlx_io_TxtOStream__data*)(self ? self->d_data : NULL);
}

void sidlx_io_TxtOStream__set_data(
  sidlx_io_TxtOStream self,
  struct sidlx_io_TxtOStream__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
