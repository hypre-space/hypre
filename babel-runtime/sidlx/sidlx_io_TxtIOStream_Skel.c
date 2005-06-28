/*
 * File:          sidlx_io_TxtIOStream_Skel.c
 * Symbol:        sidlx.io.TxtIOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for sidlx.io.TxtIOStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "sidlx_io_TxtIOStream_IOR.h"
#include "sidlx_io_TxtIOStream.h"
#include <stddef.h>

extern
void
impl_sidlx_io_TxtIOStream__load(
  void);

extern
void
impl_sidlx_io_TxtIOStream__ctor(
  /* in */ sidlx_io_TxtIOStream self);

extern
void
impl_sidlx_io_TxtIOStream__dtor(
  /* in */ sidlx_io_TxtIOStream self);

extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_io_TxtIOStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_TxtIOStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_TxtIOStream(struct 
  sidlx_io_TxtIOStream__object* obj);
extern struct sidlx_io_OStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_io_IOStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOStream(struct 
  sidlx_io_IOStream__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_io_IStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj);
extern
void
impl_sidlx_io_TxtIOStream_setFD(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t fd);

extern
sidl_bool
impl_sidlx_io_TxtIOStream_atEnd(
  /* in */ sidlx_io_TxtIOStream self);

extern
int32_t
impl_sidlx_io_TxtIOStream_read(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t nbytes,
  /* out */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_io_TxtIOStream_readline(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getBool(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ sidl_bool* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getChar(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ char* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getInt(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ int32_t* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getLong(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ int64_t* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getFloat(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ float* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getDouble(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ double* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getFcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_fcomplex* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getDcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_dcomplex* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_getString(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ char** item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_flush(
  /* in */ sidlx_io_TxtIOStream self);

extern
int32_t
impl_sidlx_io_TxtIOStream_write(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putBool(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putChar(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putInt(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putLong(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putFloat(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putDouble(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putFcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putDcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIOStream_putString(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_io_TxtIOStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_TxtIOStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_TxtIOStream(struct 
  sidlx_io_TxtIOStream__object* obj);
extern struct sidlx_io_OStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidlx_io_IOStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOStream(struct 
  sidlx_io_IOStream__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_io_IStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj);
static int32_t
skel_sidlx_io_TxtIOStream_read(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t nbytes,
  /* out */ struct sidl_char__array** data,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_char__array* data_proxy = NULL;
  _return =
    impl_sidlx_io_TxtIOStream_read(
      self,
      nbytes,
      &data_proxy,
      _ex);
  *data = sidl_char__array_ensure(data_proxy, 1, sidl_row_major_order);
  sidl_char__array_deleteRef(data_proxy);
  return _return;
}

static int32_t
skel_sidlx_io_TxtIOStream_readline(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_char__array** data,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_char__array* data_proxy = NULL;
  _return =
    impl_sidlx_io_TxtIOStream_readline(
      self,
      &data_proxy,
      _ex);
  *data = sidl_char__array_ensure(data_proxy, 1, sidl_row_major_order);
  sidl_char__array_deleteRef(data_proxy);
  return _return;
}

static int32_t
skel_sidlx_io_TxtIOStream_write(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_char__array* data,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_char__array* data_proxy = sidl_char__array_ensure(data, 1,
    sidl_row_major_order);
  _return =
    impl_sidlx_io_TxtIOStream_write(
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
sidlx_io_TxtIOStream__set_epv(struct sidlx_io_TxtIOStream__epv *epv)
{
  epv->f__ctor = impl_sidlx_io_TxtIOStream__ctor;
  epv->f__dtor = impl_sidlx_io_TxtIOStream__dtor;
  epv->f_setFD = impl_sidlx_io_TxtIOStream_setFD;
  epv->f_atEnd = impl_sidlx_io_TxtIOStream_atEnd;
  epv->f_read = skel_sidlx_io_TxtIOStream_read;
  epv->f_readline = skel_sidlx_io_TxtIOStream_readline;
  epv->f_getBool = impl_sidlx_io_TxtIOStream_getBool;
  epv->f_getChar = impl_sidlx_io_TxtIOStream_getChar;
  epv->f_getInt = impl_sidlx_io_TxtIOStream_getInt;
  epv->f_getLong = impl_sidlx_io_TxtIOStream_getLong;
  epv->f_getFloat = impl_sidlx_io_TxtIOStream_getFloat;
  epv->f_getDouble = impl_sidlx_io_TxtIOStream_getDouble;
  epv->f_getFcomplex = impl_sidlx_io_TxtIOStream_getFcomplex;
  epv->f_getDcomplex = impl_sidlx_io_TxtIOStream_getDcomplex;
  epv->f_getString = impl_sidlx_io_TxtIOStream_getString;
  epv->f_flush = impl_sidlx_io_TxtIOStream_flush;
  epv->f_write = skel_sidlx_io_TxtIOStream_write;
  epv->f_putBool = impl_sidlx_io_TxtIOStream_putBool;
  epv->f_putChar = impl_sidlx_io_TxtIOStream_putChar;
  epv->f_putInt = impl_sidlx_io_TxtIOStream_putInt;
  epv->f_putLong = impl_sidlx_io_TxtIOStream_putLong;
  epv->f_putFloat = impl_sidlx_io_TxtIOStream_putFloat;
  epv->f_putDouble = impl_sidlx_io_TxtIOStream_putDouble;
  epv->f_putFcomplex = impl_sidlx_io_TxtIOStream_putFcomplex;
  epv->f_putDcomplex = impl_sidlx_io_TxtIOStream_putDcomplex;
  epv->f_putString = impl_sidlx_io_TxtIOStream_putString;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_io_TxtIOStream__call_load(void) { 
  impl_sidlx_io_TxtIOStream__load();
}
struct sidlx_io_IOException__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOException(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_io_TxtIOStream__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidlx_io_TxtIOStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_TxtIOStream(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidlx_io_TxtIOStream(struct 
  sidlx_io_TxtIOStream__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_TxtIOStream(obj);
}

struct sidlx_io_OStream__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_OStream(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_OStream(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseInterface(obj);
}

struct sidlx_io_IOStream__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOStream(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOStream(struct 
  sidlx_io_IOStream__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOStream(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_io_IStream__object* 
  skel_sidlx_io_TxtIOStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IStream(url, _ex);
}

char* skel_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj) { 
  return impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IStream(obj);
}

struct sidlx_io_TxtIOStream__data*
sidlx_io_TxtIOStream__get_data(sidlx_io_TxtIOStream self)
{
  return (struct sidlx_io_TxtIOStream__data*)(self ? self->d_data : NULL);
}

void sidlx_io_TxtIOStream__set_data(
  sidlx_io_TxtIOStream self,
  struct sidlx_io_TxtIOStream__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
