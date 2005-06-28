/*
 * File:          sidlx_io_TxtIStream_Skel.c
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side glue code for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "sidlx_io_TxtIStream_IOR.h"
#include "sidlx_io_TxtIStream.h"
#include <stddef.h>

extern
void
impl_sidlx_io_TxtIStream__load(
  void);

extern
void
impl_sidlx_io_TxtIStream__ctor(
  /* in */ sidlx_io_TxtIStream self);

extern
void
impl_sidlx_io_TxtIStream__dtor(
  /* in */ sidlx_io_TxtIStream self);

extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_io_TxtIStream__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_TxtIStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_TxtIStream(struct 
  sidlx_io_TxtIStream__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_io_IStream__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj);
extern
void
impl_sidlx_io_TxtIStream_setFD(
  /* in */ sidlx_io_TxtIStream self,
  /* in */ int32_t fd);

extern
sidl_bool
impl_sidlx_io_TxtIStream_atEnd(
  /* in */ sidlx_io_TxtIStream self);

extern
int32_t
impl_sidlx_io_TxtIStream_read(
  /* in */ sidlx_io_TxtIStream self,
  /* in */ int32_t nbytes,
  /* out */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_io_TxtIStream_readline(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getBool(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ sidl_bool* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getChar(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ char* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getInt(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ int32_t* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getLong(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ int64_t* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getFloat(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ float* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getDouble(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ double* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getFcomplex(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_fcomplex* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getDcomplex(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_dcomplex* item,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_io_TxtIStream_getString(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ char** item,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_io_TxtIStream__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_TxtIStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_TxtIStream(struct 
  sidlx_io_TxtIStream__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern struct sidlx_io_IStream__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj);
static int32_t
skel_sidlx_io_TxtIStream_read(
  /* in */ sidlx_io_TxtIStream self,
  /* in */ int32_t nbytes,
  /* out */ struct sidl_char__array** data,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_char__array* data_proxy = NULL;
  _return =
    impl_sidlx_io_TxtIStream_read(
      self,
      nbytes,
      &data_proxy,
      _ex);
  *data = sidl_char__array_ensure(data_proxy, 1, sidl_row_major_order);
  sidl_char__array_deleteRef(data_proxy);
  return _return;
}

static int32_t
skel_sidlx_io_TxtIStream_readline(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_char__array** data,
/* out */ sidl_BaseInterface *_ex)
{
  int32_t _return;
  struct sidl_char__array* data_proxy = NULL;
  _return =
    impl_sidlx_io_TxtIStream_readline(
      self,
      &data_proxy,
      _ex);
  *data = sidl_char__array_ensure(data_proxy, 1, sidl_row_major_order);
  sidl_char__array_deleteRef(data_proxy);
  return _return;
}

#ifdef __cplusplus
extern "C" {
#endif

void
sidlx_io_TxtIStream__set_epv(struct sidlx_io_TxtIStream__epv *epv)
{
  epv->f__ctor = impl_sidlx_io_TxtIStream__ctor;
  epv->f__dtor = impl_sidlx_io_TxtIStream__dtor;
  epv->f_setFD = impl_sidlx_io_TxtIStream_setFD;
  epv->f_atEnd = impl_sidlx_io_TxtIStream_atEnd;
  epv->f_read = skel_sidlx_io_TxtIStream_read;
  epv->f_readline = skel_sidlx_io_TxtIStream_readline;
  epv->f_getBool = impl_sidlx_io_TxtIStream_getBool;
  epv->f_getChar = impl_sidlx_io_TxtIStream_getChar;
  epv->f_getInt = impl_sidlx_io_TxtIStream_getInt;
  epv->f_getLong = impl_sidlx_io_TxtIStream_getLong;
  epv->f_getFloat = impl_sidlx_io_TxtIStream_getFloat;
  epv->f_getDouble = impl_sidlx_io_TxtIStream_getDouble;
  epv->f_getFcomplex = impl_sidlx_io_TxtIStream_getFcomplex;
  epv->f_getDcomplex = impl_sidlx_io_TxtIStream_getDcomplex;
  epv->f_getString = impl_sidlx_io_TxtIStream_getString;

}
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

void sidlx_io_TxtIStream__call_load(void) { 
  impl_sidlx_io_TxtIStream__load();
}
struct sidlx_io_IOException__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IOException(url, _ex);
}

char* skel_sidlx_io_TxtIStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) { 
  return impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IOException(obj);
}

struct sidl_ClassInfo__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIStream_fconnect_sidl_ClassInfo(url, _ex);
}

char* skel_sidlx_io_TxtIStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) { 
  return impl_sidlx_io_TxtIStream_fgetURL_sidl_ClassInfo(obj);
}

struct sidlx_io_TxtIStream__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidlx_io_TxtIStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIStream_fconnect_sidlx_io_TxtIStream(url, _ex);
}

char* skel_sidlx_io_TxtIStream_fgetURL_sidlx_io_TxtIStream(struct 
  sidlx_io_TxtIStream__object* obj) { 
  return impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_TxtIStream(obj);
}

struct sidl_BaseInterface__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIStream_fconnect_sidl_BaseInterface(url, _ex);
}

char* skel_sidlx_io_TxtIStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) { 
  return impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseInterface(obj);
}

struct sidl_BaseClass__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIStream_fconnect_sidl_BaseClass(url, _ex);
}

char* skel_sidlx_io_TxtIStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) { 
  return impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseClass(obj);
}

struct sidlx_io_IStream__object* 
  skel_sidlx_io_TxtIStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex) { 
  return impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IStream(url, _ex);
}

char* skel_sidlx_io_TxtIStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj) { 
  return impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IStream(obj);
}

struct sidlx_io_TxtIStream__data*
sidlx_io_TxtIStream__get_data(sidlx_io_TxtIStream self)
{
  return (struct sidlx_io_TxtIStream__data*)(self ? self->d_data : NULL);
}

void sidlx_io_TxtIStream__set_data(
  sidlx_io_TxtIStream self,
  struct sidlx_io_TxtIStream__data* data)
{
  if (self) {
    self->d_data = data;
  }
}
#ifdef __cplusplus
}
#endif
