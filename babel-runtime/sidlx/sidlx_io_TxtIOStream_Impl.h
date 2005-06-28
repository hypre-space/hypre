/*
 * File:          sidlx_io_TxtIOStream_Impl.h
 * Symbol:        sidlx.io.TxtIOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.io.TxtIOStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_sidlx_io_TxtIOStream_Impl_h
#define included_sidlx_io_TxtIOStream_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_io_TxtIOStream_h
#include "sidlx_io_TxtIOStream.h"
#endif
#ifndef included_sidlx_io_OStream_h
#include "sidlx_io_OStream.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidlx_io_IOStream_h
#include "sidlx_io_IOStream.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidlx_io_IStream_h
#include "sidlx_io_IStream.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._includes) */

/*
 * Private data for class sidlx.io.TxtIOStream
 */

struct sidlx_io_TxtIOStream__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_io_TxtIOStream__data*
sidlx_io_TxtIOStream__get_data(
  sidlx_io_TxtIOStream);

extern void
sidlx_io_TxtIOStream__set_data(
  sidlx_io_TxtIOStream,
  struct sidlx_io_TxtIOStream__data*);

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

/*
 * User-defined object methods
 */

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
#ifdef __cplusplus
}
#endif
#endif
