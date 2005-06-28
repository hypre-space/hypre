/*
 * File:          sidlx_io_TxtOStream_Impl.h
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.io.TxtOStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_sidlx_io_TxtOStream_Impl_h
#define included_sidlx_io_TxtOStream_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_io_TxtOStream_h
#include "sidlx_io_TxtOStream.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_io_OStream_h
#include "sidlx_io_OStream.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._includes) */

/*
 * Private data for class sidlx.io.TxtOStream
 */

struct sidlx_io_TxtOStream__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._data) */
  int filedes;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_io_TxtOStream__data*
sidlx_io_TxtOStream__get_data(
  sidlx_io_TxtOStream);

extern void
sidlx_io_TxtOStream__set_data(
  sidlx_io_TxtOStream,
  struct sidlx_io_TxtOStream__data*);

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

/*
 * User-defined object methods
 */

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
  /* in */ struct sidl_char__array* data,
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
#ifdef __cplusplus
}
#endif
#endif
