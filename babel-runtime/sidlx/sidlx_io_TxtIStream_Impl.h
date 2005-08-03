/*
 * File:          sidlx_io_TxtIStream_Impl.h
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_io_TxtIStream_Impl_h
#define included_sidlx_io_TxtIStream_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_io_TxtIStream_h
#include "sidlx_io_TxtIStream.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidlx_io_IStream_h
#include "sidlx_io_IStream.h"
#endif

#line 38 "../../../babel/runtime/sidlx/sidlx_io_TxtIStream_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._includes) */
#line 42 "sidlx_io_TxtIStream_Impl.h"

/*
 * Private data for class sidlx.io.TxtIStream
 */

struct sidlx_io_TxtIStream__data {
#line 47 "../../../babel/runtime/sidlx/sidlx_io_TxtIStream_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._data) */
  int filedes;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._data) */
#line 53 "sidlx_io_TxtIStream_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_io_TxtIStream__data*
sidlx_io_TxtIStream__get_data(
  sidlx_io_TxtIStream);

extern void
sidlx_io_TxtIStream__set_data(
  sidlx_io_TxtIStream,
  struct sidlx_io_TxtIStream__data*);

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

/*
 * User-defined object methods
 */

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
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex);

extern
int32_t
impl_sidlx_io_TxtIStream_readline(
  /* in */ sidlx_io_TxtIStream self,
  /* out array<char,row-major> */ struct sidl_char__array** data,
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
#ifdef __cplusplus
}
#endif
#endif
