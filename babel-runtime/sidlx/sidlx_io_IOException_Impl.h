/*
 * File:          sidlx_io_IOException_Impl.h
 * Symbol:        sidlx.io.IOException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.io.IOException
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_sidlx_io_IOException_Impl_h
#define included_sidlx_io_IOException_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.io.IOException._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.io.IOException._includes) */

/*
 * Private data for class sidlx.io.IOException
 */

struct sidlx_io_IOException__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.io.IOException._data) */
  /* Put private data members here... */
  int ignore; /* dummy to force non-empty struct; remove if you add data */
  /* DO-NOT-DELETE splicer.end(sidlx.io.IOException._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_io_IOException__data*
sidlx_io_IOException__get_data(
  sidlx_io_IOException);

extern void
sidlx_io_IOException__set_data(
  sidlx_io_IOException,
  struct sidlx_io_IOException__data*);

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

/*
 * User-defined object methods
 */

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
}
#endif
#endif
