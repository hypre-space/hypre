/*
 * File:          sidlx_io_TxtOStream_Impl.h
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.TxtOStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_io_TxtOStream_Impl_h
#define included_sidlx_io_TxtOStream_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif
#ifndef included_sidlx_io_TxtOStream_h
#include "sidlx_io_TxtOStream.h"
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

extern void
impl_sidlx_io_TxtOStream__ctor(
  sidlx_io_TxtOStream);

extern void
impl_sidlx_io_TxtOStream__dtor(
  sidlx_io_TxtOStream);

/*
 * User-defined object methods
 */

extern void
impl_sidlx_io_TxtOStream_setFD(
  sidlx_io_TxtOStream,
  int32_t);

extern void
impl_sidlx_io_TxtOStream_flush(
  sidlx_io_TxtOStream);

extern int32_t
impl_sidlx_io_TxtOStream_write(
  sidlx_io_TxtOStream,
  struct sidl_char__array*,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
