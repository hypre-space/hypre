/*
 * File:          sidlx_io_TxtIOStream_Impl.h
 * Symbol:        sidlx.io.TxtIOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.TxtIOStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_io_TxtIOStream_Impl_h
#define included_sidlx_io_TxtIOStream_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif
#ifndef included_sidlx_io_TxtIOStream_h
#include "sidlx_io_TxtIOStream.h"
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

extern void
impl_sidlx_io_TxtIOStream__ctor(
  sidlx_io_TxtIOStream);

extern void
impl_sidlx_io_TxtIOStream__dtor(
  sidlx_io_TxtIOStream);

/*
 * User-defined object methods
 */

extern void
impl_sidlx_io_TxtIOStream_setFD(
  sidlx_io_TxtIOStream,
  int32_t);

extern sidl_bool
impl_sidlx_io_TxtIOStream_atEnd(
  sidlx_io_TxtIOStream);

extern int32_t
impl_sidlx_io_TxtIOStream_read(
  sidlx_io_TxtIOStream,
  int32_t,
  struct sidl_char__array**,
  sidl_BaseInterface*);

extern int32_t
impl_sidlx_io_TxtIOStream_readline(
  sidlx_io_TxtIOStream,
  struct sidl_char__array**,
  sidl_BaseInterface*);

extern void
impl_sidlx_io_TxtIOStream_flush(
  sidlx_io_TxtIOStream);

extern int32_t
impl_sidlx_io_TxtIOStream_write(
  sidlx_io_TxtIOStream,
  struct sidl_char__array*,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
