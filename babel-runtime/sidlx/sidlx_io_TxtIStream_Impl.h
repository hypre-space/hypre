/*
 * File:          sidlx_io_TxtIStream_Impl.h
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_io_TxtIStream_Impl_h
#define included_sidlx_io_TxtIStream_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_SIDLException_h
#include "sidl_SIDLException.h"
#endif
#ifndef included_sidlx_io_TxtIStream_h
#include "sidlx_io_TxtIStream.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._includes) */
/* Put additional include files here... */
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._includes) */

/*
 * Private data for class sidlx.io.TxtIStream
 */

struct sidlx_io_TxtIStream__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._data) */
  int filedes;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._data) */
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

extern void
impl_sidlx_io_TxtIStream__ctor(
  sidlx_io_TxtIStream);

extern void
impl_sidlx_io_TxtIStream__dtor(
  sidlx_io_TxtIStream);

/*
 * User-defined object methods
 */

extern void
impl_sidlx_io_TxtIStream_setFD(
  sidlx_io_TxtIStream,
  int32_t);

extern sidl_bool
impl_sidlx_io_TxtIStream_atEnd(
  sidlx_io_TxtIStream);

extern int32_t
impl_sidlx_io_TxtIStream_read(
  sidlx_io_TxtIStream,
  int32_t,
  struct sidl_char__array**,
  sidl_BaseInterface*);

extern int32_t
impl_sidlx_io_TxtIStream_readline(
  sidlx_io_TxtIStream,
  struct sidl_char__array**,
  sidl_BaseInterface*);

#ifdef __cplusplus
}
#endif
#endif
