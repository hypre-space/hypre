/*
 * File:          sidlx_io_IOException_Impl.h
 * Symbol:        sidlx.io.IOException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.IOException
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

#ifndef included_sidlx_io_IOException_Impl_h
#define included_sidlx_io_IOException_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidlx_io_IOException_h
#include "sidlx_io_IOException.h"
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

extern void
impl_sidlx_io_IOException__ctor(
  sidlx_io_IOException);

extern void
impl_sidlx_io_IOException__dtor(
  sidlx_io_IOException);

/*
 * User-defined object methods
 */

#ifdef __cplusplus
}
#endif
#endif
