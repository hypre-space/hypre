/*
 * File:          sidlx_io_TxtIStream_Impl.c
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.8.6
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.io.TxtIStream" (version 0.1)
 * 
 * Simple text-based input stream 
 * some datatypes (e.g. strings, arrays, etc require special formatting)
 * undefined behavior with non-whitespace separated fields.
 */

#include "sidlx_io_TxtIStream_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream__ctor"

void
impl_sidlx_io_TxtIStream__ctor(
  sidlx_io_TxtIStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._ctor) */
  struct sidlx_io_TxtIStream__data *data = (struct sidlx_io_TxtIStream__data *)
    malloc( sizeof (  struct sidlx_io_TxtIStream__data) );
  data->filedes = -1;
  sidlx_io_TxtIStream__set_data(self,data);
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream__dtor"

void
impl_sidlx_io_TxtIStream__dtor(
  sidlx_io_TxtIStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._dtor) */
  struct sidlx_io_TxtIStream__data *data = sidlx_io_TxtIStream__get_data(self);
  if (data) { 
    free((void*)data);
  }
  sidlx_io_TxtIStream__set_data(self,NULL);
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._dtor) */
}

/*
 * Method:  setFD[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_setFD"

void
impl_sidlx_io_TxtIStream_setFD(
  sidlx_io_TxtIStream self, int32_t fd)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.setFD) */
  sidlx_io_TxtIStream__get_data(self)->filedes=fd;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.setFD) */
}

/*
 * returns true iff the stream is at its end, or closed 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_atEnd"

sidl_bool
impl_sidlx_io_TxtIStream_atEnd(
  sidlx_io_TxtIStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.atEnd) */
  /* Insert the implementation of the atEnd method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.atEnd) */
}

/*
 * low level read an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_read"

int32_t
impl_sidlx_io_TxtIStream_read(
  sidlx_io_TxtIStream self, int32_t nbytes, struct sidl_char__array** data,
    sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.read) */
  int fd;
  int n;
  fd = sidlx_io_TxtIStream__get_data(self)->filedes;

  n = s_readn( fd, nbytes, data, _ex ); SIDL_CHECK(*_ex);
 EXIT:
  return n;

  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.read) */
}

/*
 * low level read 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_readline"

int32_t
impl_sidlx_io_TxtIStream_readline(
  sidlx_io_TxtIStream self, struct sidl_char__array** data,
    sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.readline) */
  int fd;
  int maxlen=4*1024;
  int n;
  fd = sidlx_io_TxtIStream__get_data(self)->filedes;
  
  n = s_readline( fd, maxlen, data, _ex ); SIDL_CHECK(*_ex);
 EXIT:
  return n;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.readline) */
}
