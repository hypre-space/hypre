/*
 * File:          sidlx_io_TxtOStream_Impl.c
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.TxtOStream
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
 * Symbol "sidlx.io.TxtOStream" (version 0.1)
 * 
 * Simple text-based output stream appends spaces
 */

#include "sidlx_io_TxtOStream_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream__ctor"

void
impl_sidlx_io_TxtOStream__ctor(
  sidlx_io_TxtOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._ctor) */
  struct sidlx_io_TxtOStream__data *data = (struct sidlx_io_TxtOStream__data *)
    malloc( sizeof (  struct sidlx_io_TxtOStream__data) );
  data->filedes = -1;
  sidlx_io_TxtOStream__set_data(self,data);
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream__dtor"

void
impl_sidlx_io_TxtOStream__dtor(
  sidlx_io_TxtOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._dtor) */
  struct sidlx_io_TxtOStream__data *data = sidlx_io_TxtOStream__get_data(self);
  if (data) { 
    free((void*)data);
  }
  sidlx_io_TxtOStream__set_data(self,NULL);
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._dtor) */
}

/*
 * Method:  setFD[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_setFD"

void
impl_sidlx_io_TxtOStream_setFD(
  sidlx_io_TxtOStream self, int32_t fd)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.setFD) */
  /* Insert the implementation of the destructor method here... */
  sidlx_io_TxtOStream__get_data(self)->filedes=fd;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.setFD) */
}

/*
 * flushes the buffer, if any 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_flush"

void
impl_sidlx_io_TxtOStream_flush(
  sidlx_io_TxtOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.flush) */

  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.flush) */
}

/*
 * low level write for an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_write"

int32_t
impl_sidlx_io_TxtOStream_write(
  sidlx_io_TxtOStream self, struct sidl_char__array* data,
    sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.write) */
  int n;
  int fd;
  fd = sidlx_io_TxtOStream__get_data(self)->filedes;

  n = s_writen( fd, data, _ex ); SIDL_CHECK(_ex);
 EXIT:
  return n;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.write) */
}
