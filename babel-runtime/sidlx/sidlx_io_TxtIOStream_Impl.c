/*
 * File:          sidlx_io_TxtIOStream_Impl.c
 * Symbol:        sidlx.io.TxtIOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.8.6
 * Description:   Server-side implementation for sidlx.io.TxtIOStream
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
 * Symbol "sidlx.io.TxtIOStream" (version 0.1)
 * 
 * Simple text-based output stream appends spaces
 */

#include "sidlx_io_TxtIOStream_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._includes) */

/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream__ctor"

void
impl_sidlx_io_TxtIOStream__ctor(
  sidlx_io_TxtIOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream__dtor"

void
impl_sidlx_io_TxtIOStream__dtor(
  sidlx_io_TxtIOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._dtor) */
}

/*
 * Method:  setFD[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_setFD"

void
impl_sidlx_io_TxtIOStream_setFD(
  sidlx_io_TxtIOStream self, int32_t fd)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.setFD) */
  /* Insert the implementation of the setFD method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.setFD) */
}

/*
 * returns true iff the stream is at its end, or closed 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_atEnd"

sidl_bool
impl_sidlx_io_TxtIOStream_atEnd(
  sidlx_io_TxtIOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.atEnd) */
  /* Insert the implementation of the atEnd method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.atEnd) */
}

/*
 * low level read an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_read"

int32_t
impl_sidlx_io_TxtIOStream_read(
  sidlx_io_TxtIOStream self, int32_t nbytes, struct sidl_char__array** data,
    sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.read) */
  /* Insert the implementation of the read method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.read) */
}

/*
 * low level read 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_readline"

int32_t
impl_sidlx_io_TxtIOStream_readline(
  sidlx_io_TxtIOStream self, struct sidl_char__array** data,
    sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.readline) */
  /* Insert the implementation of the readline method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.readline) */
}

/*
 * flushes the buffer, if any 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_flush"

void
impl_sidlx_io_TxtIOStream_flush(
  sidlx_io_TxtIOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.flush) */
  /* Insert the implementation of the flush method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.flush) */
}

/*
 * low level write for an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_write"

int32_t
impl_sidlx_io_TxtIOStream_write(
  sidlx_io_TxtIOStream self, struct sidl_char__array* data,
    sidl_BaseInterface* _ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.write) */
  /* Insert the implementation of the write method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.write) */
}
