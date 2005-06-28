/*
 * File:          sidlx_io_TxtIStream_Impl.c
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
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
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream__ctor(
  /* in */ sidlx_io_TxtIStream self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream__dtor(
  /* in */ sidlx_io_TxtIStream self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_setFD(
  /* in */ sidlx_io_TxtIStream self,
  /* in */ int32_t fd)
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

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_io_TxtIStream_atEnd(
  /* in */ sidlx_io_TxtIStream self)
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

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_io_TxtIStream_read(
  /* in */ sidlx_io_TxtIStream self,
  /* in */ int32_t nbytes,
  /* out */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.read) */
  int fd = sidlx_io_TxtIStream__get_data(self)->filedes;
  int n = s_readn( fd, nbytes, data, _ex ); SIDL_CHECK(*_ex);
 EXIT:
  return n;

  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.read) */
}

/*
 * low level read 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_readline"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_io_TxtIStream_readline(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
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

/*
 * Method:  get[Bool]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getBool(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ sidl_bool* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getBool) */
  const int fd = sidlx_io_TxtIStream__get_data(self)->filedes;
  char buffer[2], *tmp = buffer;
  int32_t k = s_readn2( fd, 1, &tmp, _ex); SIDL_CHECK(*_ex);
  if ( k>=1 ) { 
    if (buffer[0]=='T') { 
      *item = TRUE;
    } else if ( buffer[0]=='F') { 
      *item = FALSE;
    } else { 
      /* format error */
      SIDL_THROW( *_ex, sidlx_io_IOException,"format error: expected 'T' or 'F'");
    }
  } else { 
    SIDL_THROW( *_ex, sidlx_io_IOException,"I/O error: insufficient data read");
  }
 
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getBool) */
}

/*
 * Method:  get[Char]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getChar(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ char* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getChar) */
  const int fd = sidlx_io_TxtIStream__get_data(self)->filedes;
  char buffer[2], *tmp = buffer;
  ssize_t k = s_readn2( fd, 1, &tmp, _ex); SIDL_CHECK(*_ex);
  if ( k>=1 ) { 
    *item = buffer[0];
  } else { 
    SIDL_THROW( *_ex, sidlx_io_IOException,"I/O error: insufficient data read");
  }
  
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getChar) */
}

/*
 * Method:  get[Int]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getInt(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ int32_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getInt) */
  const int fd = sidlx_io_TxtIStream__get_data(self)->filedes;
  char buffer[5], *tmp=buffer; /*int32_t is 4 bytes */
		    
  ssize_t k = s_readn2( fd, 4, &tmp, _ex); SIDL_CHECK(*_ex);
  /* should massage the buffer to be portable */

  if ( k>=1 ) { 
    *item = (int32_t) *buffer;
  } else { 
    SIDL_THROW( *_ex, sidlx_io_IOException,"I/O error: insufficient data read");
  }
    
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getInt) */
}

/*
 * Method:  get[Long]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getLong(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ int64_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getLong) */
  /* Insert the implementation of the getLong method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getLong) */
}

/*
 * Method:  get[Float]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getFloat(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ float* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getFloat) */
  /* Insert the implementation of the getFloat method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getFloat) */
}

/*
 * Method:  get[Double]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getDouble(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ double* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getDouble) */
  /* Insert the implementation of the getDouble method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getDouble) */
}

/*
 * Method:  get[Fcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getFcomplex(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_fcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getFcomplex) */
  /* Insert the implementation of the getFcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getFcomplex) */
}

/*
 * Method:  get[Dcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getDcomplex(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ struct sidl_dcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getDcomplex) */
  /* Insert the implementation of the getDcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getDcomplex) */
}

/*
 * Method:  get[String]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIStream_getString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIStream_getString(
  /* in */ sidlx_io_TxtIStream self,
  /* out */ char** item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIStream.getString) */
  /* Insert the implementation of the getString method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIStream.getString) */
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_IOException__connect(url, _ex);
}
char * impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) {
  return sidlx_io_IOException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_io_TxtIStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_io_TxtIStream__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_TxtIStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_TxtIStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_TxtIStream(struct 
  sidlx_io_TxtIStream__object* obj) {
  return sidlx_io_TxtIStream__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_io_TxtIStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
struct sidlx_io_IStream__object* 
  impl_sidlx_io_TxtIStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_IStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtIStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj) {
  return sidlx_io_IStream__getURL(obj);
}
