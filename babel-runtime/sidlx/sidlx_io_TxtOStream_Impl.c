/*
 * File:          sidlx_io_TxtOStream_Impl.c
 * Symbol:        sidlx.io.TxtOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.io.TxtOStream
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
 * Symbol "sidlx.io.TxtOStream" (version 0.1)
 * 
 * Simple text-based output stream appends spaces
 */

#include "sidlx_io_TxtOStream_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream__ctor(
  /* in */ sidlx_io_TxtOStream self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream__dtor(
  /* in */ sidlx_io_TxtOStream self)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_setFD(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int32_t fd)
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

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_flush(
  /* in */ sidlx_io_TxtOStream self)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.flush) */

  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.flush) */
}

/*
 * low level write for an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_write"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_io_TxtOStream_write(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.write) */
  int n;
  int fd = sidlx_io_TxtOStream__get_data(self)->filedes;
  
  n = s_writen( fd, -1, data, _ex ); SIDL_CHECK(_ex);
 EXIT:
  return n;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.write) */
}

/*
 * Method:  put[Bool]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putBool(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putBool) */
  const char true = 'T';
  const char false = 'F';
  const char * p = (item==TRUE) ? &true : &false;
  const int fd = sidlx_io_TxtOStream__get_data(self)->filedes;
  int32_t k = s_writen2(fd, 1, p, _ex); SIDL_CHECK(*_ex);
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putBool) */
}

/*
 * Method:  put[Char]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putChar(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putChar) */
  const int fd = sidlx_io_TxtOStream__get_data(self)->filedes;
  int32_t k = s_writen2(fd, 1, &item, _ex); SIDL_CHECK(*_ex);
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putChar) */
}

/*
 * Method:  put[Int]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putInt(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putInt) */
  const int fd = sidlx_io_TxtOStream__get_data(self)->filedes;
  char buffer[5]; 
  int32_t k ;
  *((int32_t*)buffer) = item;
  /* should massage buffer to be portable */
  k=s_writen2(fd, 4, buffer, _ex); SIDL_CHECK(*_ex);
 EXIT:
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putInt) */
}

/*
 * Method:  put[Long]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putLong(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putLong) */
  /* Insert the implementation of the putLong method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putLong) */
}

/*
 * Method:  put[Float]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putFloat(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putFloat) */
  /* Insert the implementation of the putFloat method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putFloat) */
}

/*
 * Method:  put[Double]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putDouble(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putDouble) */
  /* Insert the implementation of the putDouble method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putDouble) */
}

/*
 * Method:  put[Fcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putFcomplex(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putFcomplex) */
  /* Insert the implementation of the putFcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putFcomplex) */
}

/*
 * Method:  put[Dcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putDcomplex(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putDcomplex) */
  /* Insert the implementation of the putDcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putDcomplex) */
}

/*
 * Method:  put[String]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtOStream_putString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtOStream_putString(
  /* in */ sidlx_io_TxtOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtOStream.putString) */
  /* Insert the implementation of the putString method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtOStream.putString) */
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_io_TxtOStream__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_TxtOStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_TxtOStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_TxtOStream(struct 
  sidlx_io_TxtOStream__object* obj) {
  return sidlx_io_TxtOStream__getURL(obj);
}
struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_IOException__connect(url, _ex);
}
char * impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) {
  return sidlx_io_IOException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_io_TxtOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_io_OStream__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_OStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj) {
  return sidlx_io_OStream__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_io_TxtOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
