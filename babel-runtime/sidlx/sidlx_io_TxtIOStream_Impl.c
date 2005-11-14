/*
 * File:          sidlx_io_TxtIOStream_Impl.c
 * Symbol:        sidlx.io.TxtIOStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.io.TxtIOStream
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
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

#line 26 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._includes) */
#include "sidlx_common.h"
/* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._includes) */
#line 30 "sidlx_io_TxtIOStream_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream__load(
  void)
{
#line 44 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._load) */
#line 50 "sidlx_io_TxtIOStream_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream__ctor(
  /* in */ sidlx_io_TxtIOStream self)
{
#line 62 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._ctor) */
#line 70 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream__dtor(
  /* in */ sidlx_io_TxtIOStream self)
{
#line 81 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream._dtor) */
#line 91 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  setFD[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_setFD"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_setFD(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t fd)
{
#line 101 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.setFD) */
  /* Insert the implementation of the setFD method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.setFD) */
#line 113 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * returns true iff the stream is at its end, or closed 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_atEnd"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_io_TxtIOStream_atEnd(
  /* in */ sidlx_io_TxtIOStream self)
{
#line 120 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.atEnd) */
  /* Insert the implementation of the atEnd method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.atEnd) */
#line 134 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * low level read an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_read"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_io_TxtIOStream_read(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t nbytes,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 142 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.read) */
  /* Insert the implementation of the read method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.read) */
#line 158 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * low level read 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_readline"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_io_TxtIOStream_readline(
  /* in */ sidlx_io_TxtIOStream self,
  /* out array<char,row-major> */ struct sidl_char__array** data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 163 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.readline) */
  /* Insert the implementation of the readline method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.readline) */
#line 181 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Bool]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getBool(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ sidl_bool* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 184 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getBool) */
  /* Insert the implementation of the getBool method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getBool) */
#line 204 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Char]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getChar(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ char* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 205 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getChar) */
  /* Insert the implementation of the getChar method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getChar) */
#line 227 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Int]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getInt(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ int32_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 226 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getInt) */
  /* Insert the implementation of the getInt method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getInt) */
#line 250 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Long]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getLong(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ int64_t* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 247 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getLong) */
  /* Insert the implementation of the getLong method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getLong) */
#line 273 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Float]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getFloat(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ float* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 268 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getFloat) */
  /* Insert the implementation of the getFloat method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getFloat) */
#line 296 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Double]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getDouble(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ double* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 289 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getDouble) */
  /* Insert the implementation of the getDouble method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getDouble) */
#line 319 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Fcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getFcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_fcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 310 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getFcomplex) */
  /* Insert the implementation of the getFcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getFcomplex) */
#line 342 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[Dcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getDcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ struct sidl_dcomplex* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 331 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getDcomplex) */
  /* Insert the implementation of the getDcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getDcomplex) */
#line 365 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  get[String]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_getString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_getString(
  /* in */ sidlx_io_TxtIOStream self,
  /* out */ char** item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 352 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.getString) */
  /* Insert the implementation of the getString method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.getString) */
#line 388 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * flushes the buffer, if any 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_flush"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_flush(
  /* in */ sidlx_io_TxtIOStream self)
{
#line 371 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.flush) */
  /* Insert the implementation of the flush method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.flush) */
#line 409 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * low level write for an array of bytes 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_write"

#ifdef __cplusplus
extern "C"
#endif
int32_t
impl_sidlx_io_TxtIOStream_write(
  /* in */ sidlx_io_TxtIOStream self,
  /* in array<char,row-major> */ struct sidl_char__array* data,
  /* out */ sidl_BaseInterface *_ex)
{
#line 392 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.write) */
  /* Insert the implementation of the write method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.write) */
#line 432 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Bool]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putBool"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putBool(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ sidl_bool item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 413 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putBool) */
  /* Insert the implementation of the putBool method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putBool) */
#line 455 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Char]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putChar"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putChar(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ char item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 434 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putChar) */
  /* Insert the implementation of the putChar method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putChar) */
#line 478 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Int]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putInt"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putInt(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int32_t item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 455 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putInt) */
  /* Insert the implementation of the putInt method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putInt) */
#line 501 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Long]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putLong"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putLong(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ int64_t item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 476 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putLong) */
  /* Insert the implementation of the putLong method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putLong) */
#line 524 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Float]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putFloat"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putFloat(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ float item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 497 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putFloat) */
  /* Insert the implementation of the putFloat method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putFloat) */
#line 547 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Double]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putDouble"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putDouble(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ double item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 518 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putDouble) */
  /* Insert the implementation of the putDouble method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putDouble) */
#line 570 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Fcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putFcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putFcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_fcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 539 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putFcomplex) */
  /* Insert the implementation of the putFcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putFcomplex) */
#line 593 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[Dcomplex]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putDcomplex"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putDcomplex(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ struct sidl_dcomplex item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 560 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putDcomplex) */
  /* Insert the implementation of the putDcomplex method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putDcomplex) */
#line 616 "sidlx_io_TxtIOStream_Impl.c"
}

/*
 * Method:  put[String]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_io_TxtIOStream_putString"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_io_TxtIOStream_putString(
  /* in */ sidlx_io_TxtIOStream self,
  /* in */ const char* item,
  /* out */ sidl_BaseInterface *_ex)
{
#line 581 "../../../babel/runtime/sidlx/sidlx_io_TxtIOStream_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.io.TxtIOStream.putString) */
  /* Insert the implementation of the putString method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.io.TxtIOStream.putString) */
#line 639 "sidlx_io_TxtIOStream_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_io_IOException__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOException(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_IOException__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOException(struct 
  sidlx_io_IOException__object* obj) {
  return sidlx_io_IOException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_io_TxtIOStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_TxtIOStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_TxtIOStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_TxtIOStream(struct 
  sidlx_io_TxtIOStream__object* obj) {
  return sidlx_io_TxtIOStream__getURL(obj);
}
struct sidlx_io_OStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_OStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_OStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_OStream(struct 
  sidlx_io_OStream__object* obj) {
  return sidlx_io_OStream__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_io_IOStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IOStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_IOStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IOStream(struct 
  sidlx_io_IOStream__object* obj) {
  return sidlx_io_IOStream__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
struct sidlx_io_IStream__object* 
  impl_sidlx_io_TxtIOStream_fconnect_sidlx_io_IStream(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_io_IStream__connect(url, _ex);
}
char * impl_sidlx_io_TxtIOStream_fgetURL_sidlx_io_IStream(struct 
  sidlx_io_IStream__object* obj) {
  return sidlx_io_IStream__getURL(obj);
}
