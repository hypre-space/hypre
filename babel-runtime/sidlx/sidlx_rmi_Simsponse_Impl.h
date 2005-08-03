/*
 * File:          sidlx_rmi_Simsponse_Impl.h
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Server-side implementation for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.8
 */

#ifndef included_sidlx_rmi_Simsponse_Impl_h
#define included_sidlx_rmi_Simsponse_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifndef included_sidlx_rmi_Simsponse_h
#include "sidlx_rmi_Simsponse.h"
#endif
#ifndef included_sidl_io_IOException_h
#include "sidl_io_IOException.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidlx_rmi_Socket_h
#include "sidlx_rmi_Socket.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseException_h
#include "sidl_BaseException.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 50 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._includes) */
/* insert implementation here: sidlx.rmi.Simsponse._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._includes) */
#line 54 "sidlx_rmi_Simsponse_Impl.h"

/*
 * Private data for class sidlx.rmi.Simsponse
 */

struct sidlx_rmi_Simsponse__data {
#line 59 "../../../babel/runtime/sidlx/sidlx_rmi_Simsponse_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simsponse._data) */
  /* insert implementation here: sidlx.rmi.Simsponse._data (private data members) */
  /*  int d_len; */
  /* char *d_buf; */
  struct sidl_char__array *d_carray;
  sidlx_rmi_Socket d_sock;
  char *d_methodName;
  char *d_className;
  char *d_objectID;
  int d_current;
  sidl_BaseException d_exception;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simsponse._data) */
#line 74 "sidlx_rmi_Simsponse_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_Simsponse__data*
sidlx_rmi_Simsponse__get_data(
  sidlx_rmi_Simsponse);

extern void
sidlx_rmi_Simsponse__set_data(
  sidlx_rmi_Simsponse,
  struct sidlx_rmi_Simsponse__data*);

extern
void
impl_sidlx_rmi_Simsponse__load(
  void);

extern
void
impl_sidlx_rmi_Simsponse__ctor(
  /* in */ sidlx_rmi_Simsponse self);

extern
void
impl_sidlx_rmi_Simsponse__dtor(
  /* in */ sidlx_rmi_Simsponse self);

/*
 * User-defined object methods
 */

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_Simsponse_init(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getMethodName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getClassName(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simsponse_getObjectID(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackBool(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackChar(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackInt(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackLong(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackFloat(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackDouble(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackFcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackDcomplex(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simsponse_unpackString(
  /* in */ sidlx_rmi_Simsponse self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_BaseException
impl_sidlx_rmi_Simsponse_getExceptionThrown(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_Simsponse_done(
  /* in */ sidlx_rmi_Simsponse self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_Simsponse__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Simsponse(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Simsponse(struct 
  sidlx_rmi_Simsponse__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseException__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseException(struct 
  sidl_BaseException__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simsponse_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simsponse_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
