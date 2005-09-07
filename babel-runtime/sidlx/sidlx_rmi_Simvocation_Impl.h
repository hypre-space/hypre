/*
 * File:          sidlx_rmi_Simvocation_Impl.h
 * Symbol:        sidlx.rmi.Simvocation-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.Simvocation
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_rmi_Simvocation_Impl_h
#define included_sidlx_rmi_Simvocation_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidlx_rmi_Simvocation_h
#include "sidlx_rmi_Simvocation.h"
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
#ifndef included_sidl_io_Serializer_h
#include "sidl_io_Serializer.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 50 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation._includes) */
/* insert implementation here: sidlx.rmi.Simvocation._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation._includes) */
#line 54 "sidlx_rmi_Simvocation_Impl.h"

/*
 * Private data for class sidlx.rmi.Simvocation
 */

struct sidlx_rmi_Simvocation__data {
#line 59 "../../../babel/runtime/sidlx/sidlx_rmi_Simvocation_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.Simvocation._data) */
  /* insert implementation here: sidlx.rmi.Simvocation._data (private data members) */
  int d_len;
  int d_capacity;
  char *d_buf;
  sidlx_rmi_Socket d_sock;
  char *d_methodName;
  char *d_className;
  char *d_objectID;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.Simvocation._data) */
#line 72 "sidlx_rmi_Simvocation_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_Simvocation__data*
sidlx_rmi_Simvocation__get_data(
  sidlx_rmi_Simvocation);

extern void
sidlx_rmi_Simvocation__set_data(
  sidlx_rmi_Simvocation,
  struct sidlx_rmi_Simvocation__data*);

extern
void
impl_sidlx_rmi_Simvocation__load(
  void);

extern
void
impl_sidlx_rmi_Simvocation__ctor(
  /* in */ sidlx_rmi_Simvocation self);

extern
void
impl_sidlx_rmi_Simvocation__dtor(
  /* in */ sidlx_rmi_Simvocation self);

/*
 * User-defined object methods
 */

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(struct 
  sidlx_rmi_Simvocation__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_Simvocation_init(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_Simvocation_getMethodName(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packBool(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packChar(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packInt(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packLong(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packFloat(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packDouble(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packFcomplex(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packDcomplex(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_Simvocation_packString(
  /* in */ sidlx_rmi_Simvocation self,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Response
impl_sidlx_rmi_Simvocation_invokeMethod(
  /* in */ sidlx_rmi_Simvocation self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_Response__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Response(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Response(struct 
  sidl_rmi_Response__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_Simvocation__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Simvocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Simvocation(struct 
  sidlx_rmi_Simvocation__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* 
  impl_sidlx_rmi_Simvocation_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_Simvocation_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_Simvocation_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
