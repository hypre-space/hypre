/*
 * File:          sidlx_rmi_SimReturn_Impl.h
 * Symbol:        sidlx.rmi.SimReturn-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.SimReturn
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

#ifndef included_sidlx_rmi_SimReturn_Impl_h
#define included_sidlx_rmi_SimReturn_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidlx_rmi_SimReturn_h
#include "sidlx_rmi_SimReturn.h"
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

#line 44 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._includes) */
/* insert implementation here: sidlx.rmi.SimReturn._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._includes) */
#line 48 "sidlx_rmi_SimReturn_Impl.h"

/*
 * Private data for class sidlx.rmi.SimReturn
 */

struct sidlx_rmi_SimReturn__data {
#line 53 "../../../babel/runtime/sidlx/sidlx_rmi_SimReturn_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimReturn._data) */
  /* insert implementation here: sidlx.rmi.SimReturn._data (private data members) */
  int d_len;
  int d_capacity;
  char *d_buf;
  sidlx_rmi_Socket d_sock;
  char *d_methodName;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimReturn._data) */
#line 64 "sidlx_rmi_SimReturn_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimReturn__data*
sidlx_rmi_SimReturn__get_data(
  sidlx_rmi_SimReturn);

extern void
sidlx_rmi_SimReturn__set_data(
  sidlx_rmi_SimReturn,
  struct sidlx_rmi_SimReturn__data*);

extern
void
impl_sidlx_rmi_SimReturn__load(
  void);

extern
void
impl_sidlx_rmi_SimReturn__ctor(
  /* in */ sidlx_rmi_SimReturn self);

extern
void
impl_sidlx_rmi_SimReturn__dtor(
  /* in */ sidlx_rmi_SimReturn self);

/*
 * User-defined object methods
 */

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(struct 
  sidlx_rmi_SimReturn__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_SimReturn_init(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* methodName,
  /* in */ const char* className,
  /* in */ const char* objectid,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimReturn_getMethodName(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_SendReturn(
  /* in */ sidlx_rmi_SimReturn self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packBool(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ sidl_bool value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packChar(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ char value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packInt(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ int32_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packLong(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ int64_t value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packFloat(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ float value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packDouble(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ double value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packFcomplex(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ struct sidl_fcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packDcomplex(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ struct sidl_dcomplex value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimReturn_packString(
  /* in */ sidlx_rmi_SimReturn self,
  /* in */ const char* key,
  /* in */ const char* value,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidlx_rmi_SimReturn__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_SimReturn(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_SimReturn(struct 
  sidlx_rmi_SimReturn__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_io_Serializer__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_io_Serializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_io_Serializer(struct 
  sidl_io_Serializer__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimReturn_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimReturn_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
