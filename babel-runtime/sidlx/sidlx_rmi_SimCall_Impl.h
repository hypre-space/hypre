/*
 * File:          sidlx_rmi_SimCall_Impl.h
 * Symbol:        sidlx.rmi.SimCall-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Server-side implementation for sidlx.rmi.SimCall
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

#ifndef included_sidlx_rmi_SimCall_Impl_h
#define included_sidlx_rmi_SimCall_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_io_Deserializer_h
#include "sidl_io_Deserializer.h"
#endif
#ifndef included_sidlx_rmi_SimCall_h
#include "sidlx_rmi_SimCall.h"
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
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall._includes) */
/* insert implementation here: sidlx.rmi.SimCall._includes (include files) */
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall._includes) */

/*
 * Private data for class sidlx.rmi.SimCall
 */

struct sidlx_rmi_SimCall__data {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimCall._data) */
  /* insert implementation here: sidlx.rmi.SimCall._data (private data members) */
  struct sidl_char__array *d_carray;
  int d_current;  /*Index into d_carray data*/

  sidlx_rmi_Socket d_sock;
  char *d_methodName;
  char *d_clsid;
  char *d_objid;
  enum sidlx_rmi_CallType__enum d_calltype; /*EXEC, CREATE, or CONNECT*/
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimCall._data) */
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimCall__data*
sidlx_rmi_SimCall__get_data(
  sidlx_rmi_SimCall);

extern void
sidlx_rmi_SimCall__set_data(
  sidlx_rmi_SimCall,
  struct sidlx_rmi_SimCall__data*);

extern
void
impl_sidlx_rmi_SimCall__load(
  void);

extern
void
impl_sidlx_rmi_SimCall__ctor(
  /* in */ sidlx_rmi_SimCall self);

extern
void
impl_sidlx_rmi_SimCall__dtor(
  /* in */ sidlx_rmi_SimCall self);

/*
 * User-defined object methods
 */

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(struct 
  sidlx_rmi_SimCall__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
void
impl_sidlx_rmi_SimCall_init(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimCall_getMethodName(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimCall_getObjectID(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimCall_getClassName(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
enum sidlx_rmi_CallType__enum
impl_sidlx_rmi_SimCall_getCallType(
  /* in */ sidlx_rmi_SimCall self,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackBool(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ sidl_bool* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackChar(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ char* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackInt(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ int32_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackLong(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ int64_t* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackFloat(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ float* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackDouble(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ double* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackFcomplex(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ struct sidl_fcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackDcomplex(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ struct sidl_dcomplex* value,
  /* out */ sidl_BaseInterface *_ex);

extern
void
impl_sidlx_rmi_SimCall_unpackString(
  /* in */ sidlx_rmi_SimCall self,
  /* in */ const char* key,
  /* out */ char** value,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_io_Deserializer__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_Deserializer(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_Deserializer(struct 
  sidl_io_Deserializer__object* obj);
extern struct sidlx_rmi_SimCall__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_SimCall(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_SimCall(struct 
  sidlx_rmi_SimCall__object* obj);
extern struct sidl_io_IOException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_io_IOException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_io_IOException(struct 
  sidl_io_IOException__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimCall_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimCall_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
