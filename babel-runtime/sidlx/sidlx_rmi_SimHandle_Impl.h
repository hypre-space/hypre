/*
 * File:          sidlx_rmi_SimHandle_Impl.h
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_rmi_SimHandle_Impl_h
#define included_sidlx_rmi_SimHandle_Impl_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidlx_rmi_SimHandle_h
#include "sidlx_rmi_SimHandle.h"
#endif
#ifndef included_sidl_rmi_NetworkException_h
#include "sidl_rmi_NetworkException.h"
#endif
#ifndef included_sidl_BaseInterface_h
#include "sidl_BaseInterface.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif

#line 41 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.h"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._includes) */
#include "sidlx_rmi_ClientSocket.h"
#include "sidlx_rmi_Socket.h"
#include "sidlx_rmi_GenNetworkException.h"
#include "sidlx_rmi_Simsponse.h"
#include "sidlx_rmi_Simvocation.h"
#include "sidl_Exception.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._includes) */
#line 50 "sidlx_rmi_SimHandle_Impl.h"

/*
 * Private data for class sidlx.rmi.SimHandle
 */

struct sidlx_rmi_SimHandle__data {
#line 55 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.h"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._data) */
  /* insert implementation here: sidlx.rmi.SimHandle._data (private data members) */
  char * d_protocol;
  char * d_server;
  int32_t d_port;
  char * d_typeName;
  char * d_objectID;
  /* Changed my mind.  A connection will be created whenever an invocation is made*/
  /*sidlx_rmi_Socket d_sock;   For now, I think I'll just keep the connection open, later*/
                        /* I should really close it between calls*/
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._data) */
#line 69 "sidlx_rmi_SimHandle_Impl.h"
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Access functions for class private data and built-in methods
 */

extern struct sidlx_rmi_SimHandle__data*
sidlx_rmi_SimHandle__get_data(
  sidlx_rmi_SimHandle);

extern void
sidlx_rmi_SimHandle__set_data(
  sidlx_rmi_SimHandle,
  struct sidlx_rmi_SimHandle__data*);

extern
void
impl_sidlx_rmi_SimHandle__load(
  void);

extern
void
impl_sidlx_rmi_SimHandle__ctor(
  /* in */ sidlx_rmi_SimHandle self);

extern
void
impl_sidlx_rmi_SimHandle__dtor(
  /* in */ sidlx_rmi_SimHandle self);

/*
 * User-defined object methods
 */

extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
extern
sidl_bool
impl_sidlx_rmi_SimHandle_initCreate(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimHandle_initConnect(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getProtocol(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getObjectID(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
char*
impl_sidlx_rmi_SimHandle_getURL(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_rmi_Invocation
impl_sidlx_rmi_SimHandle_createInvocation(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* methodName,
  /* out */ sidl_BaseInterface *_ex);

extern
sidl_bool
impl_sidlx_rmi_SimHandle_close(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex);

extern struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj);
extern struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj);
extern struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj);
extern struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj);
extern struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj);
extern struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj);
extern struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex);
extern char* impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj);
#ifdef __cplusplus
}
#endif
#endif
