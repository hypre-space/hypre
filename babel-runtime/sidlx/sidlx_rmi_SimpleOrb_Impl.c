/*
 * File:          sidlx_rmi_SimpleOrb_Impl.c
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Server-side implementation for sidlx.rmi.SimpleOrb
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
 * Symbol "sidlx.rmi.SimpleOrb" (version 0.1)
 * 
 * An incomplete crack at a an orb
 */

#include "sidlx_rmi_SimpleOrb_Impl.h"

#line 26 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._includes) */
#include "sidlx_common.h"
#include "sidl_Loader.h"
#include "sidlx_rmi_SimCall.h"
#include "sidlx_rmi_SimReturn.h"
#include "sidl_String.h"
#include "sidl_io_Deserializer.h"
#include "sidl_io_Serializer.h"
#include "sidl_rmi_InstanceRegistry.h"
/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._includes) */
#line 37 "sidlx_rmi_SimpleOrb_Impl.c"

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleOrb__load(
  void)
{
#line 51 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._load) */
#line 57 "sidlx_rmi_SimpleOrb_Impl.c"
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleOrb__ctor(
  /* in */ sidlx_rmi_SimpleOrb self)
{
#line 69 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._ctor) */
#line 77 "sidlx_rmi_SimpleOrb_Impl.c"
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleOrb__dtor(
  /* in */ sidlx_rmi_SimpleOrb self)
{
#line 88 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._dtor) */
#line 98 "sidlx_rmi_SimpleOrb_Impl.c"
}

/*
 * Method:  serviceRequest[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb_serviceRequest"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleOrb_serviceRequest(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ sidlx_rmi_Socket sock,
  /* out */ sidl_BaseInterface *_ex)
{
#line 109 "../../../babel/runtime/sidlx/sidlx_rmi_SimpleOrb_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.serviceRequest) */
  sidlx_rmi_SimCall call = sidlx_rmi_SimCall__create();
  sidlx_rmi_SimReturn ret = sidlx_rmi_SimReturn__create();
  enum sidlx_rmi_CallType__enum ct;
  char * objid = NULL;
  char * methodName = NULL;
  char * className = NULL;

  sidlx_rmi_Socket_addRef(sock);
  sidlx_rmi_SimCall_init(call, sock, _ex);SIDL_CHECK(*_ex);
  ct = sidlx_rmi_SimCall_getCallType(call, _ex);SIDL_CHECK(*_ex);
  
  if(ct == sidlx_rmi_CallType_CREATE) {
    char * clsName = sidlx_rmi_SimCall_getClassName(call,_ex);
    sidl_DLL dll = sidl_Loader_findLibrary(clsName, "ior/impl", sidl_Scope_SCLSCOPE, sidl_Resolve_SCLRESOLVE);
    sidl_BaseClass h = sidl_DLL_createClass(dll, sidlx_rmi_SimCall_getClassName(call,_ex));
    char* ih = sidl_rmi_InstanceRegistry_registerInstance(h, _ex); SIDL_CHECK(*_ex);
    
    
    sidlx_rmi_SimReturn_init(ret,sidlx_rmi_SimCall_getMethodName(call,_ex), sidlx_rmi_SimCall_getClassName(call,_ex), ih, sock, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_SimReturn_SendReturn(ret, _ex); SIDL_CHECK(*_ex);
    
    sidlx_rmi_SimCall_deleteRef(call);
    sidlx_rmi_SimReturn_deleteRef(ret);

  } else if(ct == sidlx_rmi_CallType_EXEC) {
    sidl_BaseClass h = NULL;
    sidl_io_Deserializer inArgs= NULL;
    sidl_io_Serializer outArgs = NULL;
    objid = sidlx_rmi_SimCall_getObjectID(call,_ex); SIDL_CHECK(*_ex);
    methodName = sidlx_rmi_SimCall_getMethodName(call,_ex); SIDL_CHECK(*_ex);
    className = sidlx_rmi_SimCall_getClassName(call,_ex); SIDL_CHECK(*_ex);
    h = sidl_rmi_InstanceRegistry_getInstance(objid, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_SimReturn_init(ret,methodName, className, objid, sock, _ex); SIDL_CHECK(*_ex);
    inArgs=sidl_io_Deserializer__cast(call);
    outArgs=sidl_io_Serializer__cast(ret);
    
    sidl_BaseClass__exec(h, methodName, inArgs, outArgs);
    
    sidlx_rmi_SimReturn_SendReturn(ret, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_SimCall_deleteRef(call);
    sidlx_rmi_SimReturn_deleteRef(ret);
    /* FIXME: shouldn't h, inArgs, and outArgs all be deleteRef'ed */
  } else if(ct == sidlx_rmi_CallType_CONNECT) {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimpleOrb.serviceRequest:Connect not yet allowed!"); 
  } else {
    SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "SimpleOrb.serviceRequest:Unknown CallType!"); 
  }
  return; /* FIXME: shouldn't this pass through to memory cleanup? */
 EXIT:
  if (objid) free(objid);
  if (methodName) free(methodName);
  if (className) free(className);
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.serviceRequest) */
#line 173 "sidlx_rmi_SimpleOrb_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleOrb__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleOrb(struct 
  sidlx_rmi_SimpleOrb__object* obj) {
  return sidlx_rmi_SimpleOrb__getURL(obj);
}
struct sidl_SIDLException__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_SIDLException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_SIDLException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_SIDLException(struct 
  sidl_SIDLException__object* obj) {
  return sidl_SIDLException__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_Socket(struct 
  sidlx_rmi_Socket__object* obj) {
  return sidlx_rmi_Socket__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleServer__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidlx_rmi_SimpleServer(struct 
  sidlx_rmi_SimpleServer__object* obj) {
  return sidlx_rmi_SimpleServer__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_SimpleOrb_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
