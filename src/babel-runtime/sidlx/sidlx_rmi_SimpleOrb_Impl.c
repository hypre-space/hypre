/*
 * File:          sidlx_rmi_SimpleOrb_Impl.c
 * Symbol:        sidlx.rmi.SimpleOrb-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.SimpleOrb
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimpleOrb" (version 0.1)
 * 
 * A simple example orb, using the simhandle protocol (written by Jim)
 */

#include "sidlx_rmi_SimpleOrb_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._includes) */
#include "sidlx_common.h"
#include "sidl_String.h"
#include "sidl_Loader.h"
#include "sidlx_rmi_SimCall.h"
#include "sidlx_rmi_SimReturn.h"
#include "sidl_rmi_Call.h"
#include "sidl_rmi_Return.h"
#include "sidl_String.h"
#include "sidl_io_Deserializer.h"
#include "sidl_io_Serializer.h"
#include "sidl_rmi_InstanceRegistry.h"
#include "sidl_rmi_NetworkException.h"
#include "sidl_rmi_ObjectDoesNotExistException.h"
#include "stdio.h"
#include "sidlx_rmi_Common.h"
#include "sidl_exec_err.h"

#ifdef HAVE_PTHREAD
#include <pthread.h>
static pthread_mutex_t                s_log_mutex; /*lock for the exception log*/
#endif /* HAVE_PTHREAD */



/* logs an exception that could not be thrown back to the caller.*/
void log_exception(sidlx_rmi_SimpleOrb self, sidl_BaseInterface exception) { 
  
  sidl_BaseInterface throwaway_exception = NULL;
  struct sidlx_rmi_SimpleOrb__data *dptr = NULL;
  sidl_io_Serializable exp = NULL;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_log_mutex));
#endif /* HAVE_PTHREAD */
  
  dptr = sidlx_rmi_SimpleOrb__get_data(self);
  if(!dptr) {
    dptr= malloc(sizeof(struct sidlx_rmi_SimpleOrb__data));
    if(!dptr) { goto EXIT; }
    dptr->d_exceptions = sidl_io_Serializable__array_create1d(4);
    dptr->d_used = 0;
    sidlx_rmi_SimpleOrb__set_data(self, dptr);
  }
  /*If our array of exceptions is full, expand it.*/
  if(sidl_io_Serializable__array_length(dptr->d_exceptions,0) == dptr->d_used) {
    struct sidl_io_Serializable__array* new_array = sidl_io_Serializable__array_create1d(dptr->d_used*2);
    sidl_io_Serializable__array_copy(dptr->d_exceptions, new_array);
    sidl_io_Serializable__array_deleteRef(dptr->d_exceptions);
    dptr->d_exceptions = new_array;
  }
  exp = sidl_io_Serializable__cast(exception, &throwaway_exception); SIDL_CHECK(throwaway_exception);
  sidl_io_Serializable__array_set1(dptr->d_exceptions, dptr->d_used, exp);
  sidl_io_Serializable_deleteRef(exp, &throwaway_exception); SIDL_CHECK(throwaway_exception);
  ++(dptr->d_used);
 EXIT:
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_log_mutex));
#endif /* HAVE_PTHREAD */
  return;
}

/* clears the exception logs*/
void clear_exception_log(sidlx_rmi_SimpleOrb self) { 
  
  struct sidlx_rmi_SimpleOrb__data *dptr = NULL;
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_log_mutex));
#endif /* HAVE_PTHREAD */

  dptr = sidlx_rmi_SimpleOrb__get_data(self);
  if(dptr) {
    struct sidl_io_Serializable__array* new_array = 
      sidl_io_Serializable__array_create1d(sidl_io_Serializable__array_length(dptr->d_exceptions,0));
    sidl_io_Serializable__array_deleteRef(dptr->d_exceptions); 
    dptr->d_exceptions = new_array;
    dptr->d_used = 0;
  }

#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_log_mutex));
#endif /* HAVE_PTHREAD */
}

/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
static const struct sidlx_rmi_SimpleServer__epv* superEPV = NULL;

void sidlx_rmi_SimpleOrb__superEPV(
struct sidlx_rmi_SimpleServer__epv* parentEPV){
  superEPV = parentEPV;
}
/*
 * Get the full URL for exporting objects
 */

static char*
super_getServerURL(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex)
{
  return (*superEPV->f_getServerURL)((struct sidlx_rmi_SimpleServer__object*)
    self,
    objID,
    _ex);
}

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
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._load) */
  /* Insert the implementation of the static class initializer method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._load) */
  }
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
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimpleOrb__ctor2(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.SimpleOrb._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._ctor2) */
  }
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
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb._dtor) */
  }
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
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.serviceRequest) */

  /** Remember that everything in this ORB is totally implementation dependent,
   *  EXCEPT dynamic loading to objects to create and calling exec to call
   *  methods by name.  You don't want a calltype enumeration? Fine.  You don't
   *  like the SimCall and SimReturn object? Cool. You think typename should be 
   *  passed with the method call?  Right on.
   */

  sidl_BaseInterface _throwaway_exception = NULL;
  sidl_BaseInterface _ex2 = NULL;
  sidl_BaseException _SIDLex = NULL;

  enum sidlx_rmi_CallType__enum ct;
  char * objid = NULL;
  char * methodName = NULL;
  char * className = NULL;

  sidl_BaseClass h = NULL;
  sidl_io_Serializable ser = NULL;
  sidlx_rmi_SimCall call = NULL;
  sidlx_rmi_SimReturn ret = NULL;

  sidl_rmi_Call inArgs= NULL;
  sidl_rmi_Return outArgs = NULL;

  sidl_DLL dll = NULL;


  call = sidlx_rmi_SimCall__create(_ex); SIDL_CHECK(*_ex);
  ret = sidlx_rmi_SimReturn__create(_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_SimCall_init(call, sock, _ex);SIDL_CHECK(*_ex);
  ct = sidlx_rmi_SimCall_getCallType(call, _ex);SIDL_CHECK(*_ex);
  
  if(ct == sidlx_rmi_CallType_CREATE) {

    /* In order to create an object, first get the class name from the deserializer.
     * Then dynamically load the library and create it. Then register it. Return */
    sidlx_rmi_SimCall_unpackString(call, "className", &className,_ex);SIDL_CHECK(*_ex);
    methodName = sidlx_rmi_SimCall_getMethodName(call,_ex); SIDL_CHECK(*_ex);

    dll = sidl_Loader_findLibrary(className, 
				  "ior/impl", 
				  sidl_Scope_SCLSCOPE,
				  sidl_Resolve_SCLRESOLVE, _ex); SIDL_CHECK(*_ex);
    if(dll == NULL) {
      char ex_msg[1024];
      
      sidlx_rmi_SimReturn_init(ret, methodName, 0, sock, _ex); SIDL_CHECK(*_ex);
      
      sprintf(ex_msg,
	      "SimpleOrb: Unable to load DLL for class %s. check SIDL_DLL_PATH.",
	      className);
      SIDL_THROW(*_ex, sidl_rmi_ObjectDoesNotExistException,
		 ex_msg);
    }

    h = sidl_DLL_createClass(dll, className, _ex); SIDL_CHECK(*_ex);
    objid = sidl_rmi_InstanceRegistry_registerInstance(h, _ex); SIDL_CHECK(*_ex);
    sidl_BaseClass_addRef(h, _ex); SIDL_CHECK(*_ex);
    
    sidlx_rmi_SimReturn_init(ret,sidlx_rmi_SimCall_getMethodName(call,_ex), 
			     objid, sock, _ex); SIDL_CHECK(*_ex);

    sidlx_rmi_SimReturn_SendReturn(ret, _ex); SIDL_CHECK(*_ex);


  } else if (ct == sidlx_rmi_CallType_EXEC){ /* then (ct == sidlx_rmi_CallType_EXEC)*/
    /* This is a method call, so get the instance from the InstanceRegistry, create a
     * serializer, up cast the serializer and deserializer, and call the method with 
     * exec. */
    
    objid = sidlx_rmi_SimCall_getObjectID(call,_ex); SIDL_CHECK(*_ex);
    methodName = sidlx_rmi_SimCall_getMethodName(call,_ex); SIDL_CHECK(*_ex);

    h = sidl_rmi_InstanceRegistry_getInstanceByString(objid, _ex); 
    SIDL_CHECK(*_ex);
    sidlx_rmi_SimReturn_init(ret,methodName, objid, sock, _ex); SIDL_CHECK(*_ex);
    if(h ==NULL) {
      SIDL_THROW(*_ex, sidl_rmi_ObjectDoesNotExistException, "SimpleOrb: Bad ObjectID, no such object.");
    }
    inArgs=sidl_rmi_Call__cast(call,_ex); SIDL_CHECK(*_ex);
    outArgs=sidl_rmi_Return__cast(ret,_ex); SIDL_CHECK(*_ex);

    sidl_BaseClass__exec(h, methodName, inArgs, outArgs,_ex); SIDL_CHECK(*_ex);

 
    sidlx_rmi_SimReturn_SendReturn(ret, _ex); SIDL_CHECK(*_ex);

  } else if (ct == sidlx_rmi_CallType_SERIAL) { /* then (ct == sidlx_rmi_CallType_SERIAL)*/

    /* In order to serialize an object, first get the objectID from the deserializer.
     * Then get the object from the InstanceRegistry. Serialize it back */
    sidlx_rmi_SimCall_unpackString(call, "objid", &objid,_ex);SIDL_CHECK(*_ex);
 
    sidlx_rmi_SimReturn_init(ret,sidlx_rmi_SimCall_getMethodName(call,_ex), 
			     objid, sock, _ex); SIDL_CHECK(*_ex);
    h = sidl_rmi_InstanceRegistry_getInstanceByString(objid, _ex); 
    SIDL_CHECK(*_ex);
    if(h ==NULL) {
      SIDL_THROW(*_ex, sidl_rmi_ObjectDoesNotExistException, "SimpleOrb: Bad ObjectID, no such object.");
    }
    ser = sidl_io_Serializable__cast(h, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_SimReturn_packSerializable(ret, NULL, ser, _ex); SIDL_CHECK(*_ex);
    
    /* Here we say there was no exception thrown.  In normal calls, this is done automatically*/ 

    /*   sidlx_rmi_SimReturn_packBool(ret, "ex_thrown", FALSE, _ex);SIDL_CHECK(*_ex);*/

    sidlx_rmi_SimReturn_SendReturn(ret, _ex); SIDL_CHECK(*_ex);

  
  } else { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "SimCall.init:Improperly formed call!");
  }

  goto CLEANUP;

 EXIT:
  /*There's nowhere above this to throw the exception, so we have to either
   *throw it to the caller or log it*/
  /* Try to throw it again*/
  {
    _SIDLex = sidl_BaseException__cast(*_ex,&_ex2); EXEC_CHECK(_ex2);
    if(ret) {
      sidlx_rmi_SimReturn_throwException(ret, _SIDLex, &_ex2); EXEC_CHECK(_ex2);
      sidlx_rmi_SimReturn_SendReturn(ret, &_ex2); SIDL_CHECK(_ex2);
      sidl_BaseException_deleteRef(_SIDLex, &_ex2); EXEC_CHECK(_ex2);
      sidl_BaseInterface_deleteRef(*_ex, &_ex2); EXEC_CHECK(_ex2);
      *_ex = NULL;
      goto CLEANUP;
    }
  }
  
 EXEC_ERR:
  /*Having failed to get the message back to the caller, log the error.*/
  log_exception(self, *_ex);
  log_exception(self, _ex2);
  
 CLEANUP:
  if (*_ex) {sidl_BaseInterface_deleteRef(*_ex,&_throwaway_exception); *_ex=NULL;}
  if (_ex2) {sidl_BaseInterface_deleteRef(_ex2,&_throwaway_exception); _ex2=NULL;}
  if (objid) { free(objid); objid=NULL;}
  if (methodName) { free(methodName); methodName=NULL;}
  if (className) { free(className); className=NULL;}
  if (call) {sidlx_rmi_SimCall_deleteRef(call,&_throwaway_exception);call=NULL;}
  if (ret) {sidlx_rmi_SimReturn_deleteRef(ret,&_throwaway_exception);ret=NULL;}
  if (h) {sidl_BaseClass_deleteRef(h, &_throwaway_exception); h=NULL;}
  if (inArgs) {sidl_rmi_Call_deleteRef(inArgs, &_throwaway_exception); inArgs=NULL;}
  if (outArgs) {sidl_rmi_Return_deleteRef(outArgs, &_throwaway_exception); outArgs=NULL;}
  if (dll) {sidl_DLL_deleteRef(dll, &_throwaway_exception); dll=NULL;}
  /*  if (ser) {sidl_io_Serializable_deleteRef(ser, &_throwaway_exception); ser=NULL;}*/
  return;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.serviceRequest) */
  }
}

/*
 * Get the full URL for exporting objects
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb_getServerURL"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimpleOrb_getServerURL(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ const char* objID,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.getServerURL) */
  int len = 0;
  char * protocol = NULL;
  char * hostname = NULL;
  int32_t port = 0;
  char * url = NULL;
  protocol = sidlx_rmi_SimpleOrb_getProtocol(self, _ex);  SIDL_CHECK(*_ex);
  hostname = sidlx_rmi_SimpleOrb_getServerName(self, _ex);  SIDL_CHECK(*_ex);
  port = sidlx_rmi_SimpleOrb_getPort(self, _ex);  SIDL_CHECK(*_ex);

  /*FORMAT:   protocol:\\hostname:port/objID*/
  len = sidl_String_strlen(protocol) + 3 + sidl_String_strlen(hostname) + 1 +
    6+1+sidl_String_strlen(objID)+1;
  url = sidl_String_alloc(len);
  sprintf(url, "%s://%s:%d/%s",protocol, 
	  hostname, port, objID);
  sidl_String_free(protocol);
  sidl_String_free(hostname);

  return url;

 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.getServerURL) */
  }
}

/*
 * Method:  isLocalObject[]
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb_isLocalObject"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimpleOrb_isLocalObject(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.isLocalObject) */
  char* ret = NULL;
  char* protocol = NULL;
  char* server = NULL;
  int32_t port = 0;
  /*char* typename = NULL;*/
  char* objectID = NULL;
  char* myurl = NULL;
  char* localhost = NULL;
  int32_t localport = 0; 
  int32_t remoteIP = 0;
  int32_t localIP = 0; 
  /* I hope someone has a smarter way to do this. */

  myurl = sidl_String_strdup(url);
  sidlx_parseURL(myurl, &protocol, &server, &port, NULL, &objectID, _ex);SIDL_CHECK(*_ex);

  localhost =  sidlx_rmi_SimpleOrb_getServerName(self, _ex);SIDL_CHECK(*_ex);
  localport = sidlx_rmi_SimpleOrb_getPort(self, _ex);SIDL_CHECK(*_ex);
  
  localIP = sidlx_rmi_Common_getHostIP(localhost, _ex);  SIDL_CHECK(*_ex);
  remoteIP = sidlx_rmi_Common_getHostIP(server, _ex);SIDL_CHECK(*_ex);

  /*if the remoteIP is a loopback address, it is the localIP (127.*.*.* is loopback)*/
  if((remoteIP >> 24) == 127) {
    remoteIP = localIP;
  }

  if((remoteIP == localIP) && (localport == port)) {
    ret = objectID;
  }
 EXIT:
  if(myurl) {free(myurl);}
  if(protocol) {free(protocol);}
  if(server) {free(server);}
  if(localhost) {free(localhost);}
  if(!ret && objectID) {free(objectID);} /*If we aren't returning objectID, free it*/
  return ret;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.isLocalObject) */
  }
}

/*
 * Get the short name of the protocol this ORB supports
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb_getProtocol"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimpleOrb_getProtocol(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.getProtocol) */
  return sidl_String_strdup("simhandle");
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.getProtocol) */
  }
}

/*
 * This gets an array of logged exceptions.  If an exception can
 * not be thrown back to the caller, we log it with the Server.  This 
 * gets the array of all those exceptions.
 * THIS IS SOMETHING OF A TEST! THIS MAY CHANGE!
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimpleOrb_getExceptions"

#ifdef __cplusplus
extern "C"
#endif
struct sidl_io_Serializable__array*
impl_sidlx_rmi_SimpleOrb_getExceptions(
  /* in */ sidlx_rmi_SimpleOrb self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimpleOrb.getExceptions) */
#ifdef HAVE_PTHREAD
  (void)pthread_mutex_lock(&(s_log_mutex));
#endif /* HAVE_PTHREAD */

  /** I don't use smartcopy here because I specifically don't want the server's
   *  array to be edited outside of the mutex blocks
   */
  struct sidl_io_Serializable__array* ret = NULL;
  struct sidlx_rmi_SimpleOrb__data *dptr = sidlx_rmi_SimpleOrb__get_data(self);
  if(dptr) {
    ret = sidl_io_Serializable__array_create1d(sidl_io_Serializable__array_length(dptr->d_exceptions,0));
    sidl_io_Serializable__array_copy(dptr->d_exceptions, ret);
  }

#ifdef HAVE_PTHREAD
  (void)pthread_mutex_unlock(&(s_log_mutex));
#endif /* HAVE_PTHREAD */
  return ret;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimpleOrb.getExceptions) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimpleOrb_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimpleOrb_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_io_Serializable(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidl_rmi_ServerInfo(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_ServerInfo__connectI(url, ar, _ex);
}
struct sidl_rmi_ServerInfo__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidl_rmi_ServerInfo(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_rmi_ServerInfo__cast(bi, _ex);
}
struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleOrb(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleOrb__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimpleOrb__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleOrb(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimpleOrb__cast(bi, _ex);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_SimpleServer(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimpleServer__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimpleServer__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_SimpleServer(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimpleServer__cast(bi, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fconnect_sidlx_rmi_Socket(const char* url, sidl_bool 
  ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_Socket__connectI(url, ar, _ex);
}
struct sidlx_rmi_Socket__object* 
  impl_sidlx_rmi_SimpleOrb_fcast_sidlx_rmi_Socket(void* bi, sidl_BaseInterface* 
  _ex) {
  return sidlx_rmi_Socket__cast(bi, _ex);
}
