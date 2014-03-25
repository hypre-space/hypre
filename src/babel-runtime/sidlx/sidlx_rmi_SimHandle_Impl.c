/*
 * File:          sidlx_rmi_SimHandle_Impl.c
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Server-side implementation for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimHandle" (version 0.1)
 * 
 * implementation of InstanceHandle using the Simhandle Protocol (written by Jim)
 */

#include "sidlx_rmi_SimHandle_Impl.h"
#include "sidl_NotImplementedException.h"
#include "sidl_Exception.h"

/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._includes) */
#include "sidl_String.h"
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "sidlx_common.h"
#include "sidl_rmi_MalformedURLException.h"
/* This class, SimHandle, implements InstanceHandle, the starting
   point for Babel RMI Protocols.  It should be pointed out to anyone
   planning to make a Babel protocol, that most of this is
   implementation dependent.  The InstanceHandle's purpose is pretty
   much just to return serializers and deserializers for the Babel
   client side.  These are where the real work is done with
   communicating over the network.  In Simple Protocol's case, we open
   a connection to the server, serialize everything into a buffer,then
   push the buffer over the wire, then we wait to recieve the
   response, which is copied into a buffer for deserialization.  After
   we get the response we close the connection.  The user deserializes
   the buffer at his leisure. It would probably be more efficent to
   use streams or something.  Protocol writers can do this, the Babel
   interface is flexible enough to handle just about anything.  So, in
   looking at this protocol for protocol writing guidence, don't worry
   too much about the implementation here, just the basics of what the
   functions do, not how.
*/

/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._includes) */

#define SIDL_IOR_MAJOR_VERSION 1
#define SIDL_IOR_MINOR_VERSION 0
/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimHandle__load(
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._load) */
  /* insert implementation here: sidlx.rmi.SimHandle._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._load) */
  }
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimHandle__ctor(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._ctor) */
  struct sidlx_rmi_SimHandle__data *dptr = 
    malloc(sizeof(struct sidlx_rmi_SimHandle__data));
  sidlx_rmi_SimHandle__set_data(self, dptr);
  /* initialize data */
  dptr->d_prefix=NULL;
  dptr->d_server=NULL;
  dptr->d_port=-1;
  dptr->d_objectID=NULL;
  dptr->d_typeName=NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._ctor) */
  }
}

/*
 * Special Class constructor called when the user wants to wrap his own private data.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle__ctor2"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimHandle__ctor2(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ void* private_data,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._ctor2) */
  /* Insert-Code-Here {sidlx.rmi.SimHandle._ctor2} (special constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._ctor2) */
  }
}
/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidlx_rmi_SimHandle__dtor(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._dtor) */
  struct sidlx_rmi_SimHandle__data *dptr = NULL;

  sidlx_rmi_SimHandle_close(self,_ex); SIDL_CHECK(*_ex);
  dptr = sidlx_rmi_SimHandle__get_data(self);

  if(dptr) {
    if (dptr->d_prefix) sidl_String_free(dptr->d_prefix);
    if (dptr->d_server) sidl_String_free(dptr->d_server);
    if (dptr->d_objectID) sidl_String_free(dptr->d_objectID);
    if (dptr->d_typeName) sidl_String_free(dptr->d_typeName);
    free(dptr);
    sidlx_rmi_SimHandle__set_data(self,NULL);
  }
 EXIT:
  return;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._dtor) */
  }
}

/*
 *  initialize a connection (intended for use by the
 * ProtocolFactory, (see above).  This should parse the url and
 * do everything necessary to create the remote object.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_initCreate"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimHandle_initCreate(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.initCreate) */
  /* This function creates a remote object on the server named in the url.*/
  sidl_BaseInterface _throwaway_exception = NULL;
  sidl_BaseException _be = NULL;

  struct sidlx_rmi_SimHandle__data *dptr = NULL;
  sidlx_rmi_ClientSocket connSock = NULL;
  sidlx_rmi_Socket locSock = NULL;
  sidlx_rmi_Simsponse resp = NULL;
  int lower, upper;
  int32_t clsLen, port;
  char* str = NULL;
  struct sidl_char__array * carray= NULL;
  char* prefix = NULL;
  char* server = NULL;
  char* objectID = NULL;

  dptr=sidlx_rmi_SimHandle__get_data(self);
  if (!dptr) {
        SIDL_THROW(*_ex, sidl_rmi_NetworkException, "ERROR: simhandle was not ctor'd\n");   

  }
  sidlx_parseURL(url, &prefix, &server, &port, NULL, &objectID, _ex);SIDL_CHECK(*_ex);

  if(!prefix || !server || !port || objectID) {
    SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "ERROR: malformed URL\n");
  }
  
  /*
   * Here where we make a connection to the ORB
   */
  connSock = sidlx_rmi_ClientSocket__create(_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_ClientSocket_init(connSock, sidl_String_strdup(server), port,_ex);SIDL_CHECK(*_ex);
  locSock = sidlx_rmi_Socket__cast(connSock,_ex); SIDL_CHECK(*_ex);
  
  /*
   * Connected to orb, create typename
   * ( The request format looks like "CREATE:"typename"" ) 
   */
  lower = 0;
  upper = 11+sidl_String_strlen(typeName); /*remember +1 char for the NULL char.*/
  carray = sidl_char__array_createRow(1, &lower, &upper);
  str = sidl_char__array_first(carray);
  sidl_String_strcpy(str, "CREATE:");
  clsLen = sidl_String_strlen(typeName);  /* serialize the typename as a
                                             string. */
  clsLen = htonl(clsLen);
  memmove(str+7, (char*)(&clsLen),4);
  sidl_String_strcpy(str+11, typeName);

  /* Send it over the network*/
  sidlx_rmi_Socket_writestring(locSock,upper+1,carray, _ex); SIDL_CHECK(*_ex);

  resp = sidlx_rmi_Simsponse__create(_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_Simsponse_init(resp, "CREATE", NULL, locSock, _ex);SIDL_CHECK(*_ex);
  /*  sidlx_rmi_Simsponse_test(resp,-1,-1,_ex);SIDL_CHECK(*_ex);*/
  sidlx_rmi_Simsponse_pullData(resp, _ex); SIDL_CHECK(*_ex);

  _be = sidlx_rmi_Simsponse_getExceptionThrown(resp, _ex);SIDL_CHECK(*_ex);
  if(_be != NULL) {
    sidl_BaseException_addLine(_be, "Exception unserialized from remote create call.", 
			       &_throwaway_exception);
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
							    &_throwaway_exception);
    goto EXIT;
  }

  /* Don't set this stuff until we're sure the object actually was created*/
  dptr->d_prefix = prefix;
  dptr->d_server = server;
  dptr->d_port = port;
  dptr->d_typeName = sidl_String_strdup(typeName);
  dptr->d_objectID = sidlx_rmi_Simsponse_getObjectID(resp, _ex); SIDL_CHECK(*_ex);

  sidl_char__array_deleteRef(carray);
  sidlx_rmi_Simsponse_deleteRef(resp,_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_ClientSocket_deleteRef(connSock,_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_Socket_deleteRef(locSock,_ex); SIDL_CHECK(*_ex);
  return 1; /*true*/
 EXIT:
  if(carray) { sidl_char__array_deleteRef(carray); }
  if(resp) { sidlx_rmi_Simsponse_deleteRef(resp,&_throwaway_exception);}
  if(locSock) { sidlx_rmi_Socket_deleteRef(locSock,&_throwaway_exception);}
  if(connSock){ sidlx_rmi_ClientSocket_deleteRef(connSock, &_throwaway_exception);}
  return 0; /*false*/
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.initCreate) */
  }
}

/*
 * initialize a connection (intended for use by the ProtocolFactory) 
 * This should parse the url and do everything necessary to connect 
 * to a remote object.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_initConnect"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimHandle_initConnect(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* in */ sidl_bool ar,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.initConnect) */
  /* Alerts the ORB to the additional remote connection to this object.
   * In Simple Protocol, this really just means that it calls remote addRef
   */
  sidl_BaseInterface _throwaway_exception = NULL;
  sidlx_rmi_ClientSocket connSock = NULL;
  sidlx_rmi_Socket locSock = NULL;
  /*sidlx_rmi_Simsponse resp = NULL;*/
  sidl_rmi_Response resp = NULL;
  sidlx_rmi_Simvocation obj = NULL; 
  char* prefix = NULL;
  char* server = NULL;
  int32_t port = 0;
  /*char* typename = NULL;*/
  char* objectID = NULL;
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);

  sidlx_parseURL(url, &prefix, &server, &port, NULL, &objectID, _ex);SIDL_CHECK(*_ex);

  if(!prefix || !server || !port || !objectID) {
    SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "ERROR: malformed URL\n");
  }
  dptr->d_prefix = prefix;
  dptr->d_server = server;
  dptr->d_port = port;
  dptr->d_objectID = objectID;
  dptr->d_typeName = NULL; /*sidl_String_strdup(typename);*/

  /* In implicit connection cases, we don't want to addRef the remote object.
   * if ar is false, do not addRef.  If it's true, addRef
   */
  if(ar) {
    obj = sidlx_rmi_Simvocation__create(_ex); SIDL_CHECK(*_ex);
    /* Open a connection*/
    connSock = sidlx_rmi_ClientSocket__create(_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ClientSocket_init(connSock, dptr->d_server, dptr->d_port,_ex);SIDL_CHECK(*_ex);
    locSock = sidlx_rmi_Socket__cast(connSock, _ex); SIDL_CHECK(*_ex);
    
    /* Call addRef */    
    sidlx_rmi_Simvocation_init(obj, "addRef", /*dptr->d_typeName,*/dptr->d_objectID, 
			       locSock, _ex); SIDL_CHECK(*_ex);
    resp = sidlx_rmi_Simvocation_invokeMethod(obj, _ex); SIDL_CHECK(*_ex);
    
    /* Return*/    
    sidl_rmi_Response_deleteRef(resp,_ex); SIDL_CHECK(*_ex);
    resp = NULL;
    sidlx_rmi_Simvocation_deleteRef(obj,_ex); SIDL_CHECK(*_ex);
    obj = NULL;
    sidlx_rmi_Socket_deleteRef(locSock,_ex);SIDL_CHECK(*_ex);
    locSock = NULL;
    sidlx_rmi_ClientSocket_deleteRef(connSock, _ex); SIDL_CHECK(*_ex);
    connSock = NULL;
  }
  return 1;
 EXIT:
  if(locSock) { sidlx_rmi_Socket_deleteRef(locSock,&_throwaway_exception); }
  if(connSock) { sidlx_rmi_ClientSocket_deleteRef(connSock,&_throwaway_exception); }
  if(resp) { sidl_rmi_Response_deleteRef(resp,&_throwaway_exception); }
  if(obj) { sidlx_rmi_Simvocation_deleteRef(obj,&_throwaway_exception);}
  return 0; /*false*/
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.initConnect) */
  }
}

/*
 *  Get a connection specifically for the purpose for requesting a 
 * serialization of a remote object (intended for use by the
 * ProtocolFactory, (see above).  This should parse the url and
 * request the object.  It should return a deserializer..
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_initUnserialize"

#ifdef __cplusplus
extern "C"
#endif
sidl_io_Serializable
impl_sidlx_rmi_SimHandle_initUnserialize(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.initUnserialize) */
  *_ex = 0;
#line 150 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  sidl_BaseInterface _throwaway_exception = NULL;
  sidl_BaseException _be = NULL;

  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);

  sidlx_rmi_ClientSocket connSock = NULL;
  sidlx_rmi_Socket locSock = NULL;
  sidlx_rmi_Simsponse resp = NULL;
  int lower, upper;
  int32_t objLen, port;
  char* str = NULL;
  struct sidl_char__array * carray= NULL;
  char* prefix = NULL;
  char* server = NULL;
  char* objectID = NULL;
  sidl_io_Serializable ser = NULL;
  if (!dptr) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "ERROR: simhandle was not ctor'd\n");   
  }
  sidlx_parseURL(url, &prefix, &server, &port, NULL, &objectID, _ex);SIDL_CHECK(*_ex);

  if(!prefix || !server || !port || !objectID) {
    SIDL_THROW(*_ex, sidl_rmi_MalformedURLException, "ERROR: malformed URL\n");
  }
  
  dptr->d_prefix = prefix;
  dptr->d_server = server;
  dptr->d_port = port;
  dptr->d_typeName = NULL;
  dptr->d_objectID = objectID;
  /*dptr->d_sock = NULL;*/

  /*
   * Here where we make a connection to the ORB
   */
  connSock = sidlx_rmi_ClientSocket__create(_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_ClientSocket_init(connSock, sidl_String_strdup(server), port,_ex);SIDL_CHECK(*_ex);
  locSock = sidlx_rmi_Socket__cast(connSock,_ex); SIDL_CHECK(*_ex);
  
  /*
   * Connected to orb, create typename
   * ( The request format looks like "CREATE:"typename"" ) 
   */
  lower = 0;
  upper = 11+sidl_String_strlen(objectID); /*remember +1 char for the NULL char.*/
  carray = sidl_char__array_createRow(1, &lower, &upper);
  str = sidl_char__array_first(carray);
  sidl_String_strcpy(str, "SERIAL:");
  objLen = sidl_String_strlen(objectID);  /* serialize the typename as a
                                             string. */
  objLen = htonl(objLen);
  memmove(str+7, (char*)(&objLen),4);
  sidl_String_strcpy(str+11, objectID);

  /* Send it over the network*/
  sidlx_rmi_Socket_writestring(locSock,upper+1,carray, _ex); SIDL_CHECK(*_ex);

  resp = sidlx_rmi_Simsponse__create(_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_Simsponse_init(resp, "SERIAL", objectID, locSock, _ex);
  SIDL_CHECK(*_ex);
  sidlx_rmi_Simsponse_pullData(resp, _ex); SIDL_CHECK(*_ex);

  _be = sidlx_rmi_Simsponse_getExceptionThrown(resp, _ex);SIDL_CHECK(*_ex);
  if(_be != NULL) {
    sidl_BaseException_addLine(_be, "Exception unserialized from remote create call.", 
			       &_throwaway_exception);
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
							    &_throwaway_exception);
    goto EXIT;
  }

  /*dptr->d_objectID = sidlx_rmi_Simsponse_getObjectID(resp, _ex); SIDL_CHECK(*_ex);*/
  sidlx_rmi_Simsponse_unpackSerializable(resp, NULL, &ser, _ex); SIDL_CHECK(*_ex);

  sidl_char__array_deleteRef(carray);
  sidlx_rmi_Simsponse_deleteRef(resp,_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_ClientSocket_deleteRef(connSock,_ex); SIDL_CHECK(*_ex);
  sidlx_rmi_Socket_deleteRef(locSock,_ex); SIDL_CHECK(*_ex);
  return ser; /*true*/
 EXIT:
  if(carray) { sidl_char__array_deleteRef(carray); }
  if(resp) { sidlx_rmi_Simsponse_deleteRef(resp,&_throwaway_exception);}
  if(locSock) { sidlx_rmi_Socket_deleteRef(locSock,&_throwaway_exception);}
  if(connSock){ sidlx_rmi_ClientSocket_deleteRef(connSock, &_throwaway_exception);}
  if(ser) {sidl_io_Serializable_deleteRef(ser, &_throwaway_exception);}
  return 0; /*false*/

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.initUnserialize) */
  }
}

/*
 *  return the short name of the protocol 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_getProtocol"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimHandle_getProtocol(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.getProtocol) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);

  if (dptr) {
    return sidl_String_strdup(dptr->d_prefix);;
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.getProtocol) */
  }
}

/*
 *  return the object ID for the remote object
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_getObjectID"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimHandle_getObjectID(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.getObjectID) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);

  if (dptr) {
    return sidl_String_strdup(dptr->d_objectID);
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.getObjectID) */
  }
}

/*
 *  
 * return the full URL for this object, takes the form: 
 * protocol://serviceID/objectID (where serviceID would = server:port 
 * on TCP/IP)
 * So usually, like this: protocol://server:port/objectID
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_getObjectURL"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimHandle_getObjectURL(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.getObjectURL) */
  int len = 0;
  char * url = NULL;
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);

  if (dptr) {
    if(dptr->d_port > 65536) {
      SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
		 "Simhandle.getURL: port number is too large!");
    }
    len = sidl_String_strlen(dptr->d_prefix) + 3 + 
      sidl_String_strlen(dptr->d_server) + 1 + 6 /*maximum port length*/  
      + 1 + sidl_String_strlen(dptr->d_objectID) + 1; 
    /* FORMAT: prefix://server:port/typename/objectID*/
    
    url = sidl_String_alloc(len);
    sprintf(url, "%s://%s:%d/%s",dptr->d_prefix, 
	    dptr->d_server, dptr->d_port, dptr->d_objectID);
    
    return url;
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;
  
  
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.getObjectURL) */
  }
}

/*
 *  create a serializer handle to invoke the named method 
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_createInvocation"

#ifdef __cplusplus
extern "C"
#endif
sidl_rmi_Invocation
impl_sidlx_rmi_SimHandle_createInvocation(
  /* in */ sidlx_rmi_SimHandle self,
  /* in */ const char* methodName,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.createInvocation) */
  /* This function prepares one to make function calls over the network*/
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);

  if (dptr) {
    sidl_rmi_Invocation ret = NULL;
    sidlx_rmi_Simvocation obj = sidlx_rmi_Simvocation__create(_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ClientSocket connSock = NULL;
    sidlx_rmi_Socket locSock= NULL;
    /*
     * TODO: THIS IS A BAD PLACE TO MAKE THE CONNECTION, IF SERIALIZATION FAILS, IT'S
     * DIFFICULT TO TELL THE SERVER THAT!  (WE need some way to tell the server 
     * serialization failed I guess.)
     *
     * Here where we make a connection to the ORB
     */
    connSock = sidlx_rmi_ClientSocket__create(_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ClientSocket_init(connSock, dptr->d_server, dptr->d_port,_ex);SIDL_CHECK(*_ex);
    locSock = sidlx_rmi_Socket__cast(connSock, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_Simvocation_init(obj, methodName, /*dptr->d_typeName,*/dptr->d_objectID, 
			       locSock, _ex); SIDL_CHECK(*_ex);
    ret = sidl_rmi_Invocation__cast(obj, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_Simvocation_deleteRef(obj, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_Socket_deleteRef(locSock, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ClientSocket_deleteRef(connSock, _ex); SIDL_CHECK(*_ex);
    return ret;
  }
  SIDL_THROW(*_ex, sidl_rmi_NetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.createInvocation) */
  }
}

/*
 *  
 * closes the connection (called by the destructor, if not done
 * explicitly) returns true if successful, false otherwise
 * (including subsequent calls)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_close"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidlx_rmi_SimHandle_close(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
  *_ex = 0;
  {
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.close) */
  sidl_BaseInterface _throwaway_exception = NULL;
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);


  /* Make sure that dptr exists, and the object has been init'd*/
  if(dptr && dptr->d_server) {
    /* Remote deleteRef the object we were connected to */
    sidlx_rmi_Simvocation obj = NULL; 
    sidlx_rmi_Socket locSock = NULL;
    sidlx_rmi_ClientSocket connSock = NULL;
    sidl_rmi_Response resp = NULL;

    obj = sidlx_rmi_Simvocation__create(_ex); SIDL_CHECK(*_ex);
    connSock = sidlx_rmi_ClientSocket__create(_ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ClientSocket_init(connSock, dptr->d_server, dptr->d_port,_ex);SIDL_CHECK(*_ex);
    locSock = sidlx_rmi_Socket__cast(connSock, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_Simvocation_init(obj, "deleteRef", /*dptr->d_typeName,*/dptr->d_objectID, 
			       locSock, _ex); SIDL_CHECK(*_ex);
    resp = sidlx_rmi_Simvocation_invokeMethod(obj, _ex); SIDL_CHECK(*_ex);
    
    sidl_rmi_Response_deleteRef(resp, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_Simvocation_deleteRef(obj, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_Socket_deleteRef(locSock, _ex); SIDL_CHECK(*_ex);
    sidlx_rmi_ClientSocket_deleteRef(connSock, _ex); SIDL_CHECK(*_ex);
 
    return 1; /*true*/

 EXIT:
    if(resp) { sidl_rmi_Response_deleteRef(resp, &_throwaway_exception);}
    if(obj) { sidlx_rmi_Simvocation_deleteRef(obj, &_throwaway_exception);}
    if(locSock) { sidlx_rmi_Socket_deleteRef(locSock, &_throwaway_exception);}
    if(connSock) { sidlx_rmi_ClientSocket_deleteRef(connSock, &_throwaway_exception); }
    return 0; /*false*/
    
  }  
  return 0; /*false*/

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.close) */
  }
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_BaseClass__object* impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connectI(url, ar, _ex);
}
struct sidl_BaseClass__object* impl_sidlx_rmi_SimHandle_fcast_sidl_BaseClass(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_BaseClass__cast(bi, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connectI(url, ar, _ex);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_BaseInterface(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_BaseInterface__cast(bi, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(
  const char* url, sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connectI(url, ar, _ex);
}
struct sidl_ClassInfo__object* impl_sidlx_rmi_SimHandle_fcast_sidl_ClassInfo(
  void* bi, sidl_BaseInterface* _ex) {
  return sidl_ClassInfo__cast(bi, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_RuntimeException(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_RuntimeException__connectI(url, ar, _ex);
}
struct sidl_RuntimeException__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_RuntimeException(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_RuntimeException__cast(bi, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_io_Serializable(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_io_Serializable__connectI(url, ar, _ex);
}
struct sidl_io_Serializable__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_io_Serializable(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_io_Serializable__cast(bi, _ex);
}
struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_InstanceHandle__connectI(url, ar, _ex);
}
struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_InstanceHandle(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_rmi_InstanceHandle__cast(bi, _ex);
}
struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidl_rmi_Invocation__connectI(url, ar, _ex);
}
struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidl_rmi_Invocation(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidl_rmi_Invocation__cast(bi, _ex);
}
struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(const char* url, 
  sidl_bool ar, sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimHandle__connectI(url, ar, _ex);
}
struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fcast_sidlx_rmi_SimHandle(void* bi, 
  sidl_BaseInterface* _ex) {
  return sidlx_rmi_SimHandle__cast(bi, _ex);
}
