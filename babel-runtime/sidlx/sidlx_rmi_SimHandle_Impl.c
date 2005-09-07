/*
 * File:          sidlx_rmi_SimHandle_Impl.c
 * Symbol:        sidlx.rmi.SimHandle-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.10
 * Description:   Server-side implementation for sidlx.rmi.SimHandle
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.10
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidlx.rmi.SimHandle" (version 0.1)
 * 
 * implementation of InstanceHandle using the Simocol (simple-protocol), 
 * 	contains all the serialization code
 */

#include "sidlx_rmi_SimHandle_Impl.h"

#line 27 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
/* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._includes) */
#include "sidl_String.h"

/* This function parses a url into the pointers provided (they are all out parameters)
   url, protocol, and server are required, and the method will throw an if they are
   null.  port, className, and objectID are optional, and may be passed in as NULL
*/ 
static void parseURL(char* url, char** protocol, char** server, int* port, 
		     char** className, char** objectID, sidl_BaseInterface *_ex) {

  int i = 0, start=0;
  int end = strlen(url);
  char tmp = '\0';

  if(url == NULL || protocol == NULL || server == NULL) {
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl_rmi_ProtocolFactory.praseURL: Required arg is NULL\n");
  }

  /* extract the protocol name */
  while ((i<end) && (url[i]!=':')) { 
    i++;
  }
  if ( (i==start) || (i==end) ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "ERROR: malformed URL\n");
  }
  
  url[i]=0;
  if(protocol != NULL) {
    *protocol=sidl_String_strdup(url);
  }
  url[i]=':';

  /* skip colons & slashes (should be "://") */
  if ( ((i+3)>=end) || (url[i]!=':') || (url[i+1]!='/') || (url[i+2]!='/')) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "ERROR: malformed URL\n");
  } else { 
    i+=3;
  }
  /* extract server name */
  start=i;
  while ( (i<end) && url[i]!=':'&& url[i]!='/') { 
    i++;
  }

  if (i==start) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, "ERROR: invalid URL format\n");
  }
  tmp=url[i];
  url[i]=0;
  if(server != NULL) {
    *server = sidl_String_strdup(url + start);
  }
  url[i]=tmp;

  /* extract port number (if it exists ) */
  if ( (i<end) && (url[i]==':')) {
    ++i;
    start=i;
    while ((i<end) && (url[i] != '/')) { 
      if ( (url[i]<'0') || url[i]>'9') { 
	SIDL_THROW(*_ex, sidl_rmi_NetworkException, "ERROR: invalid URL format\n");
      }
      i++;
    }
    tmp = url[i];
    url[i]=0;
    if(port!=NULL) {
      *port = atoi( url+start );
    }
    url[i]=tmp;
  }

  

  /* Continue onward to extract the classname, if it exists*/
  if ( (i<end) && (url[i]=='/')) {
    ++i;
    start=i;
    while ((i<end) && (url[i] != '/')) { 
      i++;
    }
    tmp = url[i];
    url[i]=0;
    if(className!=NULL) {
      *className = sidl_String_strdup( url+start );
    }
    url[i]=tmp;
  } else {
    return;
  }

  /* Continue onward to extract the objectid, if it exists*/
  if ( (i<end) && (url[i]=='/')) {
    ++i;
    start=i;
    while ((i<end) && (url[i] != '/')) { 
      i++;
    }
    tmp = url[i];
    url[i]=0;
    if(objectID!=NULL) {
      *objectID = sidl_String_strdup( url+start );
    }
    url[i]=tmp;
  } else {
    return;
  }

 EXIT:
  return;

}


/* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._includes) */
#line 144 "sidlx_rmi_SimHandle_Impl.c"

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
  void)
{
#line 158 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._load) */
  /* insert implementation here: sidlx.rmi.SimHandle._load (static class initializer method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._load) */
#line 164 "sidlx_rmi_SimHandle_Impl.c"
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
  /* in */ sidlx_rmi_SimHandle self)
{
#line 176 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._ctor) */
  /* insert implementation here: sidlx.rmi.SimHandle._ctor (constructor method) */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._ctor) */
#line 184 "sidlx_rmi_SimHandle_Impl.c"
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
  /* in */ sidlx_rmi_SimHandle self)
{
#line 195 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle._dtor) */
  
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  if(dptr) {
    sidl_String_free(dptr->d_protocol);
    sidl_String_free(dptr->d_server);
    sidl_String_free(dptr->d_typeName);
    sidl_String_free(dptr->d_objectID);
    sidlx_rmi_SimHandle__set_data(self, NULL);
  }
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle._dtor) */
#line 214 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * initialize a connection (intended for use by the ProtocolFactory) 
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
#line 226 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.initCreate) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  sidlx_rmi_ClientSocket connSock = NULL;
  sidlx_rmi_Socket locSock = NULL;
  sidlx_rmi_Simsponse resp = NULL;
  int len, lower, upper;
  int32_t clsLen, port;
  char* str = NULL;
  struct sidl_char__array * carray= NULL;
  char* protocol = NULL;
  char* server = NULL;
  char* myurl = sidl_String_strdup( url );
  if (!dptr) {
    dptr = malloc(sizeof(struct sidlx_rmi_SimHandle__data));
  }
  parseURL(myurl, &protocol, &server, &port, NULL, NULL, _ex);

  dptr->d_protocol = sidl_String_strdup(protocol);
  dptr->d_server = sidl_String_strdup(server);
  dptr->d_port = port;
  dptr->d_typeName = sidl_String_strdup(typeName);
  dptr->d_objectID = NULL;
  /*dptr->d_sock = NULL;*/
  sidlx_rmi_SimHandle__set_data(self, dptr);

  /*
   * Here where we make a connection to the ORB
   */
  connSock = sidlx_rmi_ClientSocket__create();
  sidlx_rmi_ClientSocket_init(connSock, sidl_String_strdup(server), port,_ex);SIDL_CHECK(*_ex);
  locSock = sidlx_rmi_Socket__cast(connSock);
  /*dptr->d_sock = sidlx_rmi_Socket__cast(connSock);*/
  
  /*
   * Connected to orb, create typename
   */
  lower = 0;
  upper = 11+sidl_String_strlen(typeName); /*1 for the NULL char.*/
  carray = sidl_char__array_createRow(1, &lower, &upper);
  str = sidl_char__array_first(carray);
  sidl_String_strcpy(str, "CREATE:");
  clsLen = sidl_String_strlen(typeName);  /* serialize the typename as a
                                             string. */
  clsLen = htonl(clsLen);
  memmove(str+7, (char*)(&clsLen),4);
  sidl_String_strcpy(str+11, typeName);

  sidlx_rmi_Socket_writestring(locSock,upper+1,carray, _ex); SIDL_CHECK(*_ex);

  resp = sidlx_rmi_Simsponse__create();
  sidlx_rmi_Simsponse_init(resp, "CREATE", sidl_String_strdup(typeName), NULL, locSock, _ex);
  SIDL_CHECK(*_ex);
  dptr->d_objectID = sidlx_rmi_Simsponse_getObjectID(resp, _ex); SIDL_CHECK(*_ex);
  /*sidlx_rmi_Simsponse_unpackString(resp, "return", &(dptr->d_objectid), _ex);SIDL_CHECK(*_ex);*/

  sidl_char__array_deleteRef(carray);
  sidlx_rmi_Simsponse_deleteRef(resp);
  sidlx_rmi_ClientSocket_deleteRef(connSock);
  if(protocol) {sidl_String_free(protocol); }
  if(server) {sidl_String_free(server); }
  if(myurl) {sidl_String_free(myurl); }
  return 1; /*true*/
 EXIT:
  if(carray) { sidl_char__array_deleteRef(carray); }
  if(resp) { sidlx_rmi_Simsponse_deleteRef(resp); }
  if(connSock){ sidlx_rmi_ClientSocket_deleteRef(connSock); }
  if(protocol) {sidl_String_free(protocol); }
  if(server) {sidl_String_free(server); }
  if(myurl) {sidl_String_free(myurl); }
  return 0; /*false*/

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.initCreate) */
#line 308 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * initialize a connection (intended for use by the ProtocolFactory) 
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
  /* out */ sidl_BaseInterface *_ex)
{
#line 317 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.initConnect) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  sidlx_rmi_ClientSocket connSock = NULL;
  sidlx_rmi_Socket locSock = NULL;
  /*sidlx_rmi_Simsponse resp = NULL;*/
  sidl_rmi_Response resp = NULL;
  sidlx_rmi_Simvocation obj = sidlx_rmi_Simvocation__create();
  char* protocol = NULL;
  char* server = NULL;
  int32_t port = 0;
  char* typename = NULL;
  char* objectID = NULL;
  char* myurl = sidl_String_strdup(url);
  if (!dptr) {
    dptr = malloc(sizeof(struct sidlx_rmi_SimHandle__data));
  }
  parseURL(myurl, &protocol, &server, &port, &typename, &objectID, _ex);
  dptr->d_protocol = sidl_String_strdup(protocol);
  dptr->d_server = sidl_String_strdup(server);
  dptr->d_port = port;
  dptr->d_typeName = sidl_String_strdup(typename);
  dptr->d_objectID = sidl_String_strdup(objectID);
  sidlx_rmi_SimHandle__set_data(self, dptr);

  connSock = sidlx_rmi_ClientSocket__create();
  sidlx_rmi_ClientSocket_init(connSock, dptr->d_server, dptr->d_port,_ex);SIDL_CHECK(*_ex);
  locSock = sidlx_rmi_Socket__cast(connSock);

  sidlx_rmi_Simvocation_init(obj, "addRef", dptr->d_typeName,dptr->d_objectID, 
			     locSock, _ex); SIDL_CHECK(*_ex);
  resp = sidlx_rmi_Simvocation_invokeMethod(obj, _ex); SIDL_CHECK(*_ex);
  /* Do something here to check for exceptions and pass them on*/

  sidl_rmi_Response_deleteRef(resp);
  sidlx_rmi_Simvocation_deleteRef(obj);
  sidlx_rmi_ClientSocket_deleteRef(connSock);
  if(protocol) {sidl_String_free(protocol); }
  if(server) {sidl_String_free(server); }
  if(myurl) {sidl_String_free(myurl); }
  if(typename) {sidl_String_free(typename); }
  if(objectID) {sidl_String_free(objectID); }
  return 1;
 EXIT:
  if(connSock) { sidlx_rmi_ClientSocket_deleteRef(connSock); }
  if(resp) { sidl_rmi_Response_deleteRef(resp); }
  if(obj) { sidlx_rmi_Simvocation_deleteRef(obj); }
  if(protocol) {sidl_String_free(protocol); }
  if(server) {sidl_String_free(server); }
  if(myurl) {sidl_String_free(myurl); }
  if(typename) {sidl_String_free(typename); }
  if(objectID) {sidl_String_free(objectID); }
  return 0; /*false*/
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.initConnect) */
#line 382 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * return the name of the protocol 
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
#line 388 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.getProtocol) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  if (dptr) {
    return sidl_String_strdup(dptr->d_protocol);;
  }
  SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.getProtocol) */
#line 411 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * return the session ID 
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
#line 415 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.getObjectID) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  if (dptr) {
    return sidl_String_strdup(dptr->d_objectID);
  }
  SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.getObjectID) */
#line 440 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * return the full URL for this object, takes the form: 
 *  protocol://server:port/class/objectID
 */

#undef __FUNC__
#define __FUNC__ "impl_sidlx_rmi_SimHandle_getURL"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidlx_rmi_SimHandle_getURL(
  /* in */ sidlx_rmi_SimHandle self,
  /* out */ sidl_BaseInterface *_ex)
{
#line 443 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.getURL) */
  int len = 0;
  char * url = NULL;
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  if (dptr) {
    if(dptr->d_port > 65536) {
      SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, 
		 "Simhandle.getURL: port number is too large!");
    }
    len = sidl_String_strlen(dptr->d_protocol) + 3 + 
      sidl_String_strlen(dptr->d_server) + 1 + 6 /*maximum port length*/  
      + 1 + sidl_String_strlen(dptr->d_typeName) + 1 + 
      sidl_String_strlen(dptr->d_objectID) + 1; 
    /* FORMAT: protocol://server:port/typename/objectID*/
    
    url = sidl_String_alloc(len);
    sprintf(url, "%s://%s:%d/%s/%s",dptr->d_protocol, 
	    dptr->d_server, dptr->d_port, dptr->d_typeName, dptr->d_objectID);
    
    return url;
  }
  SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;
  

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.getURL) */
#line 488 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * create a handle to invoke a named method 
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
#line 489 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.createInvocation) */
  struct sidlx_rmi_SimHandle__data *dptr =
    sidlx_rmi_SimHandle__get_data(self);
  if (dptr) {
    sidl_rmi_Invocation ret = NULL;
    sidlx_rmi_Simvocation obj = sidlx_rmi_Simvocation__create();
    sidlx_rmi_ClientSocket connSock = NULL;
    sidlx_rmi_Socket locSock= NULL;
    /*
     * Here where we make a connection to the ORB
     */
    connSock = sidlx_rmi_ClientSocket__create();
    sidlx_rmi_ClientSocket_init(connSock, dptr->d_server, dptr->d_port,_ex);SIDL_CHECK(*_ex);
    locSock = sidlx_rmi_Socket__cast(connSock);
    sidlx_rmi_Simvocation_init(obj, methodName, dptr->d_typeName,dptr->d_objectID, 
			       locSock, _ex); SIDL_CHECK(*_ex);
    ret = sidl_rmi_Invocation__cast(obj);
    sidlx_rmi_ClientSocket_deleteRef(connSock);
    return ret;
  }
  SIDL_THROW(*_ex, sidlx_rmi_GenNetworkException, "Simhandle has not been initialized");
 EXIT:
  return NULL;

  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.createInvocation) */
#line 533 "sidlx_rmi_SimHandle_Impl.c"
}

/*
 * closes the connection (called be destructor, if not done explicitly) 
 * returns true if successful, false otherwise (including subsequent calls)
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
#line 532 "../../../babel/runtime/sidlx/sidlx_rmi_SimHandle_Impl.c"
  /* DO-NOT-DELETE splicer.begin(sidlx.rmi.SimHandle.close) */
  /*
   * TODO: In the future I think this will deleteref.
   */
  /* DO-NOT-DELETE splicer.end(sidlx.rmi.SimHandle.close) */
#line 558 "sidlx_rmi_SimHandle_Impl.c"
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_rmi_InstanceHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_InstanceHandle__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj) {
  return sidl_rmi_InstanceHandle__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_rmi_Invocation__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_Invocation(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_Invocation__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_Invocation(struct 
  sidl_rmi_Invocation__object* obj) {
  return sidl_rmi_Invocation__getURL(obj);
}
struct sidlx_rmi_SimHandle__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidlx_rmi_SimHandle(char* url,
  sidl_BaseInterface *_ex) {
  return sidlx_rmi_SimHandle__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidlx_rmi_SimHandle(struct 
  sidlx_rmi_SimHandle__object* obj) {
  return sidlx_rmi_SimHandle__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidlx_rmi_SimHandle_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidlx_rmi_SimHandle_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
