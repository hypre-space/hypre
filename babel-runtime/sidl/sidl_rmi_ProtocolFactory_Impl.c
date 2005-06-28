/*
 * File:          sidl_rmi_ProtocolFactory_Impl.c
 * Symbol:        sidl.rmi.ProtocolFactory-v0.9.3
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Release:       $Name$
 * Revision:      @(#) $Id$
 * Description:   Server-side implementation for sidl.rmi.ProtocolFactory
 * 
 * Copyright (c) 2000-2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the Components Team <components@llnl.gov>
 * All rights reserved.
 * 
 * This file is part of Babel. For more information, see
 * http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
 * for Our Notice and the LICENSE file for the GNU Lesser General Public
 * License.
 * 
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License (as published by
 * the Free Software Foundation) version 2.1 dated February 1999.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
 * conditions of the GNU Lesser General Public License for more details.
 * 
 * You should have recieved a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * WARNING: Automatically generated; only changes within splicers preserved
 * 
 * babel-version = 0.10.4
 */

/*
 * DEVELOPERS ARE EXPECTED TO PROVIDE IMPLEMENTATIONS
 * FOR THE FOLLOWING METHODS BETWEEN SPLICER PAIRS.
 */

/*
 * Symbol "sidl.rmi.ProtocolFactory" (version 0.9.3)
 * 
 * This singleton class keeps a table of string prefixes (e.g. "babel" or "proteus")
 * to protocol implementations.  The intent is to parse a URL (e.g. "babel://server:port/class")
 * and create classes that implement <code>sidl.rmi.InstanceHandle</code>.
 */

#include "sidl_rmi_ProtocolFactory_Impl.h"

/* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory._includes) */
#include <string.h>
#include <stdio.h>
#include "sidl_String.h"
#include "sidl_DLL.h"
#include "sidl_Loader.h"
#include "sidl_BaseClass.h"
#include "sidl_rmi_InstanceHandle.h"
#include "sidl_Exception.h"
#include "sidl_rmi_NetworkException.h"
static char ** reg; /* reg[2*i] is associated w/ reg[2*i+1] */
static int len; /* really 2*len entries */
static int maxlen; /* 1/2 size of buffer malloc'd */

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

/* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory._includes) */

/*
 * Static class initializer called exactly once before any user-defined method is dispatched
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory__load"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ProtocolFactory__load(
  void)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory._load) */
  maxlen=1024;
  reg = (char**) malloc ( sizeof(char*)*maxlen*2 );
  len = 0;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory._load) */
}
/*
 * Class constructor called when the class is created.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory__ctor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ProtocolFactory__ctor(
  /* in */ sidl_rmi_ProtocolFactory self)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory._ctor) */
  /* Insert the implementation of the constructor method here... */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory._ctor) */
}

/*
 * Class destructor called when the class is deleted.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory__dtor"

#ifdef __cplusplus
extern "C"
#endif
void
impl_sidl_rmi_ProtocolFactory__dtor(
  /* in */ sidl_rmi_ProtocolFactory self)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory._dtor) */
  /* Insert the implementation of the destructor method here... */
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory._dtor) */
}

/*
 * Associate a particular prefix in the URL to a typeName <code>sidl.Loader</code> can find.
 * The actual type is expected to implement <code>sidl.rmi.InstanceHandle</code>
 * Return true iff the addition is successful.  (no collisions allowed)
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory_addProtocol"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidl_rmi_ProtocolFactory_addProtocol(
  /* in */ const char* prefix,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory.addProtocol) */
  int i;
  /* push new protocol to back of list */
  if ( len >= maxlen ) { 
    ;/*TODO implement realloc */
  } 
  /* return false is prefix already exists */
  for (i=0;i<2*len;i+=2) { 
    if ( !strcmp( reg[i], prefix )) { 
      return FALSE;
    }
  }
  /* now add prefix into list */
  reg[2*len] = (char*)sidl_String_strdup(prefix);
  reg[2*len+1] = (char*) sidl_String_strdup(typeName);
  len++;
  return TRUE;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory.addProtocol) */
}

/*
 * Return the typeName associated with a particular prefix.
 * Return empty string if the prefix
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory_getProtocol"

#ifdef __cplusplus
extern "C"
#endif
char*
impl_sidl_rmi_ProtocolFactory_getProtocol(
  /* in */ const char* prefix,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory.getProtocol) */
  int i;
  for (i=0;i<2*len;i+=2) { 
    if ( !strcmp( reg[i], prefix )) { 
      return (char*) sidl_String_strdup( reg[i+1] );
    }
  }
  return NULL;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory.getProtocol) */
}

/*
 * Remove a protocol from the active list.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory_deleteProtocol"

#ifdef __cplusplus
extern "C"
#endif
sidl_bool
impl_sidl_rmi_ProtocolFactory_deleteProtocol(
  /* in */ const char* prefix,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory.deleteProtocol) */
  int i;
  for (i=0;i<2*len;i+=2) { 
    if ( !strcmp( reg[i], prefix )) { 
      --len;
      if (i<2*len) { 
	/* swap i entry to back of list */
	char *tmp;
	tmp=reg[i];
	reg[i]=reg[2*len];
	reg[2*len]= tmp;
	tmp=reg[i+1];
	reg[i+1] = reg[2*len+1];
	reg[2*len+1] = tmp;
      }
      sidl_String_free(reg[2*len]);
      reg[2*len]=NULL;
      sidl_String_free(reg[2*len+1]);
      reg[2*len+1]=NULL;
      return TRUE;
    }
  }
  return FALSE;
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory.deleteProtocol) */
}

/*
 * Create a new remote object and a return an instance handle for that object. 
 * The server and port number are in the url.  Return nil 
 * if protocol unknown or InstanceHandle.init() failed.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory_createInstance"

#ifdef __cplusplus
extern "C"
#endif
sidl_rmi_InstanceHandle
impl_sidl_rmi_ProtocolFactory_createInstance(
  /* in */ const char* url,
  /* in */ const char* typeName,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory.createInstance) */
  char * prefix = NULL;
  char * protocol = NULL;
  char * server_name = NULL;
  int port = 0;
  sidl_BaseClass bc;
  sidl_rmi_InstanceHandle ih = NULL;
  char* myurl = sidl_String_strdup( url );
  sidl_DLL dll;

  parseURL(myurl, &prefix, &server_name, &port, NULL, NULL, _ex);

  /* now find the protocol associated with the prefix */
  
  protocol = sidl_rmi_ProtocolFactory_getProtocol( prefix, _ex );
  if ( protocol == NULL ) { return NULL; }
  dll = sidl_Loader_findLibrary( protocol, "ior/impl", 
					  sidl_Scope_SCLSCOPE, 
					  sidl_Resolve_SCLRESOLVE );
  if ( dll == NULL ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl.rmi.ProtocolFactory: Protocol cannot be loaded\n"); 
  }
  bc = sidl_DLL_createClass( dll, protocol );
  if ( bc == NULL ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl.rmi.ProtocolFactory: Protocol cannot be created\n");
  }    
  ih = sidl_rmi_InstanceHandle__cast( bc );
  if ( ih == NULL ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl.rmi.ProtocolFactory: Protocol doesn't implement InstanceHandle\n");
  }
  sidl_rmi_InstanceHandle_initCreate( ih, url, typeName, _ex );
 EXIT:
  sidl_String_free(myurl);
  sidl_String_free(protocol);
  sidl_String_free(server_name);
  if(dll) {
    sidl_DLL_deleteRef(dll);
  }
  return ih;  /* On error, ih will be NULL*/
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory.createInstance) */
}

/*
 * Create an new connection linked to an already existing object on a remote 
 * server.  The server and port number are in the url, the objectID is the unique ID
 * of the remote object in the remote instance registry. 
 * Return nil if protocol unknown or InstanceHandle.init() failed.
 */

#undef __FUNC__
#define __FUNC__ "impl_sidl_rmi_ProtocolFactory_connectInstance"

#ifdef __cplusplus
extern "C"
#endif
sidl_rmi_InstanceHandle
impl_sidl_rmi_ProtocolFactory_connectInstance(
  /* in */ const char* url,
  /* out */ sidl_BaseInterface *_ex)
{
  /* DO-NOT-DELETE splicer.begin(sidl.rmi.ProtocolFactory.connectInstance) */
  char * prefix = NULL;
  char * protocol = NULL;
  char * server_name = NULL;
  char * typeName = NULL;
  char * objectID = NULL;
  char* myurl = sidl_String_strdup( url );
  int port = 0;
  sidl_BaseClass bc;
  sidl_rmi_InstanceHandle ih;
  sidl_DLL dll;

  parseURL(myurl, &prefix, &server_name, &port, &typeName, &objectID, _ex);

  /* now find the protocol associated with the prefix */
  
  protocol = sidl_rmi_ProtocolFactory_getProtocol( prefix, _ex );
  if ( protocol == NULL ) { return NULL; }
  dll = sidl_Loader_findLibrary( protocol, "ior/impl", 
					  sidl_Scope_SCLSCOPE, 
					  sidl_Resolve_SCLRESOLVE );
  if ( dll == NULL ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl.rmi.ProtocolFactory: Protocol cannot be loaded\n"); 
  }
  bc = sidl_DLL_createClass( dll, protocol );
  if ( bc == NULL ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl.rmi.ProtocolFactory: Protocol cannot be created\n");
  }    
  ih = sidl_rmi_InstanceHandle__cast( bc );
  if ( ih == NULL ) { 
    SIDL_THROW(*_ex, sidl_rmi_NetworkException, 
	       "sidl.rmi.ProtocolFactory: Protocol doesn't implement InstanceHandle\n");
  }
  sidl_rmi_InstanceHandle_initConnect( ih, url, _ex );
 
 EXIT:
  sidl_String_free(myurl);
  sidl_String_free(protocol);
  sidl_String_free(server_name);
  sidl_String_free(objectID);
  sidl_String_free(typeName);
  if(dll) {
    sidl_DLL_deleteRef(dll);
  }
  return ih; /* Will be NULL if there is a problem*/
  /* DO-NOT-DELETE splicer.end(sidl.rmi.ProtocolFactory.connectInstance) */
}
/* Babel internal methods, Users should not edit below this line. */
struct sidl_rmi_InstanceHandle__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_InstanceHandle(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_InstanceHandle__connect(url, _ex);
}
char * impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_InstanceHandle(struct 
  sidl_rmi_InstanceHandle__object* obj) {
  return sidl_rmi_InstanceHandle__getURL(obj);
}
struct sidl_ClassInfo__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_ClassInfo(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_ClassInfo__connect(url, _ex);
}
char * impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_ClassInfo(struct 
  sidl_ClassInfo__object* obj) {
  return sidl_ClassInfo__getURL(obj);
}
struct sidl_rmi_NetworkException__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_NetworkException(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_NetworkException__connect(url, _ex);
}
char * impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_NetworkException(struct 
  sidl_rmi_NetworkException__object* obj) {
  return sidl_rmi_NetworkException__getURL(obj);
}
struct sidl_BaseInterface__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseInterface(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseInterface__connect(url, _ex);
}
char * impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseInterface(struct 
  sidl_BaseInterface__object* obj) {
  return sidl_BaseInterface__getURL(obj);
}
struct sidl_rmi_ProtocolFactory__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_rmi_ProtocolFactory(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_rmi_ProtocolFactory__connect(url, _ex);
}
char * impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_rmi_ProtocolFactory(struct 
  sidl_rmi_ProtocolFactory__object* obj) {
  return sidl_rmi_ProtocolFactory__getURL(obj);
}
struct sidl_BaseClass__object* 
  impl_sidl_rmi_ProtocolFactory_fconnect_sidl_BaseClass(char* url,
  sidl_BaseInterface *_ex) {
  return sidl_BaseClass__connect(url, _ex);
}
char * impl_sidl_rmi_ProtocolFactory_fgetURL_sidl_BaseClass(struct 
  sidl_BaseClass__object* obj) {
  return sidl_BaseClass__getURL(obj);
}
