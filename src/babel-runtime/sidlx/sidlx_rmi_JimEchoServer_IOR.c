/*
 * File:          sidlx_rmi_JimEchoServer_IOR.c
 * Symbol:        sidlx.rmi.JimEchoServer-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for sidlx.rmi.JimEchoServer
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

/*
 * Begin: RMI includes
 */

#include "sidl_rmi_InstanceHandle.h"
#include "sidl_rmi_InstanceRegistry.h"
#include "sidl_rmi_ServerRegistry.h"
#include "sidl_rmi_Call.h"
#include "sidl_rmi_Return.h"
#include "sidl_Exception.h"
#include "sidl_exec_err.h"
#include "sidl_PreViolation.h"
#include <stdio.h>
/*
 * End: RMI includes
 */

#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidlx_rmi_JimEchoServer_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidlx_rmi_JimEchoServer__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_JimEchoServer__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_JimEchoServer__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_JimEchoServer__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 1;
static const int32_t s_IOR_MINOR_VERSION = 0;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;

/*
 * Static variable to make sure _load called no more than once
 */

static int s_load_called = 0;
/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;

static struct sidlx_rmi_JimEchoServer__epv s_new_epv__sidlx_rmi_jimechoserver;

static struct sidlx_rmi_JimEchoServer__epv 
  s_new_epv_hooks__sidlx_rmi_jimechoserver;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv  s_new_epv_hooks__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv_hooks__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_new_epv_hooks__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv_hooks__sidl_baseinterface;

static struct sidl_rmi_ServerInfo__epv  s_new_epv__sidl_rmi_serverinfo;
static struct sidl_rmi_ServerInfo__epv  s_new_epv_hooks__sidl_rmi_serverinfo;
static struct sidl_rmi_ServerInfo__epv* s_old_epv__sidl_rmi_serverinfo;
static struct sidl_rmi_ServerInfo__epv* s_old_epv_hooks__sidl_rmi_serverinfo;

static struct sidlx_rmi_SimpleServer__epv  s_new_epv__sidlx_rmi_simpleserver;
static struct sidlx_rmi_SimpleServer__epv  
  s_new_epv_hooks__sidlx_rmi_simpleserver;
static struct sidlx_rmi_SimpleServer__epv* s_old_epv__sidlx_rmi_simpleserver;
static struct sidlx_rmi_SimpleServer__epv* 
  s_old_epv_hooks__sidlx_rmi_simpleserver;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_rmi_JimEchoServer__set_epv(
  struct sidlx_rmi_JimEchoServer__epv* epv);
extern void sidlx_rmi_JimEchoServer__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_rmi_JimEchoServer_serviceRequest__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* sock_str = NULL;
  struct sidlx_rmi_Socket__object* sock = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "sock", &sock_str, _ex);SIDL_CHECK(*_ex);
  sock = skel_sidlx_rmi_JimEchoServer_fconnect_sidlx_rmi_Socket(sock_str, TRUE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_serviceRequest)(
    self,
    sock,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(sock) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)sock, _ex); SIDL_CHECK(
      *_ex);
    if(sock_str) {free(sock_str);}
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_isLocalObject__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* url= NULL;
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "url", &url, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_isLocalObject)(
    self,
    url,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packString( outArgs, "_retval", _retval, _ex);SIDL_CHECK(
    *_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(_retval) {
    free(_retval);
  }
  if(url) {free(url);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_getExceptions__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_io_Serializable__array* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getExceptions)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packSerializableArray( outArgs, "_retval", _retval,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)_retval);
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_setMaxThreadPool__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t max = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "max", &max, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_setMaxThreadPool)(
    self,
    max,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_requestPort__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t port = 0;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "port", &port, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_requestPort)(
    self,
    port,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packBool( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_requestPortInRange__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t minport = 0;
  int32_t maxport = 0;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "minport", &minport, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "maxport", &maxport, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_requestPortInRange)(
    self,
    minport,
    maxport,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packBool( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_getPort__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getPort)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_getServerName__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getServerName)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packString( outArgs, "_retval", _retval, _ex);SIDL_CHECK(
    *_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(_retval) {
    free(_retval);
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_getServerURL__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* objID= NULL;
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "objID", &objID, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_getServerURL)(
    self,
    objID,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packString( outArgs, "_retval", _retval, _ex);SIDL_CHECK(
    *_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(_retval) {
    free(_retval);
  }
  if(objID) {free(objID);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_run__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int64_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_run)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packLong( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_shutdown__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_shutdown)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_addRef__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_deleteRef__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_isSame__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* iobj_str = NULL;
  struct sidl_BaseInterface__object* iobj = NULL;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "iobj", &iobj_str, _ex);SIDL_CHECK(*_ex);
  iobj = skel_sidlx_rmi_JimEchoServer_fconnect_sidl_BaseInterface(iobj_str, 
    TRUE, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packBool( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(iobj) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)iobj, _ex); SIDL_CHECK(
      *_ex);
    if(iobj_str) {free(iobj_str);}
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_isType__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "name", &name, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packBool( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(name) {free(name);}
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void
sidlx_rmi_JimEchoServer_getClassInfo__exec(
        struct sidlx_rmi_JimEchoServer__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  if(_retval){
    char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)_retval, 
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Return_packString( outArgs, "_retval", _url, _ex);SIDL_CHECK(*_ex);
    free((void*)_url);
  } else {
    sidl_rmi_Return_packString( outArgs, "_retval", NULL, _ex);SIDL_CHECK(*_ex);
  }
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(_retval && sidl_BaseInterface__isRemote((sidl_BaseInterface)_retval, _ex)) 
    {
    (*((sidl_BaseInterface)_retval)->d_epv->f__raddRef)(((
      sidl_BaseInterface)_retval)->d_object, _ex); SIDL_CHECK(*_ex);
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)_retval, _ex); SIDL_CHECK(
      *_ex);
  }
  return;

  EXIT:
  _SIDLex = sidl_BaseException__cast(*_ex,&_ex3); EXEC_CHECK(_ex3);
  sidl_rmi_Return_throwException(outArgs, _SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseException_deleteRef(_SIDLex, &_ex3); EXEC_CHECK(_ex3);
  sidl_BaseInterface_deleteRef(*_ex, &_ex3); EXEC_CHECK(_ex3);
  *_ex = NULL;
  return;
  EXEC_ERR:
  {
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseInterface_deleteRef(_ex3, &_throwaway);
    return;
  }
}

static void ior_sidlx_rmi_JimEchoServer__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_rmi_JimEchoServer__call_load();
    s_load_called=1;
  }
}

/* CAST: dynamic type casting support. */
static void* ior_sidlx_rmi_JimEchoServer__cast(
  struct sidlx_rmi_JimEchoServer__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.rmi.ServerInfo");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidlx_rmi_simpleserver.d_sidl_rmi_serverinfo);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((
        *self).d_sidlx_rmi_simpleserver.d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.BaseClass");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct sidl_BaseClass__object*)self);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidlx.rmi.SimpleServer");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct sidlx_rmi_SimpleServer__object*)self);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidlx.rmi.JimEchoServer");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct sidlx_rmi_JimEchoServer__object*)self);
        return cast;
      }
    }
  }
  return cast;
  EXIT:
  return NULL;
}

/*
 * HOOKS: set hooks activation.
 */

static void ior_sidlx_rmi_JimEchoServer__set_hooks(
  struct sidlx_rmi_JimEchoServer__object* self,
  int on, struct sidl_BaseInterface__object **_ex ) { 
  *_ex = NULL;
  /*
   * Nothing else to do since hooks support not needed.
   */

}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidlx_rmi_JimEchoServer__delete(
  struct sidlx_rmi_JimEchoServer__object* self, struct 
    sidl_BaseInterface__object **_ex)
{
  *_ex = NULL; /* default to no exception */
  sidlx_rmi_JimEchoServer__fini(self,_ex);
  memset((void*)self, 0, sizeof(struct sidlx_rmi_JimEchoServer__object));
  free((void*) self);
}

static char*
ior_sidlx_rmi_JimEchoServer__getURL(
    struct sidlx_rmi_JimEchoServer__object* self,
    struct sidl_BaseInterface__object **_ex) {
  char* ret = NULL;
  char* objid = sidl_rmi_InstanceRegistry_getInstanceByClass((
    sidl_BaseClass)self, _ex); SIDL_CHECK(*_ex);
  if(!objid) {
    objid = sidl_rmi_InstanceRegistry_registerInstance((sidl_BaseClass)self, 
      _ex); SIDL_CHECK(*_ex);
  }
  ret = sidl_rmi_ServerRegistry_getServerURL(objid, _ex); SIDL_CHECK(*_ex);
  return ret;
  EXIT:
  return NULL;
}
static void
ior_sidlx_rmi_JimEchoServer__raddRef(
    struct sidlx_rmi_JimEchoServer__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_sidlx_rmi_JimEchoServer__isRemote(
    struct sidlx_rmi_JimEchoServer__object* self, sidl_BaseInterface* _ex) {
  *_ex = NULL; /* default to no exception */
  return FALSE;
}

struct sidlx_rmi_JimEchoServer__method {
  const char *d_name;
  void (*d_func)(struct sidlx_rmi_JimEchoServer__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

static void
ior_sidlx_rmi_JimEchoServer__exec(
    struct sidlx_rmi_JimEchoServer__object* self,
    const char* methodName,
    struct sidl_rmi_Call__object* inArgs,
    struct sidl_rmi_Return__object* outArgs,
    struct sidl_BaseInterface__object **_ex ) { 
  static const struct sidlx_rmi_JimEchoServer__method  s_methods[] = {
    { "addRef", sidlx_rmi_JimEchoServer_addRef__exec },
    { "deleteRef", sidlx_rmi_JimEchoServer_deleteRef__exec },
    { "getClassInfo", sidlx_rmi_JimEchoServer_getClassInfo__exec },
    { "getExceptions", sidlx_rmi_JimEchoServer_getExceptions__exec },
    { "getPort", sidlx_rmi_JimEchoServer_getPort__exec },
    { "getServerName", sidlx_rmi_JimEchoServer_getServerName__exec },
    { "getServerURL", sidlx_rmi_JimEchoServer_getServerURL__exec },
    { "isLocalObject", sidlx_rmi_JimEchoServer_isLocalObject__exec },
    { "isSame", sidlx_rmi_JimEchoServer_isSame__exec },
    { "isType", sidlx_rmi_JimEchoServer_isType__exec },
    { "requestPort", sidlx_rmi_JimEchoServer_requestPort__exec },
    { "requestPortInRange", sidlx_rmi_JimEchoServer_requestPortInRange__exec },
    { "run", sidlx_rmi_JimEchoServer_run__exec },
    { "serviceRequest", sidlx_rmi_JimEchoServer_serviceRequest__exec },
    { "setMaxThreadPool", sidlx_rmi_JimEchoServer_setMaxThreadPool__exec },
    { "shutdown", sidlx_rmi_JimEchoServer_shutdown__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_rmi_JimEchoServer__method);
  *_ex = NULL; /* default to no exception */
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs, _ex); SIDL_CHECK(*_ex);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
  SIDL_THROW(*_ex,sidl_PreViolation,"method name not found");
  EXIT:
  return;
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void sidlx_rmi_JimEchoServer__init_epv(void)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_rmi_JimEchoServer__epv*  epv  = 
    &s_new_epv__sidlx_rmi_jimechoserver;
  struct sidlx_rmi_JimEchoServer__epv*  hepv = 
    &s_new_epv_hooks__sidlx_rmi_jimechoserver;
  struct sidl_BaseClass__epv*           e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseClass__epv*           he0  = &s_new_epv_hooks__sidl_baseclass;
  struct sidl_BaseInterface__epv*       e1   = &s_new_epv__sidl_baseinterface;
  struct sidl_BaseInterface__epv*       he1  = 
    &s_new_epv_hooks__sidl_baseinterface;
  struct sidl_rmi_ServerInfo__epv*      e2   = &s_new_epv__sidl_rmi_serverinfo;
  struct sidl_rmi_ServerInfo__epv*      he2  = 
    &s_new_epv_hooks__sidl_rmi_serverinfo;
  struct sidlx_rmi_SimpleServer__epv*   e3   = 
    &s_new_epv__sidlx_rmi_simpleserver;
  struct sidlx_rmi_SimpleServer__epv*   he3  = 
    &s_new_epv_hooks__sidlx_rmi_simpleserver;

  struct sidlx_rmi_SimpleServer__epv*  s1 = NULL;
  struct sidlx_rmi_SimpleServer__epv*  h1 = NULL;
  struct sidl_BaseClass__epv*          s2 = NULL;
  struct sidl_BaseClass__epv*          h2 = NULL;

  sidlx_rmi_SimpleServer__getEPVs(
    &s_old_epv__sidl_baseinterface,
    &s_old_epv_hooks__sidl_baseinterface,
    &s_old_epv__sidl_baseclass,&s_old_epv_hooks__sidl_baseclass,
    &s_old_epv__sidl_rmi_serverinfo,
    &s_old_epv_hooks__sidl_rmi_serverinfo,
    &s_old_epv__sidlx_rmi_simpleserver,
      &s_old_epv_hooks__sidlx_rmi_simpleserver);
  /*
   * Here we alias the static epvs to some handy small names
   */

  s2  =  s_old_epv__sidl_baseclass;
  h2  =  s_old_epv_hooks__sidl_baseclass;
  s1  =  s_old_epv__sidlx_rmi_simpleserver;
  h1  =  s_old_epv_hooks__sidlx_rmi_simpleserver;

  epv->f__cast                    = ior_sidlx_rmi_JimEchoServer__cast;
  epv->f__delete                  = ior_sidlx_rmi_JimEchoServer__delete;
  epv->f__exec                    = ior_sidlx_rmi_JimEchoServer__exec;
  epv->f__getURL                  = ior_sidlx_rmi_JimEchoServer__getURL;
  epv->f__raddRef                 = ior_sidlx_rmi_JimEchoServer__raddRef;
  epv->f__isRemote                = ior_sidlx_rmi_JimEchoServer__isRemote;
  epv->f__set_hooks               = ior_sidlx_rmi_JimEchoServer__set_hooks;
  epv->f__ctor                    = NULL;
  epv->f__ctor2                   = NULL;
  epv->f__dtor                    = NULL;
  epv->f_serviceRequest           = (void (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidlx_rmi_Socket__object*,struct 
    sidl_BaseInterface__object **)) s1->f_serviceRequest;
  epv->f_isLocalObject            = (char* (*)(struct 
    sidlx_rmi_JimEchoServer__object*,const char*,struct 
    sidl_BaseInterface__object **)) s1->f_isLocalObject;
  epv->f_getExceptions            = (struct sidl_io_Serializable__array* (*)(
    struct sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object 
    **)) s1->f_getExceptions;
  epv->f_setMaxThreadPool         = (void (*)(struct 
    sidlx_rmi_JimEchoServer__object*,int32_t,struct sidl_BaseInterface__object 
    **)) s1->f_setMaxThreadPool;
  epv->f_requestPort              = (sidl_bool (*)(struct 
    sidlx_rmi_JimEchoServer__object*,int32_t,struct sidl_BaseInterface__object 
    **)) s1->f_requestPort;
  epv->f_requestPortInRange       = (sidl_bool (*)(struct 
    sidlx_rmi_JimEchoServer__object*,int32_t,int32_t,struct 
    sidl_BaseInterface__object **)) s1->f_requestPortInRange;
  epv->f_getPort                  = (int32_t (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getPort;
  epv->f_getServerName            = (char* (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getServerName;
  epv->f_getServerURL             = (char* (*)(struct 
    sidlx_rmi_JimEchoServer__object*,const char*,struct 
    sidl_BaseInterface__object **)) s1->f_getServerURL;
  epv->f_run                      = (int64_t (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_run;
  epv->f_shutdown                 = (void (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_shutdown;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object*,struct 
    sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_rmi_JimEchoServer__object*,const char*,struct 
    sidl_BaseInterface__object **)) s1->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_JimEchoServer__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getClassInfo;

  sidlx_rmi_JimEchoServer__set_epv(epv);

  memcpy((void*)hepv, epv, sizeof(struct sidlx_rmi_JimEchoServer__epv));
  e0->f__cast               = (void* (*)(struct sidl_BaseClass__object*,const 
    char*, struct sidl_BaseInterface__object**)) epv->f__cast;
  e0->f__delete             = (void (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e0->f__getURL             = (char* (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e0->f__raddRef            = (void (*)(struct sidl_BaseClass__object*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e0->f__isRemote           = (sidl_bool (*)(struct sidl_BaseClass__object*, 
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e0->f__exec               = (void (*)(struct sidl_BaseClass__object*,const 
    char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef              = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e0->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;

  memcpy((void*) he0, e0, sizeof(struct sidl_BaseClass__epv));

  e1->f__cast               = (void* (*)(void*,const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e1->f__delete             = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e1->f__getURL             = (char* (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e1->f__raddRef            = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e1->f__isRemote           = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__isRemote;
  e1->f__exec               = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef              = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_addRef;
  e1->f_deleteRef           = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_deleteRef;
  e1->f_isSame              = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e1->f_isType              = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he1, e1, sizeof(struct sidl_BaseInterface__epv));

  e2->f__cast               = (void* (*)(void*,const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e2->f__delete             = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e2->f__getURL             = (char* (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e2->f__raddRef            = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e2->f__isRemote           = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__isRemote;
  e2->f__exec               = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_getServerURL        = (char* (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_getServerURL;
  e2->f_isLocalObject       = (char* (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isLocalObject;
  e2->f_getExceptions       = (struct sidl_io_Serializable__array* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getExceptions;
  e2->f_addRef              = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e2->f_isType              = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he2, e2, sizeof(struct sidl_rmi_ServerInfo__epv));

  e3->f__cast               = (void* (*)(struct sidlx_rmi_SimpleServer__object*,
    const char*, struct sidl_BaseInterface__object**)) epv->f__cast;
  e3->f__delete             = (void (*)(struct sidlx_rmi_SimpleServer__object*, 
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e3->f__getURL             = (char* (*)(struct sidlx_rmi_SimpleServer__object*,
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e3->f__raddRef            = (void (*)(struct sidlx_rmi_SimpleServer__object*, 
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e3->f__isRemote           = (sidl_bool (*)(struct 
    sidlx_rmi_SimpleServer__object*, struct sidl_BaseInterface__object **)) 
    epv->f__isRemote;
  e3->f__exec               = (void (*)(struct sidlx_rmi_SimpleServer__object*,
    const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_setMaxThreadPool    = (void (*)(struct sidlx_rmi_SimpleServer__object*,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_setMaxThreadPool;
  e3->f_requestPort         = (sidl_bool (*)(struct 
    sidlx_rmi_SimpleServer__object*,int32_t,struct sidl_BaseInterface__object 
    **)) epv->f_requestPort;
  e3->f_requestPortInRange  = (sidl_bool (*)(struct 
    sidlx_rmi_SimpleServer__object*,int32_t,int32_t,struct 
    sidl_BaseInterface__object **)) epv->f_requestPortInRange;
  e3->f_getPort             = (int32_t (*)(struct 
    sidlx_rmi_SimpleServer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getPort;
  e3->f_getServerName       = (char* (*)(struct sidlx_rmi_SimpleServer__object*,
    struct sidl_BaseInterface__object **)) epv->f_getServerName;
  e3->f_getServerURL        = (char* (*)(struct sidlx_rmi_SimpleServer__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_getServerURL;
  e3->f_run                 = (int64_t (*)(struct 
    sidlx_rmi_SimpleServer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_run;
  e3->f_shutdown            = (void (*)(struct sidlx_rmi_SimpleServer__object*,
    struct sidl_BaseInterface__object **)) epv->f_shutdown;
  e3->f_serviceRequest      = (void (*)(struct sidlx_rmi_SimpleServer__object*,
    struct sidlx_rmi_Socket__object*,struct sidl_BaseInterface__object **)) 
    epv->f_serviceRequest;
  e3->f_getExceptions       = (struct sidl_io_Serializable__array* (*)(struct 
    sidlx_rmi_SimpleServer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getExceptions;
  e3->f_addRef              = (void (*)(struct sidlx_rmi_SimpleServer__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef           = (void (*)(struct sidlx_rmi_SimpleServer__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(struct 
    sidlx_rmi_SimpleServer__object*,struct sidl_BaseInterface__object*,struct 
    sidl_BaseInterface__object **)) epv->f_isSame;
  e3->f_isType              = (sidl_bool (*)(struct 
    sidlx_rmi_SimpleServer__object*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_SimpleServer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;
  e3->f_isLocalObject       = (char* (*)(struct sidlx_rmi_SimpleServer__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isLocalObject;

  memcpy((void*) he3, e3, sizeof(struct sidlx_rmi_SimpleServer__epv));

  s_method_initialized = 1;
  ior_sidlx_rmi_JimEchoServer__ensure_load_called();
}

void sidlx_rmi_JimEchoServer__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_rmi_ServerInfo__epv **s_arg_epv__sidl_rmi_serverinfo,
  struct sidl_rmi_ServerInfo__epv **s_arg_epv_hooks__sidl_rmi_serverinfo,
  struct sidlx_rmi_SimpleServer__epv **s_arg_epv__sidlx_rmi_simpleserver,struct 
    sidlx_rmi_SimpleServer__epv **s_arg_epv_hooks__sidlx_rmi_simpleserver,
  struct sidlx_rmi_JimEchoServer__epv **s_arg_epv__sidlx_rmi_jimechoserver,
    struct sidlx_rmi_JimEchoServer__epv 
    **s_arg_epv_hooks__sidlx_rmi_jimechoserver)
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_JimEchoServer__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_new_epv__sidl_baseinterface;
  *s_arg_epv_hooks__sidl_baseinterface = &s_new_epv_hooks__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_new_epv__sidl_baseclass;
  *s_arg_epv_hooks__sidl_baseclass = &s_new_epv_hooks__sidl_baseclass;
  *s_arg_epv__sidl_rmi_serverinfo = &s_new_epv__sidl_rmi_serverinfo;
  *s_arg_epv_hooks__sidl_rmi_serverinfo = &s_new_epv_hooks__sidl_rmi_serverinfo;
  *s_arg_epv__sidlx_rmi_simpleserver = &s_new_epv__sidlx_rmi_simpleserver;
  *s_arg_epv_hooks__sidlx_rmi_simpleserver = 
    &s_new_epv_hooks__sidlx_rmi_simpleserver;
  *s_arg_epv__sidlx_rmi_jimechoserver = &s_new_epv__sidlx_rmi_jimechoserver;
  *s_arg_epv_hooks__sidlx_rmi_jimechoserver = 
    &s_new_epv_hooks__sidlx_rmi_jimechoserver;
}
/*
 * SUPER: returns parent's non-overrided EPV
 */

static struct sidlx_rmi_SimpleServer__epv* sidlx_rmi_JimEchoServer__super(void) 
  {
  return s_old_epv__sidlx_rmi_simpleserver;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info, struct sidl_BaseInterface__object **_ex)
{
  LOCK_STATIC_GLOBALS;
  *_ex = NULL; /* default to no exception */
  if (!s_classInfo) {
    sidl_ClassInfoI impl;
    impl = sidl_ClassInfoI__create(_ex);
    s_classInfo = sidl_ClassInfo__cast(impl,_ex);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "sidlx.rmi.JimEchoServer",_ex);
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION, 
        s_IOR_MINOR_VERSION,_ex);
      sidl_ClassInfoI_deleteRef(impl,_ex);
      sidl_atexit(sidl_deleteRef_atexit, &s_classInfo);
    }
  }
  UNLOCK_STATIC_GLOBALS;
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info,_ex);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info,_ex);
  }
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct sidlx_rmi_JimEchoServer__object* self, sidl_BaseInterface* 
  _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((
      *self).d_sidlx_rmi_simpleserver.d_sidl_baseclass.d_data);
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo),_ex); SIDL_CHECK(*_ex);
    }
  }
EXIT:
return;
}

/*
 * NEW: allocate object and initialize it.
 */

struct sidlx_rmi_JimEchoServer__object*
sidlx_rmi_JimEchoServer__new(void* ddata, struct sidl_BaseInterface__object ** 
  _ex)
{
  struct sidlx_rmi_JimEchoServer__object* self =
    (struct sidlx_rmi_JimEchoServer__object*) malloc(
      sizeof(struct sidlx_rmi_JimEchoServer__object));
  *_ex = NULL; /* default to no exception */
  sidlx_rmi_JimEchoServer__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_rmi_JimEchoServer__init(
  struct sidlx_rmi_JimEchoServer__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidlx_rmi_JimEchoServer__object* s0 = self;
  struct sidlx_rmi_SimpleServer__object*  s1 = &s0->d_sidlx_rmi_simpleserver;
  struct sidl_BaseClass__object*          s2 = &s1->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_JimEchoServer__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidlx_rmi_SimpleServer__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

  s2->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s2->d_epv                      = &s_new_epv__sidl_baseclass;

  s1->d_sidl_rmi_serverinfo.d_epv = &s_new_epv__sidl_rmi_serverinfo;
  s1->d_epv                       = &s_new_epv__sidlx_rmi_simpleserver;

  s0->d_epv    = &s_new_epv__sidlx_rmi_jimechoserver;

  s0->d_data = NULL;

  ior_sidlx_rmi_JimEchoServer__set_hooks(s0, FALSE, _ex);

  if(ddata) {
    self->d_data = ddata;
    (*(self->d_epv->f__ctor2))(self,ddata,_ex); SIDL_CHECK(*_ex);
  } else { 
    (*(self->d_epv->f__ctor))(self,_ex); SIDL_CHECK(*_ex);
  }
  EXIT:
  return;
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidlx_rmi_JimEchoServer__fini(
  struct sidlx_rmi_JimEchoServer__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidlx_rmi_JimEchoServer__object* s0 = self;
  struct sidlx_rmi_SimpleServer__object*  s1 = &s0->d_sidlx_rmi_simpleserver;
  struct sidl_BaseClass__object*          s2 = &s1->d_sidl_baseclass;

  *_ex = NULL; /* default to no exception */
  (*(s0->d_epv->f__dtor))(s0,_ex);
  SIDL_CHECK(*_ex);

  s2->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s2->d_epv                      = s_old_epv__sidl_baseclass;

  s1->d_sidl_rmi_serverinfo.d_epv = s_old_epv__sidl_rmi_serverinfo;
  s1->d_epv                       = s_old_epv__sidlx_rmi_simpleserver;

  sidlx_rmi_SimpleServer__fini(s1, _ex); SIDL_CHECK(*_ex);
  EXIT:
  return;
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_rmi_JimEchoServer__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_rmi_JimEchoServer__external
s_externalEntryPoints = {
  sidlx_rmi_JimEchoServer__new,
  sidlx_rmi_JimEchoServer__super,
  1, 
  0
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_JimEchoServer__external*
sidlx_rmi_JimEchoServer__externals(void)
{
  return &s_externalEntryPoints;
}

