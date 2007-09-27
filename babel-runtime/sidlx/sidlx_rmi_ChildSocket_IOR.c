/*
 * File:          sidlx_rmi_ChildSocket_IOR.c
 * Symbol:        sidlx.rmi.ChildSocket-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for sidlx.rmi.ChildSocket
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
#include "sidlx_rmi_ChildSocket_IOR.h"
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
static struct sidl_recursive_mutex_t sidlx_rmi_ChildSocket__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_ChildSocket__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_ChildSocket__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_ChildSocket__mutex )==EDEADLOCK) */
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

static struct sidlx_rmi_ChildSocket__epv s_new_epv__sidlx_rmi_childsocket;

static struct sidlx_rmi_ChildSocket__epv s_new_epv_hooks__sidlx_rmi_childsocket;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv  s_new_epv_hooks__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv_hooks__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_new_epv_hooks__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv_hooks__sidl_baseinterface;

static struct sidlx_rmi_IPv4Socket__epv  s_new_epv__sidlx_rmi_ipv4socket;
static struct sidlx_rmi_IPv4Socket__epv  s_new_epv_hooks__sidlx_rmi_ipv4socket;
static struct sidlx_rmi_IPv4Socket__epv* s_old_epv__sidlx_rmi_ipv4socket;
static struct sidlx_rmi_IPv4Socket__epv* s_old_epv_hooks__sidlx_rmi_ipv4socket;

static struct sidlx_rmi_Socket__epv  s_new_epv__sidlx_rmi_socket;
static struct sidlx_rmi_Socket__epv  s_new_epv_hooks__sidlx_rmi_socket;
static struct sidlx_rmi_Socket__epv* s_old_epv__sidlx_rmi_socket;
static struct sidlx_rmi_Socket__epv* s_old_epv_hooks__sidlx_rmi_socket;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_rmi_ChildSocket__set_epv(
  struct sidlx_rmi_ChildSocket__epv* epv);
extern void sidlx_rmi_ChildSocket__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_rmi_ChildSocket_init__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t fileDes = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "fileDes", &fileDes, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_init)(
    self,
    fileDes,
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
sidlx_rmi_ChildSocket_getsockname__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t address_data = 0;
  int32_t* address = &address_data;
  int32_t port_data = 0;
  int32_t* port = &port_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "address", address, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "port", port, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_getsockname)(
    self,
    address,
    port,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "address", *address, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packInt( outArgs, "port", *port, _ex);SIDL_CHECK(*_ex);
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
sidlx_rmi_ChildSocket_getpeername__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t address_data = 0;
  int32_t* address = &address_data;
  int32_t port_data = 0;
  int32_t* port = &port_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "address", address, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "port", port, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_getpeername)(
    self,
    address,
    port,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "address", *address, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Return_packInt( outArgs, "port", *port, _ex);SIDL_CHECK(*_ex);
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
sidlx_rmi_ChildSocket_addRef__exec(
        struct sidlx_rmi_ChildSocket__object* self,
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
sidlx_rmi_ChildSocket_deleteRef__exec(
        struct sidlx_rmi_ChildSocket__object* self,
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
sidlx_rmi_ChildSocket_isSame__exec(
        struct sidlx_rmi_ChildSocket__object* self,
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
  iobj = skel_sidlx_rmi_ChildSocket_fconnect_sidl_BaseInterface(iobj_str, TRUE, 
    _ex);SIDL_CHECK(*_ex);

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
sidlx_rmi_ChildSocket_isType__exec(
        struct sidlx_rmi_ChildSocket__object* self,
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
sidlx_rmi_ChildSocket_getClassInfo__exec(
        struct sidlx_rmi_ChildSocket__object* self,
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

static void
sidlx_rmi_ChildSocket_close__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_close)(
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
sidlx_rmi_ChildSocket_readn__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t nbytes = 0;
  struct sidl_char__array* data_data = NULL;
  struct sidl_char__array** data = &data_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "nbytes", &nbytes, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackCharArray( inArgs, "data", data,0,0,FALSE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_readn)(
    self,
    nbytes,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packCharArray( outArgs, "data", *data,0,0,(*data==data_data), 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)*data);
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
sidlx_rmi_ChildSocket_readline__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t nbytes = 0;
  struct sidl_char__array* data_data = NULL;
  struct sidl_char__array** data = &data_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "nbytes", &nbytes, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackCharArray( inArgs, "data", data,0,0,FALSE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_readline)(
    self,
    nbytes,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packCharArray( outArgs, "data", *data,0,0,(*data==data_data), 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)*data);
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
sidlx_rmi_ChildSocket_readstring__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t nbytes = 0;
  struct sidl_char__array* data_data = NULL;
  struct sidl_char__array** data = &data_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "nbytes", &nbytes, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackCharArray( inArgs, "data", data,0,0,FALSE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_readstring)(
    self,
    nbytes,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packCharArray( outArgs, "data", *data,0,0,(*data==data_data), 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)*data);
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
sidlx_rmi_ChildSocket_readstring_alloc__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_char__array* data_data = NULL;
  struct sidl_char__array** data = &data_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackCharArray( inArgs, "data", data,0,0,FALSE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_readstring_alloc)(
    self,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packCharArray( outArgs, "data", *data,0,0,(*data==data_data), 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)*data);
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
sidlx_rmi_ChildSocket_readint__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t data_data = 0;
  int32_t* data = &data_data;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "data", data, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_readint)(
    self,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "data", *data, _ex);SIDL_CHECK(*_ex);
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
sidlx_rmi_ChildSocket_writen__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t nbytes = 0;
  struct sidl_char__array* data = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "nbytes", &nbytes, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackCharArray( inArgs, "data", &data,0,0,FALSE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_writen)(
    self,
    nbytes,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)data);
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
sidlx_rmi_ChildSocket_writestring__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t nbytes = 0;
  struct sidl_char__array* data = NULL;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "nbytes", &nbytes, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackCharArray( inArgs, "data", &data,0,0,FALSE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_writestring)(
    self,
    nbytes,
    data,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  sidl_rmi_Return_packInt( outArgs, "_retval", _retval, _ex);SIDL_CHECK(*_ex);
  /* pack out and inout argments */
  /* clean-up dangling references */
  sidl__array_deleteRef((struct sidl__array*)data);
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
sidlx_rmi_ChildSocket_writeint__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t data = 0;
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "data", &data, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_writeint)(
    self,
    data,
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
sidlx_rmi_ChildSocket_setFileDescriptor__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t fd = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "fd", &fd, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_setFileDescriptor)(
    self,
    fd,
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
sidlx_rmi_ChildSocket_getFileDescriptor__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getFileDescriptor)(
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
sidlx_rmi_ChildSocket_test__exec(
        struct sidlx_rmi_ChildSocket__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  int32_t secs = 0;
  int32_t usecs = 0;
  sidl_bool _retval = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackInt( inArgs, "secs", &secs, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "usecs", &usecs, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  _retval = (self->d_epv->f_test)(
    self,
    secs,
    usecs,
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

static void ior_sidlx_rmi_ChildSocket__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_rmi_ChildSocket__call_load();
    s_load_called=1;
  }
}

/* CAST: dynamic type casting support. */
static void* ior_sidlx_rmi_ChildSocket__cast(
  struct sidlx_rmi_ChildSocket__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidlx.rmi.ChildSocket");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = ((struct sidlx_rmi_ChildSocket__object*)self);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((
        *self).d_sidlx_rmi_ipv4socket.d_sidl_baseclass.d_sidl_baseinterface);
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
    cmp1 = strcmp(name, "sidlx.rmi.Socket");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidlx_rmi_ipv4socket.d_sidlx_rmi_socket);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidlx.rmi.IPv4Socket");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct sidlx_rmi_IPv4Socket__object*)self);
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

static void ior_sidlx_rmi_ChildSocket__set_hooks(
  struct sidlx_rmi_ChildSocket__object* self,
  int on, struct sidl_BaseInterface__object **_ex ) { 
  *_ex = NULL;
  /*
   * Nothing else to do since hooks support not needed.
   */

}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidlx_rmi_ChildSocket__delete(
  struct sidlx_rmi_ChildSocket__object* self, struct sidl_BaseInterface__object 
    **_ex)
{
  *_ex = NULL; /* default to no exception */
  sidlx_rmi_ChildSocket__fini(self,_ex);
  memset((void*)self, 0, sizeof(struct sidlx_rmi_ChildSocket__object));
  free((void*) self);
}

static char*
ior_sidlx_rmi_ChildSocket__getURL(
    struct sidlx_rmi_ChildSocket__object* self,
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
ior_sidlx_rmi_ChildSocket__raddRef(
    struct sidlx_rmi_ChildSocket__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_sidlx_rmi_ChildSocket__isRemote(
    struct sidlx_rmi_ChildSocket__object* self, sidl_BaseInterface* _ex) {
  *_ex = NULL; /* default to no exception */
  return FALSE;
}

struct sidlx_rmi_ChildSocket__method {
  const char *d_name;
  void (*d_func)(struct sidlx_rmi_ChildSocket__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

static void
ior_sidlx_rmi_ChildSocket__exec(
    struct sidlx_rmi_ChildSocket__object* self,
    const char* methodName,
    struct sidl_rmi_Call__object* inArgs,
    struct sidl_rmi_Return__object* outArgs,
    struct sidl_BaseInterface__object **_ex ) { 
  static const struct sidlx_rmi_ChildSocket__method  s_methods[] = {
    { "addRef", sidlx_rmi_ChildSocket_addRef__exec },
    { "close", sidlx_rmi_ChildSocket_close__exec },
    { "deleteRef", sidlx_rmi_ChildSocket_deleteRef__exec },
    { "getClassInfo", sidlx_rmi_ChildSocket_getClassInfo__exec },
    { "getFileDescriptor", sidlx_rmi_ChildSocket_getFileDescriptor__exec },
    { "getpeername", sidlx_rmi_ChildSocket_getpeername__exec },
    { "getsockname", sidlx_rmi_ChildSocket_getsockname__exec },
    { "init", sidlx_rmi_ChildSocket_init__exec },
    { "isSame", sidlx_rmi_ChildSocket_isSame__exec },
    { "isType", sidlx_rmi_ChildSocket_isType__exec },
    { "readint", sidlx_rmi_ChildSocket_readint__exec },
    { "readline", sidlx_rmi_ChildSocket_readline__exec },
    { "readn", sidlx_rmi_ChildSocket_readn__exec },
    { "readstring", sidlx_rmi_ChildSocket_readstring__exec },
    { "readstring_alloc", sidlx_rmi_ChildSocket_readstring_alloc__exec },
    { "setFileDescriptor", sidlx_rmi_ChildSocket_setFileDescriptor__exec },
    { "test", sidlx_rmi_ChildSocket_test__exec },
    { "writeint", sidlx_rmi_ChildSocket_writeint__exec },
    { "writen", sidlx_rmi_ChildSocket_writen__exec },
    { "writestring", sidlx_rmi_ChildSocket_writestring__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_rmi_ChildSocket__method);
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

static void sidlx_rmi_ChildSocket__init_epv(void)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_rmi_ChildSocket__epv*  epv  = &s_new_epv__sidlx_rmi_childsocket;
  struct sidlx_rmi_ChildSocket__epv*  hepv = 
    &s_new_epv_hooks__sidlx_rmi_childsocket;
  struct sidl_BaseClass__epv*         e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseClass__epv*         he0  = &s_new_epv_hooks__sidl_baseclass;
  struct sidl_BaseInterface__epv*     e1   = &s_new_epv__sidl_baseinterface;
  struct sidl_BaseInterface__epv*     he1  = 
    &s_new_epv_hooks__sidl_baseinterface;
  struct sidlx_rmi_IPv4Socket__epv*   e2   = &s_new_epv__sidlx_rmi_ipv4socket;
  struct sidlx_rmi_IPv4Socket__epv*   he2  = 
    &s_new_epv_hooks__sidlx_rmi_ipv4socket;
  struct sidlx_rmi_Socket__epv*       e3   = &s_new_epv__sidlx_rmi_socket;
  struct sidlx_rmi_Socket__epv*       he3  = &s_new_epv_hooks__sidlx_rmi_socket;

  struct sidlx_rmi_IPv4Socket__epv*  s1 = NULL;
  struct sidlx_rmi_IPv4Socket__epv*  h1 = NULL;
  struct sidl_BaseClass__epv*        s2 = NULL;
  struct sidl_BaseClass__epv*        h2 = NULL;

  sidlx_rmi_IPv4Socket__getEPVs(
    &s_old_epv__sidl_baseinterface,
    &s_old_epv_hooks__sidl_baseinterface,
    &s_old_epv__sidl_baseclass,&s_old_epv_hooks__sidl_baseclass,
    &s_old_epv__sidlx_rmi_socket,
    &s_old_epv_hooks__sidlx_rmi_socket,
    &s_old_epv__sidlx_rmi_ipv4socket,&s_old_epv_hooks__sidlx_rmi_ipv4socket);
  /*
   * Here we alias the static epvs to some handy small names
   */

  s2  =  s_old_epv__sidl_baseclass;
  h2  =  s_old_epv_hooks__sidl_baseclass;
  s1  =  s_old_epv__sidlx_rmi_ipv4socket;
  h1  =  s_old_epv_hooks__sidlx_rmi_ipv4socket;

  epv->f__cast                    = ior_sidlx_rmi_ChildSocket__cast;
  epv->f__delete                  = ior_sidlx_rmi_ChildSocket__delete;
  epv->f__exec                    = ior_sidlx_rmi_ChildSocket__exec;
  epv->f__getURL                  = ior_sidlx_rmi_ChildSocket__getURL;
  epv->f__raddRef                 = ior_sidlx_rmi_ChildSocket__raddRef;
  epv->f__isRemote                = ior_sidlx_rmi_ChildSocket__isRemote;
  epv->f__set_hooks               = ior_sidlx_rmi_ChildSocket__set_hooks;
  epv->f__ctor                    = NULL;
  epv->f__ctor2                   = NULL;
  epv->f__dtor                    = NULL;
  epv->f_init                     = NULL;
  epv->f_getsockname              = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t*,int32_t*,struct 
    sidl_BaseInterface__object **)) s1->f_getsockname;
  epv->f_getpeername              = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t*,int32_t*,struct 
    sidl_BaseInterface__object **)) s1->f_getpeername;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_BaseInterface__object **)) 
    s1->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_BaseInterface__object **)) 
    s1->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_BaseInterface__object*,struct 
    sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_rmi_ChildSocket__object*,const char*,struct 
    sidl_BaseInterface__object **)) s1->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getClassInfo;
  epv->f_close                    = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_BaseInterface__object **)) 
    s1->f_close;
  epv->f_readn                    = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_char__array**,struct 
    sidl_BaseInterface__object **)) s1->f_readn;
  epv->f_readline                 = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_char__array**,struct 
    sidl_BaseInterface__object **)) s1->f_readline;
  epv->f_readstring               = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_char__array**,struct 
    sidl_BaseInterface__object **)) s1->f_readstring;
  epv->f_readstring_alloc         = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_char__array**,struct 
    sidl_BaseInterface__object **)) s1->f_readstring_alloc;
  epv->f_readint                  = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t*,struct sidl_BaseInterface__object 
    **)) s1->f_readint;
  epv->f_writen                   = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_char__array*,struct 
    sidl_BaseInterface__object **)) s1->f_writen;
  epv->f_writestring              = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_char__array*,struct 
    sidl_BaseInterface__object **)) s1->f_writestring;
  epv->f_writeint                 = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_BaseInterface__object 
    **)) s1->f_writeint;
  epv->f_setFileDescriptor        = (void (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,struct sidl_BaseInterface__object 
    **)) s1->f_setFileDescriptor;
  epv->f_getFileDescriptor        = (int32_t (*)(struct 
    sidlx_rmi_ChildSocket__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getFileDescriptor;
  epv->f_test                     = (sidl_bool (*)(struct 
    sidlx_rmi_ChildSocket__object*,int32_t,int32_t,struct 
    sidl_BaseInterface__object **)) s1->f_test;

  sidlx_rmi_ChildSocket__set_epv(epv);

  memcpy((void*)hepv, epv, sizeof(struct sidlx_rmi_ChildSocket__epv));
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

  e2->f__cast               = (void* (*)(struct sidlx_rmi_IPv4Socket__object*,
    const char*, struct sidl_BaseInterface__object**)) epv->f__cast;
  e2->f__delete             = (void (*)(struct sidlx_rmi_IPv4Socket__object*, 
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e2->f__getURL             = (char* (*)(struct sidlx_rmi_IPv4Socket__object*, 
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e2->f__raddRef            = (void (*)(struct sidlx_rmi_IPv4Socket__object*, 
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e2->f__isRemote           = (sidl_bool (*)(struct 
    sidlx_rmi_IPv4Socket__object*, struct sidl_BaseInterface__object **)) 
    epv->f__isRemote;
  e2->f__exec               = (void (*)(struct sidlx_rmi_IPv4Socket__object*,
    const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_getsockname         = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t*,int32_t*,struct sidl_BaseInterface__object **)) epv->f_getsockname;
  e2->f_getpeername         = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t*,int32_t*,struct sidl_BaseInterface__object **)) epv->f_getpeername;
  e2->f_addRef              = (void (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(struct 
    sidlx_rmi_IPv4Socket__object*,struct sidl_BaseInterface__object*,struct 
    sidl_BaseInterface__object **)) epv->f_isSame;
  e2->f_isType              = (sidl_bool (*)(struct 
    sidlx_rmi_IPv4Socket__object*,const char*,struct sidl_BaseInterface__object 
    **)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_IPv4Socket__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;
  e2->f_close               = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object **)) epv->f_close;
  e2->f_readn               = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array**,struct sidl_BaseInterface__object **)) 
    epv->f_readn;
  e2->f_readline            = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array**,struct sidl_BaseInterface__object **)) 
    epv->f_readline;
  e2->f_readstring          = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array**,struct sidl_BaseInterface__object **)) 
    epv->f_readstring;
  e2->f_readstring_alloc    = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_char__array**,struct sidl_BaseInterface__object **)) 
    epv->f_readstring_alloc;
  e2->f_readint             = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t*,struct sidl_BaseInterface__object **)) epv->f_readint;
  e2->f_writen              = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array*,struct sidl_BaseInterface__object **)) 
    epv->f_writen;
  e2->f_writestring         = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_char__array*,struct sidl_BaseInterface__object **)) 
    epv->f_writestring;
  e2->f_writeint            = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_writeint;
  e2->f_setFileDescriptor   = (void (*)(struct sidlx_rmi_IPv4Socket__object*,
    int32_t,struct sidl_BaseInterface__object **)) epv->f_setFileDescriptor;
  e2->f_getFileDescriptor   = (int32_t (*)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object **)) epv->f_getFileDescriptor;
  e2->f_test                = (sidl_bool (*)(struct 
    sidlx_rmi_IPv4Socket__object*,int32_t,int32_t,struct 
    sidl_BaseInterface__object **)) epv->f_test;

  memcpy((void*) he2, e2, sizeof(struct sidlx_rmi_IPv4Socket__epv));

  e3->f__cast               = (void* (*)(void*,const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e3->f__delete             = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e3->f__getURL             = (char* (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e3->f__raddRef            = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e3->f__isRemote           = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__isRemote;
  e3->f__exec               = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_close               = (int32_t (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_close;
  e3->f_readn               = (int32_t (*)(void*,int32_t,struct 
    sidl_char__array**,struct sidl_BaseInterface__object **)) epv->f_readn;
  e3->f_readline            = (int32_t (*)(void*,int32_t,struct 
    sidl_char__array**,struct sidl_BaseInterface__object **)) epv->f_readline;
  e3->f_readstring          = (int32_t (*)(void*,int32_t,struct 
    sidl_char__array**,struct sidl_BaseInterface__object **)) epv->f_readstring;
  e3->f_readstring_alloc    = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring_alloc;
  e3->f_readint             = (int32_t (*)(void*,int32_t*,struct 
    sidl_BaseInterface__object **)) epv->f_readint;
  e3->f_writen              = (int32_t (*)(void*,int32_t,struct 
    sidl_char__array*,struct sidl_BaseInterface__object **)) epv->f_writen;
  e3->f_writestring         = (int32_t (*)(void*,int32_t,struct 
    sidl_char__array*,struct sidl_BaseInterface__object **)) epv->f_writestring;
  e3->f_writeint            = (int32_t (*)(void*,int32_t,struct 
    sidl_BaseInterface__object **)) epv->f_writeint;
  e3->f_setFileDescriptor   = (void (*)(void*,int32_t,struct 
    sidl_BaseInterface__object **)) epv->f_setFileDescriptor;
  e3->f_getFileDescriptor   = (int32_t (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getFileDescriptor;
  e3->f_test                = (sidl_bool (*)(void*,int32_t,int32_t,struct 
    sidl_BaseInterface__object **)) epv->f_test;
  e3->f_addRef              = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_addRef;
  e3->f_deleteRef           = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e3->f_isType              = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he3, e3, sizeof(struct sidlx_rmi_Socket__epv));

  s_method_initialized = 1;
  ior_sidlx_rmi_ChildSocket__ensure_load_called();
}

void sidlx_rmi_ChildSocket__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidlx_rmi_Socket__epv **s_arg_epv__sidlx_rmi_socket,
  struct sidlx_rmi_Socket__epv **s_arg_epv_hooks__sidlx_rmi_socket,
  struct sidlx_rmi_IPv4Socket__epv **s_arg_epv__sidlx_rmi_ipv4socket,struct 
    sidlx_rmi_IPv4Socket__epv **s_arg_epv_hooks__sidlx_rmi_ipv4socket,
  struct sidlx_rmi_ChildSocket__epv **s_arg_epv__sidlx_rmi_childsocket,struct 
    sidlx_rmi_ChildSocket__epv **s_arg_epv_hooks__sidlx_rmi_childsocket)
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_ChildSocket__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_new_epv__sidl_baseinterface;
  *s_arg_epv_hooks__sidl_baseinterface = &s_new_epv_hooks__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_new_epv__sidl_baseclass;
  *s_arg_epv_hooks__sidl_baseclass = &s_new_epv_hooks__sidl_baseclass;
  *s_arg_epv__sidlx_rmi_socket = &s_new_epv__sidlx_rmi_socket;
  *s_arg_epv_hooks__sidlx_rmi_socket = &s_new_epv_hooks__sidlx_rmi_socket;
  *s_arg_epv__sidlx_rmi_ipv4socket = &s_new_epv__sidlx_rmi_ipv4socket;
  *s_arg_epv_hooks__sidlx_rmi_ipv4socket = 
    &s_new_epv_hooks__sidlx_rmi_ipv4socket;
  *s_arg_epv__sidlx_rmi_childsocket = &s_new_epv__sidlx_rmi_childsocket;
  *s_arg_epv_hooks__sidlx_rmi_childsocket = 
    &s_new_epv_hooks__sidlx_rmi_childsocket;
}
/*
 * SUPER: returns parent's non-overrided EPV
 */

static struct sidlx_rmi_IPv4Socket__epv* sidlx_rmi_ChildSocket__super(void) {
  return s_old_epv__sidlx_rmi_ipv4socket;
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
      sidl_ClassInfoI_setName(impl, "sidlx.rmi.ChildSocket",_ex);
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
initMetadata(struct sidlx_rmi_ChildSocket__object* self, sidl_BaseInterface* 
  _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((
      *self).d_sidlx_rmi_ipv4socket.d_sidl_baseclass.d_data);
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

struct sidlx_rmi_ChildSocket__object*
sidlx_rmi_ChildSocket__new(void* ddata, struct sidl_BaseInterface__object ** 
  _ex)
{
  struct sidlx_rmi_ChildSocket__object* self =
    (struct sidlx_rmi_ChildSocket__object*) malloc(
      sizeof(struct sidlx_rmi_ChildSocket__object));
  *_ex = NULL; /* default to no exception */
  sidlx_rmi_ChildSocket__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_rmi_ChildSocket__init(
  struct sidlx_rmi_ChildSocket__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidlx_rmi_ChildSocket__object* s0 = self;
  struct sidlx_rmi_IPv4Socket__object*  s1 = &s0->d_sidlx_rmi_ipv4socket;
  struct sidl_BaseClass__object*        s2 = &s1->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_ChildSocket__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidlx_rmi_IPv4Socket__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

  s2->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s2->d_epv                      = &s_new_epv__sidl_baseclass;

  s1->d_sidlx_rmi_socket.d_epv = &s_new_epv__sidlx_rmi_socket;
  s1->d_epv                    = &s_new_epv__sidlx_rmi_ipv4socket;

  s0->d_epv    = &s_new_epv__sidlx_rmi_childsocket;

  s0->d_data = NULL;

  ior_sidlx_rmi_ChildSocket__set_hooks(s0, FALSE, _ex);

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

void sidlx_rmi_ChildSocket__fini(
  struct sidlx_rmi_ChildSocket__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidlx_rmi_ChildSocket__object* s0 = self;
  struct sidlx_rmi_IPv4Socket__object*  s1 = &s0->d_sidlx_rmi_ipv4socket;
  struct sidl_BaseClass__object*        s2 = &s1->d_sidl_baseclass;

  *_ex = NULL; /* default to no exception */
  (*(s0->d_epv->f__dtor))(s0,_ex);
  SIDL_CHECK(*_ex);

  s2->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s2->d_epv                      = s_old_epv__sidl_baseclass;

  s1->d_sidlx_rmi_socket.d_epv = s_old_epv__sidlx_rmi_socket;
  s1->d_epv                    = s_old_epv__sidlx_rmi_ipv4socket;

  sidlx_rmi_IPv4Socket__fini(s1, _ex); SIDL_CHECK(*_ex);
  EXIT:
  return;
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_rmi_ChildSocket__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_rmi_ChildSocket__external
s_externalEntryPoints = {
  sidlx_rmi_ChildSocket__new,
  sidlx_rmi_ChildSocket__super,
  1, 
  0
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_ChildSocket__external*
sidlx_rmi_ChildSocket__externals(void)
{
  return &s_externalEntryPoints;
}

