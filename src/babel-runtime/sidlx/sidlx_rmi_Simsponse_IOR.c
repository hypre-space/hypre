/*
 * File:          sidlx_rmi_Simsponse_IOR.c
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Intermediate Object Representation for sidlx.rmi.Simsponse
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
#include "sidlx_rmi_Simsponse_IOR.h"
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
static struct sidl_recursive_mutex_t sidlx_rmi_Simsponse__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_Simsponse__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_Simsponse__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_Simsponse__mutex )==EDEADLOCK) */
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

static struct sidlx_rmi_Simsponse__epv s_new_epv__sidlx_rmi_simsponse;

static struct sidlx_rmi_Simsponse__epv s_new_epv_hooks__sidlx_rmi_simsponse;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv  s_new_epv_hooks__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv_hooks__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_new_epv_hooks__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv_hooks__sidl_baseinterface;

static struct sidl_io_Deserializer__epv s_new_epv__sidl_io_deserializer;
static struct sidl_io_Deserializer__epv s_new_epv_hooks__sidl_io_deserializer;

static struct sidl_rmi_Response__epv s_new_epv__sidl_rmi_response;
static struct sidl_rmi_Response__epv s_new_epv_hooks__sidl_rmi_response;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_rmi_Simsponse__set_epv(
  struct sidlx_rmi_Simsponse__epv* epv);
extern void sidlx_rmi_Simsponse__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_rmi_Simsponse_init__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* methodName= NULL;
  char* objectid= NULL;
  char* sock_str = NULL;
  struct sidlx_rmi_Socket__object* sock = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "methodName", &methodName, 
    _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "objectid", &objectid, _ex);SIDL_CHECK(
    *_ex);
  sidl_rmi_Call_unpackString( inArgs, "sock", &sock_str, _ex);SIDL_CHECK(*_ex);
  sock = skel_sidlx_rmi_Simsponse_fconnect_sidlx_rmi_Socket(sock_str, TRUE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_init)(
    self,
    methodName,
    objectid,
    sock,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(methodName) {free(methodName);}
  if(objectid) {free(objectid);}
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
sidlx_rmi_Simsponse_test__exec(
        struct sidlx_rmi_Simsponse__object* self,
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

static void
sidlx_rmi_Simsponse_pullData__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_pullData)(
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
sidlx_rmi_Simsponse_getMethodName__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getMethodName)(
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
sidlx_rmi_Simsponse_getObjectID__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getObjectID)(
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
sidlx_rmi_Simsponse_addRef__exec(
        struct sidlx_rmi_Simsponse__object* self,
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
sidlx_rmi_Simsponse_deleteRef__exec(
        struct sidlx_rmi_Simsponse__object* self,
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
sidlx_rmi_Simsponse_isSame__exec(
        struct sidlx_rmi_Simsponse__object* self,
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
  iobj = skel_sidlx_rmi_Simsponse_fconnect_sidl_BaseInterface(iobj_str, TRUE, 
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
sidlx_rmi_Simsponse_isType__exec(
        struct sidlx_rmi_Simsponse__object* self,
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
sidlx_rmi_Simsponse_getClassInfo__exec(
        struct sidlx_rmi_Simsponse__object* self,
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
sidlx_rmi_Simsponse_getExceptionThrown__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  struct sidl_BaseException__object* _retval = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getExceptionThrown)(
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
sidlx_rmi_Simsponse_unpackBool__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  sidl_bool value_data = FALSE;
  sidl_bool* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackBool)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packBool( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackChar__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  char value_data = 0;
  char* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackChar)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packChar( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackInt__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  int32_t value_data = 0;
  int32_t* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackInt)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packInt( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackLong__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  int64_t value_data = 0;
  int64_t* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackLong)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packLong( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackOpaque__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  void* value_data = 0;
  void** value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackOpaque)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packOpaque( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackFloat__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  float value_data = 0;
  float* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackFloat)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packFloat( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackDouble__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  double value_data = 0;
  double* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackDouble)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packDouble( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackFcomplex__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_fcomplex value_data = { 0, 0 };
  struct sidl_fcomplex* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackFcomplex)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packFcomplex( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackDcomplex__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_dcomplex value_data = { 0, 0 };
  struct sidl_dcomplex* value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackDcomplex)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packDcomplex( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
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
sidlx_rmi_Simsponse_unpackString__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  char* value_data = 0;
  char** value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackString)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packString( outArgs, "value", *value, _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  if(*value) {free(*value);}
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
sidlx_rmi_Simsponse_unpackSerializable__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_io_Serializable__object* value_data = NULL;
  struct sidl_io_Serializable__object** value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackSerializable)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  if(*value){
    char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*value, 
      _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Return_packString( outArgs, "value", _url, _ex);SIDL_CHECK(*_ex);
    free((void*)_url);
  } else {
    sidl_rmi_Return_packString( outArgs, "value", NULL, _ex);SIDL_CHECK(*_ex);
  }
  /* clean-up dangling references */
  if(key) {free(key);}
  if(*value && sidl_BaseInterface__isRemote((sidl_BaseInterface)*value, _ex)) {
    (*((sidl_BaseInterface)*value)->d_epv->f__raddRef)(((
      sidl_BaseInterface)*value)->d_object, _ex); SIDL_CHECK(*_ex);
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)*value, _ex); SIDL_CHECK(
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
sidlx_rmi_Simsponse_unpackBoolArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_bool__array* value_data = NULL;
  struct sidl_bool__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackBoolArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packBoolArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackCharArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_char__array* value_data = NULL;
  struct sidl_char__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackCharArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packCharArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackIntArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_int__array* value_data = NULL;
  struct sidl_int__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackIntArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packIntArray( outArgs, "value", *value,0,0,0, _ex);SIDL_CHECK(
    *_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackLongArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_long__array* value_data = NULL;
  struct sidl_long__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackLongArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packLongArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackOpaqueArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_opaque__array* value_data = NULL;
  struct sidl_opaque__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackOpaqueArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packOpaqueArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackFloatArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_float__array* value_data = NULL;
  struct sidl_float__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackFloatArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packFloatArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackDoubleArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_double__array* value_data = NULL;
  struct sidl_double__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackDoubleArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packDoubleArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackFcomplexArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_fcomplex__array* value_data = NULL;
  struct sidl_fcomplex__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackFcomplexArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packFcomplexArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackDcomplexArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_dcomplex__array* value_data = NULL;
  struct sidl_dcomplex__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackDcomplexArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packDcomplexArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackStringArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_string__array* value_data = NULL;
  struct sidl_string__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackStringArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packStringArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackGenericArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl__array* value_data = NULL;
  struct sidl__array** value = &value_data;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackGenericArray)(
    self,
    key,
    value,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packGenericArray( outArgs, "value", *value,0, _ex);SIDL_CHECK(
    *_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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
sidlx_rmi_Simsponse_unpackSerializableArray__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_io_Serializable__array* value_data = NULL;
  struct sidl_io_Serializable__array** value = &value_data;
  int32_t ordering = 0;
  int32_t dimen = 0;
  sidl_bool isRarray = FALSE;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "key", &key, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "ordering", &ordering, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackInt( inArgs, "dimen", &dimen, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackBool( inArgs, "isRarray", &isRarray, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_unpackSerializableArray)(
    self,
    key,
    value,
    ordering,
    dimen,
    isRarray,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  sidl_rmi_Return_packSerializableArray( outArgs, "value", *value,0,0,0, 
    _ex);SIDL_CHECK(*_ex);
  /* clean-up dangling references */
  if(key) {free(key);}
  sidl__array_deleteRef((struct sidl__array*)*value);
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

static void ior_sidlx_rmi_Simsponse__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_rmi_Simsponse__call_load();
    s_load_called=1;
  }
}

/* CAST: dynamic type casting support. */
static void* ior_sidlx_rmi_Simsponse__cast(
  struct sidlx_rmi_Simsponse__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.io.Deserializer");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_sidl_io_deserializer);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
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
    cmp1 = strcmp(name, "sidlx.rmi.Simsponse");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct sidlx_rmi_Simsponse__object*)self);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.rmi.Response");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_rmi_response);
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

static void ior_sidlx_rmi_Simsponse__set_hooks(
  struct sidlx_rmi_Simsponse__object* self,
  int on, struct sidl_BaseInterface__object **_ex ) { 
  *_ex = NULL;
  /*
   * Nothing else to do since hooks support not needed.
   */

}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidlx_rmi_Simsponse__delete(
  struct sidlx_rmi_Simsponse__object* self, struct sidl_BaseInterface__object 
    **_ex)
{
  *_ex = NULL; /* default to no exception */
  sidlx_rmi_Simsponse__fini(self,_ex);
  memset((void*)self, 0, sizeof(struct sidlx_rmi_Simsponse__object));
  free((void*) self);
}

static char*
ior_sidlx_rmi_Simsponse__getURL(
    struct sidlx_rmi_Simsponse__object* self,
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
ior_sidlx_rmi_Simsponse__raddRef(
    struct sidlx_rmi_Simsponse__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_sidlx_rmi_Simsponse__isRemote(
    struct sidlx_rmi_Simsponse__object* self, sidl_BaseInterface* _ex) {
  *_ex = NULL; /* default to no exception */
  return FALSE;
}

struct sidlx_rmi_Simsponse__method {
  const char *d_name;
  void (*d_func)(struct sidlx_rmi_Simsponse__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

static void
ior_sidlx_rmi_Simsponse__exec(
    struct sidlx_rmi_Simsponse__object* self,
    const char* methodName,
    struct sidl_rmi_Call__object* inArgs,
    struct sidl_rmi_Return__object* outArgs,
    struct sidl_BaseInterface__object **_ex ) { 
  static const struct sidlx_rmi_Simsponse__method  s_methods[] = {
    { "addRef", sidlx_rmi_Simsponse_addRef__exec },
    { "deleteRef", sidlx_rmi_Simsponse_deleteRef__exec },
    { "getClassInfo", sidlx_rmi_Simsponse_getClassInfo__exec },
    { "getExceptionThrown", sidlx_rmi_Simsponse_getExceptionThrown__exec },
    { "getMethodName", sidlx_rmi_Simsponse_getMethodName__exec },
    { "getObjectID", sidlx_rmi_Simsponse_getObjectID__exec },
    { "init", sidlx_rmi_Simsponse_init__exec },
    { "isSame", sidlx_rmi_Simsponse_isSame__exec },
    { "isType", sidlx_rmi_Simsponse_isType__exec },
    { "pullData", sidlx_rmi_Simsponse_pullData__exec },
    { "test", sidlx_rmi_Simsponse_test__exec },
    { "unpackBool", sidlx_rmi_Simsponse_unpackBool__exec },
    { "unpackBoolArray", sidlx_rmi_Simsponse_unpackBoolArray__exec },
    { "unpackChar", sidlx_rmi_Simsponse_unpackChar__exec },
    { "unpackCharArray", sidlx_rmi_Simsponse_unpackCharArray__exec },
    { "unpackDcomplex", sidlx_rmi_Simsponse_unpackDcomplex__exec },
    { "unpackDcomplexArray", sidlx_rmi_Simsponse_unpackDcomplexArray__exec },
    { "unpackDouble", sidlx_rmi_Simsponse_unpackDouble__exec },
    { "unpackDoubleArray", sidlx_rmi_Simsponse_unpackDoubleArray__exec },
    { "unpackFcomplex", sidlx_rmi_Simsponse_unpackFcomplex__exec },
    { "unpackFcomplexArray", sidlx_rmi_Simsponse_unpackFcomplexArray__exec },
    { "unpackFloat", sidlx_rmi_Simsponse_unpackFloat__exec },
    { "unpackFloatArray", sidlx_rmi_Simsponse_unpackFloatArray__exec },
    { "unpackGenericArray", sidlx_rmi_Simsponse_unpackGenericArray__exec },
    { "unpackInt", sidlx_rmi_Simsponse_unpackInt__exec },
    { "unpackIntArray", sidlx_rmi_Simsponse_unpackIntArray__exec },
    { "unpackLong", sidlx_rmi_Simsponse_unpackLong__exec },
    { "unpackLongArray", sidlx_rmi_Simsponse_unpackLongArray__exec },
    { "unpackOpaque", sidlx_rmi_Simsponse_unpackOpaque__exec },
    { "unpackOpaqueArray", sidlx_rmi_Simsponse_unpackOpaqueArray__exec },
    { "unpackSerializable", sidlx_rmi_Simsponse_unpackSerializable__exec },
    { "unpackSerializableArray", 
      sidlx_rmi_Simsponse_unpackSerializableArray__exec },
    { "unpackString", sidlx_rmi_Simsponse_unpackString__exec },
    { "unpackStringArray", sidlx_rmi_Simsponse_unpackStringArray__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_rmi_Simsponse__method);
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

static void sidlx_rmi_Simsponse__init_epv(void)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_rmi_Simsponse__epv*  epv  = &s_new_epv__sidlx_rmi_simsponse;
  struct sidlx_rmi_Simsponse__epv*  hepv = 
    &s_new_epv_hooks__sidlx_rmi_simsponse;
  struct sidl_BaseClass__epv*       e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseClass__epv*       he0  = &s_new_epv_hooks__sidl_baseclass;
  struct sidl_BaseInterface__epv*   e1   = &s_new_epv__sidl_baseinterface;
  struct sidl_BaseInterface__epv*   he1  = &s_new_epv_hooks__sidl_baseinterface;
  struct sidl_io_Deserializer__epv* e2   = &s_new_epv__sidl_io_deserializer;
  struct sidl_io_Deserializer__epv* he2  = 
    &s_new_epv_hooks__sidl_io_deserializer;
  struct sidl_rmi_Response__epv*    e3   = &s_new_epv__sidl_rmi_response;
  struct sidl_rmi_Response__epv*    he3  = &s_new_epv_hooks__sidl_rmi_response;

  struct sidl_BaseClass__epv*      s1 = NULL;
  struct sidl_BaseClass__epv*      h1 = NULL;

  sidl_BaseClass__getEPVs(
    &s_old_epv__sidl_baseinterface,
    &s_old_epv_hooks__sidl_baseinterface,
    &s_old_epv__sidl_baseclass,&s_old_epv_hooks__sidl_baseclass);
  /*
   * Here we alias the static epvs to some handy small names
   */

  s1  =  s_old_epv__sidl_baseclass;
  h1  =  s_old_epv_hooks__sidl_baseclass;

  epv->f__cast                        = ior_sidlx_rmi_Simsponse__cast;
  epv->f__delete                      = ior_sidlx_rmi_Simsponse__delete;
  epv->f__exec                        = ior_sidlx_rmi_Simsponse__exec;
  epv->f__getURL                      = ior_sidlx_rmi_Simsponse__getURL;
  epv->f__raddRef                     = ior_sidlx_rmi_Simsponse__raddRef;
  epv->f__isRemote                    = ior_sidlx_rmi_Simsponse__isRemote;
  epv->f__set_hooks                   = ior_sidlx_rmi_Simsponse__set_hooks;
  epv->f__ctor                        = NULL;
  epv->f__ctor2                       = NULL;
  epv->f__dtor                        = NULL;
  epv->f_init                         = NULL;
  epv->f_test                         = NULL;
  epv->f_pullData                     = NULL;
  epv->f_getMethodName                = NULL;
  epv->f_getObjectID                  = NULL;
  epv->f_addRef                       = (void (*)(struct 
    sidlx_rmi_Simsponse__object*,struct sidl_BaseInterface__object **)) 
    s1->f_addRef;
  epv->f_deleteRef                    = (void (*)(struct 
    sidlx_rmi_Simsponse__object*,struct sidl_BaseInterface__object **)) 
    s1->f_deleteRef;
  epv->f_isSame                       = (sidl_bool (*)(struct 
    sidlx_rmi_Simsponse__object*,struct sidl_BaseInterface__object*,struct 
    sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                       = (sidl_bool (*)(struct 
    sidlx_rmi_Simsponse__object*,const char*,struct sidl_BaseInterface__object 
    **)) s1->f_isType;
  epv->f_getClassInfo                 = (struct sidl_ClassInfo__object* (*)(
    struct sidlx_rmi_Simsponse__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getClassInfo;
  epv->f_getExceptionThrown           = NULL;
  epv->f_unpackBool                   = NULL;
  epv->f_unpackChar                   = NULL;
  epv->f_unpackInt                    = NULL;
  epv->f_unpackLong                   = NULL;
  epv->f_unpackOpaque                 = NULL;
  epv->f_unpackFloat                  = NULL;
  epv->f_unpackDouble                 = NULL;
  epv->f_unpackFcomplex               = NULL;
  epv->f_unpackDcomplex               = NULL;
  epv->f_unpackString                 = NULL;
  epv->f_unpackSerializable           = NULL;
  epv->f_unpackBoolArray              = NULL;
  epv->f_unpackCharArray              = NULL;
  epv->f_unpackIntArray               = NULL;
  epv->f_unpackLongArray              = NULL;
  epv->f_unpackOpaqueArray            = NULL;
  epv->f_unpackFloatArray             = NULL;
  epv->f_unpackDoubleArray            = NULL;
  epv->f_unpackFcomplexArray          = NULL;
  epv->f_unpackDcomplexArray          = NULL;
  epv->f_unpackStringArray            = NULL;
  epv->f_unpackGenericArray           = NULL;
  epv->f_unpackSerializableArray      = NULL;

  sidlx_rmi_Simsponse__set_epv(epv);

  memcpy((void*)hepv, epv, sizeof(struct sidlx_rmi_Simsponse__epv));
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

  e2->f__cast                   = (void* (*)(void*,const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e2->f__delete                 = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e2->f__getURL                 = (char* (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e2->f__raddRef                = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e2->f__isRemote               = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__isRemote;
  e2->f__exec                   = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e2->f_unpackBool              = (void (*)(void*,const char*,sidl_bool*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackBool;
  e2->f_unpackChar              = (void (*)(void*,const char*,char*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackChar;
  e2->f_unpackInt               = (void (*)(void*,const char*,int32_t*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackInt;
  e2->f_unpackLong              = (void (*)(void*,const char*,int64_t*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackLong;
  e2->f_unpackOpaque            = (void (*)(void*,const char*,void**,struct 
    sidl_BaseInterface__object **)) epv->f_unpackOpaque;
  e2->f_unpackFloat             = (void (*)(void*,const char*,float*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e2->f_unpackDouble            = (void (*)(void*,const char*,double*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e2->f_unpackFcomplex          = (void (*)(void*,const char*,struct 
    sidl_fcomplex*,struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e2->f_unpackDcomplex          = (void (*)(void*,const char*,struct 
    sidl_dcomplex*,struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e2->f_unpackString            = (void (*)(void*,const char*,char**,struct 
    sidl_BaseInterface__object **)) epv->f_unpackString;
  e2->f_unpackSerializable      = (void (*)(void*,const char*,struct 
    sidl_io_Serializable__object**,struct sidl_BaseInterface__object **)) 
    epv->f_unpackSerializable;
  e2->f_unpackBoolArray         = (void (*)(void*,const char*,struct 
    sidl_bool__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackBoolArray;
  e2->f_unpackCharArray         = (void (*)(void*,const char*,struct 
    sidl_char__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackCharArray;
  e2->f_unpackIntArray          = (void (*)(void*,const char*,struct 
    sidl_int__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackIntArray;
  e2->f_unpackLongArray         = (void (*)(void*,const char*,struct 
    sidl_long__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackLongArray;
  e2->f_unpackOpaqueArray       = (void (*)(void*,const char*,struct 
    sidl_opaque__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackOpaqueArray;
  e2->f_unpackFloatArray        = (void (*)(void*,const char*,struct 
    sidl_float__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackFloatArray;
  e2->f_unpackDoubleArray       = (void (*)(void*,const char*,struct 
    sidl_double__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackDoubleArray;
  e2->f_unpackFcomplexArray     = (void (*)(void*,const char*,struct 
    sidl_fcomplex__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackFcomplexArray;
  e2->f_unpackDcomplexArray     = (void (*)(void*,const char*,struct 
    sidl_dcomplex__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackDcomplexArray;
  e2->f_unpackStringArray       = (void (*)(void*,const char*,struct 
    sidl_string__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackStringArray;
  e2->f_unpackGenericArray      = (void (*)(void*,const char*,struct 
    sidl__array**,struct sidl_BaseInterface__object **)) 
    epv->f_unpackGenericArray;
  e2->f_unpackSerializableArray = (void (*)(void*,const char*,struct 
    sidl_io_Serializable__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackSerializableArray;
  e2->f_addRef                  = (void (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_addRef;
  e2->f_deleteRef               = (void (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_deleteRef;
  e2->f_isSame                  = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e2->f_isType                  = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e2->f_getClassInfo            = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he2, e2, sizeof(struct sidl_io_Deserializer__epv));

  e3->f__cast                   = (void* (*)(void*,const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e3->f__delete                 = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e3->f__getURL                 = (char* (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e3->f__raddRef                = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e3->f__isRemote               = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__isRemote;
  e3->f__exec                   = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_getExceptionThrown      = (struct sidl_BaseException__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getExceptionThrown;
  e3->f_unpackBool              = (void (*)(void*,const char*,sidl_bool*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackBool;
  e3->f_unpackChar              = (void (*)(void*,const char*,char*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackChar;
  e3->f_unpackInt               = (void (*)(void*,const char*,int32_t*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackInt;
  e3->f_unpackLong              = (void (*)(void*,const char*,int64_t*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackLong;
  e3->f_unpackOpaque            = (void (*)(void*,const char*,void**,struct 
    sidl_BaseInterface__object **)) epv->f_unpackOpaque;
  e3->f_unpackFloat             = (void (*)(void*,const char*,float*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e3->f_unpackDouble            = (void (*)(void*,const char*,double*,struct 
    sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e3->f_unpackFcomplex          = (void (*)(void*,const char*,struct 
    sidl_fcomplex*,struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e3->f_unpackDcomplex          = (void (*)(void*,const char*,struct 
    sidl_dcomplex*,struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e3->f_unpackString            = (void (*)(void*,const char*,char**,struct 
    sidl_BaseInterface__object **)) epv->f_unpackString;
  e3->f_unpackSerializable      = (void (*)(void*,const char*,struct 
    sidl_io_Serializable__object**,struct sidl_BaseInterface__object **)) 
    epv->f_unpackSerializable;
  e3->f_unpackBoolArray         = (void (*)(void*,const char*,struct 
    sidl_bool__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackBoolArray;
  e3->f_unpackCharArray         = (void (*)(void*,const char*,struct 
    sidl_char__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackCharArray;
  e3->f_unpackIntArray          = (void (*)(void*,const char*,struct 
    sidl_int__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackIntArray;
  e3->f_unpackLongArray         = (void (*)(void*,const char*,struct 
    sidl_long__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackLongArray;
  e3->f_unpackOpaqueArray       = (void (*)(void*,const char*,struct 
    sidl_opaque__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackOpaqueArray;
  e3->f_unpackFloatArray        = (void (*)(void*,const char*,struct 
    sidl_float__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackFloatArray;
  e3->f_unpackDoubleArray       = (void (*)(void*,const char*,struct 
    sidl_double__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackDoubleArray;
  e3->f_unpackFcomplexArray     = (void (*)(void*,const char*,struct 
    sidl_fcomplex__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackFcomplexArray;
  e3->f_unpackDcomplexArray     = (void (*)(void*,const char*,struct 
    sidl_dcomplex__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackDcomplexArray;
  e3->f_unpackStringArray       = (void (*)(void*,const char*,struct 
    sidl_string__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackStringArray;
  e3->f_unpackGenericArray      = (void (*)(void*,const char*,struct 
    sidl__array**,struct sidl_BaseInterface__object **)) 
    epv->f_unpackGenericArray;
  e3->f_unpackSerializableArray = (void (*)(void*,const char*,struct 
    sidl_io_Serializable__array**,int32_t,int32_t,sidl_bool,struct 
    sidl_BaseInterface__object **)) epv->f_unpackSerializableArray;
  e3->f_addRef                  = (void (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_addRef;
  e3->f_deleteRef               = (void (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_deleteRef;
  e3->f_isSame                  = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e3->f_isType                  = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo            = (struct sidl_ClassInfo__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he3, e3, sizeof(struct sidl_rmi_Response__epv));

  s_method_initialized = 1;
  ior_sidlx_rmi_Simsponse__ensure_load_called();
}

void sidlx_rmi_Simsponse__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_io_Deserializer__epv **s_arg_epv__sidl_io_deserializer,
  struct sidl_io_Deserializer__epv **s_arg_epv_hooks__sidl_io_deserializer,
  struct sidl_rmi_Response__epv **s_arg_epv__sidl_rmi_response,
  struct sidl_rmi_Response__epv **s_arg_epv_hooks__sidl_rmi_response,
  struct sidlx_rmi_Simsponse__epv **s_arg_epv__sidlx_rmi_simsponse,struct 
    sidlx_rmi_Simsponse__epv **s_arg_epv_hooks__sidlx_rmi_simsponse)
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_Simsponse__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_new_epv__sidl_baseinterface;
  *s_arg_epv_hooks__sidl_baseinterface = &s_new_epv_hooks__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_new_epv__sidl_baseclass;
  *s_arg_epv_hooks__sidl_baseclass = &s_new_epv_hooks__sidl_baseclass;
  *s_arg_epv__sidl_io_deserializer = &s_new_epv__sidl_io_deserializer;
  *s_arg_epv_hooks__sidl_io_deserializer = 
    &s_new_epv_hooks__sidl_io_deserializer;
  *s_arg_epv__sidl_rmi_response = &s_new_epv__sidl_rmi_response;
  *s_arg_epv_hooks__sidl_rmi_response = &s_new_epv_hooks__sidl_rmi_response;
  *s_arg_epv__sidlx_rmi_simsponse = &s_new_epv__sidlx_rmi_simsponse;
  *s_arg_epv_hooks__sidlx_rmi_simsponse = &s_new_epv_hooks__sidlx_rmi_simsponse;
}
/*
 * SUPER: returns parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidlx_rmi_Simsponse__super(void) {
  return s_old_epv__sidl_baseclass;
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
      sidl_ClassInfoI_setName(impl, "sidlx.rmi.Simsponse",_ex);
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
initMetadata(struct sidlx_rmi_Simsponse__object* self, sidl_BaseInterface* _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((
      *self).d_sidl_baseclass.d_data);
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

struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__new(void* ddata, struct sidl_BaseInterface__object ** _ex)
{
  struct sidlx_rmi_Simsponse__object* self =
    (struct sidlx_rmi_Simsponse__object*) malloc(
      sizeof(struct sidlx_rmi_Simsponse__object));
  *_ex = NULL; /* default to no exception */
  sidlx_rmi_Simsponse__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_rmi_Simsponse__init(
  struct sidlx_rmi_Simsponse__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidlx_rmi_Simsponse__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_Simsponse__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidl_BaseClass__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_sidl_io_deserializer.d_epv = &s_new_epv__sidl_io_deserializer;
  s0->d_sidl_rmi_response.d_epv    = &s_new_epv__sidl_rmi_response;
  s0->d_epv                        = &s_new_epv__sidlx_rmi_simsponse;

  s0->d_sidl_io_deserializer.d_object = self;

  s0->d_sidl_rmi_response.d_object = self;

  s0->d_data = NULL;

  ior_sidlx_rmi_Simsponse__set_hooks(s0, FALSE, _ex);

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

void sidlx_rmi_Simsponse__fini(
  struct sidlx_rmi_Simsponse__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidlx_rmi_Simsponse__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  *_ex = NULL; /* default to no exception */
  (*(s0->d_epv->f__dtor))(s0,_ex);
  SIDL_CHECK(*_ex);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1, _ex); SIDL_CHECK(*_ex);
  EXIT:
  return;
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_rmi_Simsponse__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_rmi_Simsponse__external
s_externalEntryPoints = {
  sidlx_rmi_Simsponse__new,
  sidlx_rmi_Simsponse__super,
  1, 
  0
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_Simsponse__external*
sidlx_rmi_Simsponse__externals(void)
{
  return &s_externalEntryPoints;
}

