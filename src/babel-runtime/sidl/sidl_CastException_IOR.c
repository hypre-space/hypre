/*
 * File:          sidl_CastException_IOR.c
 * Symbol:        sidl.CastException-v0.9.15
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Release:       $Name: V2-4-0b $
 * Revision:      @(#) $Id: sidl_CastException_IOR.c,v 1.2 2007/09/27 19:35:42 painter Exp $
 * Description:   Intermediate Object Representation for sidl.CastException
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
#include "sidl_CastException_IOR.h"
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
static struct sidl_recursive_mutex_t sidl_CastException__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_CastException__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_CastException__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_CastException__mutex )==EDEADLOCK) */
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

static struct sidl_CastException__epv s_new_epv__sidl_castexception;

static struct sidl_CastException__epv s_new_epv_hooks__sidl_castexception;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv  s_new_epv_hooks__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv_hooks__sidl_baseclass;

static struct sidl_BaseException__epv  s_new_epv__sidl_baseexception;
static struct sidl_BaseException__epv  s_new_epv_hooks__sidl_baseexception;
static struct sidl_BaseException__epv* s_old_epv__sidl_baseexception;
static struct sidl_BaseException__epv* s_old_epv_hooks__sidl_baseexception;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv  s_new_epv_hooks__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv_hooks__sidl_baseinterface;

static struct sidl_RuntimeException__epv s_new_epv__sidl_runtimeexception;
static struct sidl_RuntimeException__epv s_new_epv_hooks__sidl_runtimeexception;

static struct sidl_SIDLException__epv  s_new_epv__sidl_sidlexception;
static struct sidl_SIDLException__epv  s_new_epv_hooks__sidl_sidlexception;
static struct sidl_SIDLException__epv* s_old_epv__sidl_sidlexception;
static struct sidl_SIDLException__epv* s_old_epv_hooks__sidl_sidlexception;

static struct sidl_io_Serializable__epv  s_new_epv__sidl_io_serializable;
static struct sidl_io_Serializable__epv  s_new_epv_hooks__sidl_io_serializable;
static struct sidl_io_Serializable__epv* s_old_epv__sidl_io_serializable;
static struct sidl_io_Serializable__epv* s_old_epv_hooks__sidl_io_serializable;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidl_CastException__set_epv(
  struct sidl_CastException__epv* epv);
extern void sidl_CastException__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidl_CastException_addRef__exec(
        struct sidl_CastException__object* self,
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
sidl_CastException_deleteRef__exec(
        struct sidl_CastException__object* self,
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
sidl_CastException_isSame__exec(
        struct sidl_CastException__object* self,
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
  iobj = skel_sidl_CastException_fconnect_sidl_BaseInterface(iobj_str, TRUE, 
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
sidl_CastException_isType__exec(
        struct sidl_CastException__object* self,
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
sidl_CastException_getClassInfo__exec(
        struct sidl_CastException__object* self,
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
sidl_CastException_getNote__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getNote)(
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
sidl_CastException_setNote__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* message= NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "message", &message, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_setNote)(
    self,
    message,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(message) {free(message);}
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
sidl_CastException_getTrace__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* _retval = 0;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getTrace)(
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
sidl_CastException_addLine__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* traceline= NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "traceline", &traceline, _ex);SIDL_CHECK(
    *_ex);

  /* make the call */
  (self->d_epv->f_addLine)(
    self,
    traceline,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(traceline) {free(traceline);}
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
sidl_CastException_add__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* filename= NULL;
  int32_t lineno = 0;
  char* methodname= NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "filename", &filename, _ex);SIDL_CHECK(
    *_ex);
  sidl_rmi_Call_unpackInt( inArgs, "lineno", &lineno, _ex);SIDL_CHECK(*_ex);
  sidl_rmi_Call_unpackString( inArgs, "methodname", &methodname, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_add)(
    self,
    filename,
    lineno,
    methodname,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(filename) {free(filename);}
  if(methodname) {free(methodname);}
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
sidl_CastException_packObj__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* ser_str = NULL;
  struct sidl_io_Serializer__object* ser = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "ser", &ser_str, _ex);SIDL_CHECK(*_ex);
  ser = skel_sidl_CastException_fconnect_sidl_io_Serializer(ser_str, TRUE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_packObj)(
    self,
    ser,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(ser) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)ser, _ex); SIDL_CHECK(
      *_ex);
    if(ser_str) {free(ser_str);}
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
sidl_CastException_unpackObj__exec(
        struct sidl_CastException__object* self,
        struct sidl_rmi_Call__object* inArgs,
        struct sidl_rmi_Return__object* outArgs,
        struct sidl_BaseInterface__object ** _ex) {
  /* stack space for arguments */
  char* des_str = NULL;
  struct sidl_io_Deserializer__object* des = NULL;
  sidl_BaseInterface _ex3   = NULL;
  sidl_BaseException _SIDLex = NULL;
  /* unpack in and inout argments */
  sidl_rmi_Call_unpackString( inArgs, "des", &des_str, _ex);SIDL_CHECK(*_ex);
  des = skel_sidl_CastException_fconnect_sidl_io_Deserializer(des_str, TRUE, 
    _ex);SIDL_CHECK(*_ex);

  /* make the call */
  (self->d_epv->f_unpackObj)(
    self,
    des,
    _ex);  SIDL_CHECK(*_ex);

  /* pack return value */
  /* pack out and inout argments */
  /* clean-up dangling references */
  if(des) {
    sidl_BaseInterface_deleteRef((sidl_BaseInterface)des, _ex); SIDL_CHECK(
      *_ex);
    if(des_str) {free(des_str);}
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

static void ior_sidl_CastException__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidl_CastException__call_load();
    s_load_called=1;
  }
}

/* CAST: dynamic type casting support. */
static void* ior_sidl_CastException__cast(
  struct sidl_CastException__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.CastException");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = ((struct sidl_CastException__object*)self);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "sidl.BaseException");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_sidlexception.d_sidl_baseexception);
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
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.BaseInterface");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((
          *self).d_sidl_sidlexception.d_sidl_baseclass.d_sidl_baseinterface);
        return cast;
      }
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.SIDLException");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct sidl_SIDLException__object*)self);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "sidl.RuntimeException");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_runtimeexception);
        return cast;
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "sidl.io.Serializable");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_sidlexception.d_sidl_io_serializable);
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

static void ior_sidl_CastException__set_hooks(
  struct sidl_CastException__object* self,
  int on, struct sidl_BaseInterface__object **_ex ) { 
  *_ex = NULL;
  /*
   * Nothing else to do since hooks support not needed.
   */

}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidl_CastException__delete(
  struct sidl_CastException__object* self, struct sidl_BaseInterface__object 
    **_ex)
{
  *_ex = NULL; /* default to no exception */
  sidl_CastException__fini(self,_ex);
  memset((void*)self, 0, sizeof(struct sidl_CastException__object));
  free((void*) self);
}

static char*
ior_sidl_CastException__getURL(
    struct sidl_CastException__object* self,
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
ior_sidl_CastException__raddRef(
    struct sidl_CastException__object* self, sidl_BaseInterface* _ex) {
  sidl_BaseInterface_addRef((sidl_BaseInterface)self, _ex);
}

static sidl_bool
ior_sidl_CastException__isRemote(
    struct sidl_CastException__object* self, sidl_BaseInterface* _ex) {
  *_ex = NULL; /* default to no exception */
  return FALSE;
}

struct sidl_CastException__method {
  const char *d_name;
  void (*d_func)(struct sidl_CastException__object*,
    struct sidl_rmi_Call__object *,
    struct sidl_rmi_Return__object *,
    struct sidl_BaseInterface__object **);
};

static void
ior_sidl_CastException__exec(
    struct sidl_CastException__object* self,
    const char* methodName,
    struct sidl_rmi_Call__object* inArgs,
    struct sidl_rmi_Return__object* outArgs,
    struct sidl_BaseInterface__object **_ex ) { 
  static const struct sidl_CastException__method  s_methods[] = {
    { "add", sidl_CastException_add__exec },
    { "addLine", sidl_CastException_addLine__exec },
    { "addRef", sidl_CastException_addRef__exec },
    { "deleteRef", sidl_CastException_deleteRef__exec },
    { "getClassInfo", sidl_CastException_getClassInfo__exec },
    { "getNote", sidl_CastException_getNote__exec },
    { "getTrace", sidl_CastException_getTrace__exec },
    { "isSame", sidl_CastException_isSame__exec },
    { "isType", sidl_CastException_isType__exec },
    { "packObj", sidl_CastException_packObj__exec },
    { "setNote", sidl_CastException_setNote__exec },
    { "unpackObj", sidl_CastException_unpackObj__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidl_CastException__method);
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

static void sidl_CastException__init_epv(void)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidl_CastException__epv*    epv  = &s_new_epv__sidl_castexception;
  struct sidl_CastException__epv*    hepv = 
    &s_new_epv_hooks__sidl_castexception;
  struct sidl_BaseClass__epv*        e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseClass__epv*        he0  = &s_new_epv_hooks__sidl_baseclass;
  struct sidl_BaseException__epv*    e1   = &s_new_epv__sidl_baseexception;
  struct sidl_BaseException__epv*    he1  = 
    &s_new_epv_hooks__sidl_baseexception;
  struct sidl_BaseInterface__epv*    e2   = &s_new_epv__sidl_baseinterface;
  struct sidl_BaseInterface__epv*    he2  = 
    &s_new_epv_hooks__sidl_baseinterface;
  struct sidl_RuntimeException__epv* e3   = &s_new_epv__sidl_runtimeexception;
  struct sidl_RuntimeException__epv* he3  = 
    &s_new_epv_hooks__sidl_runtimeexception;
  struct sidl_SIDLException__epv*    e4   = &s_new_epv__sidl_sidlexception;
  struct sidl_SIDLException__epv*    he4  = 
    &s_new_epv_hooks__sidl_sidlexception;
  struct sidl_io_Serializable__epv*  e5   = &s_new_epv__sidl_io_serializable;
  struct sidl_io_Serializable__epv*  he5  = 
    &s_new_epv_hooks__sidl_io_serializable;

  struct sidl_SIDLException__epv* s1 = NULL;
  struct sidl_SIDLException__epv* h1 = NULL;
  struct sidl_BaseClass__epv*     s2 = NULL;
  struct sidl_BaseClass__epv*     h2 = NULL;

  sidl_SIDLException__getEPVs(
    &s_old_epv__sidl_baseinterface,
    &s_old_epv_hooks__sidl_baseinterface,
    &s_old_epv__sidl_baseclass,&s_old_epv_hooks__sidl_baseclass,
    &s_old_epv__sidl_baseexception,
    &s_old_epv_hooks__sidl_baseexception,
    &s_old_epv__sidl_io_serializable,
    &s_old_epv_hooks__sidl_io_serializable,
    &s_old_epv__sidl_sidlexception,&s_old_epv_hooks__sidl_sidlexception);
  /*
   * Here we alias the static epvs to some handy small names
   */

  s2  =  s_old_epv__sidl_baseclass;
  h2  =  s_old_epv_hooks__sidl_baseclass;
  s1  =  s_old_epv__sidl_sidlexception;
  h1  =  s_old_epv_hooks__sidl_sidlexception;

  epv->f__cast                    = ior_sidl_CastException__cast;
  epv->f__delete                  = ior_sidl_CastException__delete;
  epv->f__exec                    = ior_sidl_CastException__exec;
  epv->f__getURL                  = ior_sidl_CastException__getURL;
  epv->f__raddRef                 = ior_sidl_CastException__raddRef;
  epv->f__isRemote                = ior_sidl_CastException__isRemote;
  epv->f__set_hooks               = ior_sidl_CastException__set_hooks;
  epv->f__ctor                    = NULL;
  epv->f__ctor2                   = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidl_CastException__object*,struct sidl_BaseInterface__object **)) 
    s1->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidl_CastException__object*,struct sidl_BaseInterface__object **)) 
    s1->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidl_CastException__object*,struct sidl_BaseInterface__object*,struct 
    sidl_BaseInterface__object **)) s1->f_isSame;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidl_CastException__object*,const char*,struct sidl_BaseInterface__object 
    **)) s1->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_CastException__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getClassInfo;
  epv->f_getNote                  = (char* (*)(struct 
    sidl_CastException__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getNote;
  epv->f_setNote                  = (void (*)(struct 
    sidl_CastException__object*,const char*,struct sidl_BaseInterface__object 
    **)) s1->f_setNote;
  epv->f_getTrace                 = (char* (*)(struct 
    sidl_CastException__object*,struct sidl_BaseInterface__object **)) 
    s1->f_getTrace;
  epv->f_addLine                  = (void (*)(struct 
    sidl_CastException__object*,const char*,struct sidl_BaseInterface__object 
    **)) s1->f_addLine;
  epv->f_add                      = (void (*)(struct 
    sidl_CastException__object*,const char*,int32_t,const char*,struct 
    sidl_BaseInterface__object **)) s1->f_add;
  epv->f_packObj                  = (void (*)(struct 
    sidl_CastException__object*,struct sidl_io_Serializer__object*,struct 
    sidl_BaseInterface__object **)) s1->f_packObj;
  epv->f_unpackObj                = (void (*)(struct 
    sidl_CastException__object*,struct sidl_io_Deserializer__object*,struct 
    sidl_BaseInterface__object **)) s1->f_unpackObj;

  sidl_CastException__set_epv(epv);

  memcpy((void*)hepv, epv, sizeof(struct sidl_CastException__epv));
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
  e1->f_getNote             = (char* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getNote;
  e1->f_setNote             = (void (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_setNote;
  e1->f_getTrace            = (char* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getTrace;
  e1->f_addLine             = (void (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_addLine;
  e1->f_add                 = (void (*)(void*,const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;
  e1->f_packObj             = (void (*)(void*,struct 
    sidl_io_Serializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_packObj;
  e1->f_unpackObj           = (void (*)(void*,struct 
    sidl_io_Deserializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_unpackObj;
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

  memcpy((void*) he1, e1, sizeof(struct sidl_BaseException__epv));

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

  memcpy((void*) he2, e2, sizeof(struct sidl_BaseInterface__epv));

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
  e3->f_getNote             = (char* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getNote;
  e3->f_setNote             = (void (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_setNote;
  e3->f_getTrace            = (char* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getTrace;
  e3->f_addLine             = (void (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_addLine;
  e3->f_add                 = (void (*)(void*,const char*,int32_t,const char*,
    struct sidl_BaseInterface__object **)) epv->f_add;
  e3->f_packObj             = (void (*)(void*,struct 
    sidl_io_Serializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_packObj;
  e3->f_unpackObj           = (void (*)(void*,struct 
    sidl_io_Deserializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_unpackObj;
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

  memcpy((void*) he3, e3, sizeof(struct sidl_RuntimeException__epv));

  e4->f__cast               = (void* (*)(struct sidl_SIDLException__object*,
    const char*, struct sidl_BaseInterface__object**)) epv->f__cast;
  e4->f__delete             = (void (*)(struct sidl_SIDLException__object*, 
    struct sidl_BaseInterface__object **)) epv->f__delete;
  e4->f__getURL             = (char* (*)(struct sidl_SIDLException__object*, 
    struct sidl_BaseInterface__object **)) epv->f__getURL;
  e4->f__raddRef            = (void (*)(struct sidl_SIDLException__object*, 
    struct sidl_BaseInterface__object **)) epv->f__raddRef;
  e4->f__isRemote           = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f__isRemote;
  e4->f__exec               = (void (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
    struct sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_addRef              = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_addRef;
  e4->f_deleteRef           = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_deleteRef;
  e4->f_isSame              = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e4->f_isType              = (sidl_bool (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_SIDLException__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;
  e4->f_getNote             = (char* (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getNote;
  e4->f_setNote             = (void (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_setNote;
  e4->f_getTrace            = (char* (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object **)) epv->f_getTrace;
  e4->f_addLine             = (void (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_BaseInterface__object **)) epv->f_addLine;
  e4->f_add                 = (void (*)(struct sidl_SIDLException__object*,
    const char*,int32_t,const char*,struct sidl_BaseInterface__object **)) 
    epv->f_add;
  e4->f_packObj             = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_io_Serializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_packObj;
  e4->f_unpackObj           = (void (*)(struct sidl_SIDLException__object*,
    struct sidl_io_Deserializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_unpackObj;

  memcpy((void*) he4, e4, sizeof(struct sidl_SIDLException__epv));

  e5->f__cast               = (void* (*)(void*,const char*, struct 
    sidl_BaseInterface__object**)) epv->f__cast;
  e5->f__delete             = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__delete;
  e5->f__getURL             = (char* (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__getURL;
  e5->f__raddRef            = (void (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__raddRef;
  e5->f__isRemote           = (sidl_bool (*)(void*, struct 
    sidl_BaseInterface__object **)) epv->f__isRemote;
  e5->f__exec               = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e5->f_packObj             = (void (*)(void*,struct 
    sidl_io_Serializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_packObj;
  e5->f_unpackObj           = (void (*)(void*,struct 
    sidl_io_Deserializer__object*,struct sidl_BaseInterface__object **)) 
    epv->f_unpackObj;
  e5->f_addRef              = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_addRef;
  e5->f_deleteRef           = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_deleteRef;
  e5->f_isSame              = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e5->f_isType              = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e5->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  memcpy((void*) he5, e5, sizeof(struct sidl_io_Serializable__epv));

  s_method_initialized = 1;
  ior_sidl_CastException__ensure_load_called();
}

void sidl_CastException__getEPVs(
  struct sidl_BaseInterface__epv **s_arg_epv__sidl_baseinterface,
  struct sidl_BaseInterface__epv **s_arg_epv_hooks__sidl_baseinterface,
  struct sidl_BaseClass__epv **s_arg_epv__sidl_baseclass,struct 
    sidl_BaseClass__epv **s_arg_epv_hooks__sidl_baseclass,
  struct sidl_BaseException__epv **s_arg_epv__sidl_baseexception,
  struct sidl_BaseException__epv **s_arg_epv_hooks__sidl_baseexception,
  struct sidl_io_Serializable__epv **s_arg_epv__sidl_io_serializable,
  struct sidl_io_Serializable__epv **s_arg_epv_hooks__sidl_io_serializable,
  struct sidl_SIDLException__epv **s_arg_epv__sidl_sidlexception,struct 
    sidl_SIDLException__epv **s_arg_epv_hooks__sidl_sidlexception,
  struct sidl_RuntimeException__epv **s_arg_epv__sidl_runtimeexception,
  struct sidl_RuntimeException__epv **s_arg_epv_hooks__sidl_runtimeexception,
  struct sidl_CastException__epv **s_arg_epv__sidl_castexception,struct 
    sidl_CastException__epv **s_arg_epv_hooks__sidl_castexception)
{
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidl_CastException__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  *s_arg_epv__sidl_baseinterface = &s_new_epv__sidl_baseinterface;
  *s_arg_epv_hooks__sidl_baseinterface = &s_new_epv_hooks__sidl_baseinterface;
  *s_arg_epv__sidl_baseclass = &s_new_epv__sidl_baseclass;
  *s_arg_epv_hooks__sidl_baseclass = &s_new_epv_hooks__sidl_baseclass;
  *s_arg_epv__sidl_baseexception = &s_new_epv__sidl_baseexception;
  *s_arg_epv_hooks__sidl_baseexception = &s_new_epv_hooks__sidl_baseexception;
  *s_arg_epv__sidl_io_serializable = &s_new_epv__sidl_io_serializable;
  *s_arg_epv_hooks__sidl_io_serializable = 
    &s_new_epv_hooks__sidl_io_serializable;
  *s_arg_epv__sidl_sidlexception = &s_new_epv__sidl_sidlexception;
  *s_arg_epv_hooks__sidl_sidlexception = &s_new_epv_hooks__sidl_sidlexception;
  *s_arg_epv__sidl_runtimeexception = &s_new_epv__sidl_runtimeexception;
  *s_arg_epv_hooks__sidl_runtimeexception = 
    &s_new_epv_hooks__sidl_runtimeexception;
  *s_arg_epv__sidl_castexception = &s_new_epv__sidl_castexception;
  *s_arg_epv_hooks__sidl_castexception = &s_new_epv_hooks__sidl_castexception;
}
/*
 * SUPER: returns parent's non-overrided EPV
 */

static struct sidl_SIDLException__epv* sidl_CastException__super(void) {
  return s_old_epv__sidl_sidlexception;
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
      sidl_ClassInfoI_setName(impl, "sidl.CastException",_ex);
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
initMetadata(struct sidl_CastException__object* self, sidl_BaseInterface* _ex)
{
  *_ex = 0; /* default no exception */
  if (self) {
    struct sidl_BaseClass__data *data = (struct sidl_BaseClass__data*)((
      *self).d_sidl_sidlexception.d_sidl_baseclass.d_data);
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

struct sidl_CastException__object*
sidl_CastException__new(void* ddata, struct sidl_BaseInterface__object ** _ex)
{
  struct sidl_CastException__object* self =
    (struct sidl_CastException__object*) malloc(
      sizeof(struct sidl_CastException__object));
  *_ex = NULL; /* default to no exception */
  sidl_CastException__init(self, ddata, _ex); SIDL_CHECK(*_ex);
  initMetadata(self, _ex); SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidl_CastException__init(
  struct sidl_CastException__object* self,
   void* ddata,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidl_CastException__object* s0 = self;
  struct sidl_SIDLException__object* s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*     s2 = &s1->d_sidl_baseclass;

  *_ex = 0; /* default no exception */
  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidl_CastException__init_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  sidl_SIDLException__init(s1, NULL, _ex); SIDL_CHECK(*_ex);

  s2->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s2->d_epv                      = &s_new_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv   = &s_new_epv__sidl_baseexception;
  s1->d_sidl_io_serializable.d_epv = &s_new_epv__sidl_io_serializable;
  s1->d_epv                        = &s_new_epv__sidl_sidlexception;

  s0->d_sidl_runtimeexception.d_epv = &s_new_epv__sidl_runtimeexception;
  s0->d_epv                         = &s_new_epv__sidl_castexception;

  s0->d_sidl_runtimeexception.d_object = self;

  s0->d_data = NULL;

  ior_sidl_CastException__set_hooks(s0, FALSE, _ex);

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

void sidl_CastException__fini(
  struct sidl_CastException__object* self,
  struct sidl_BaseInterface__object **_ex)
{
  struct sidl_CastException__object* s0 = self;
  struct sidl_SIDLException__object* s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*     s2 = &s1->d_sidl_baseclass;

  *_ex = NULL; /* default to no exception */
  (*(s0->d_epv->f__dtor))(s0,_ex);
  SIDL_CHECK(*_ex);

  s2->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s2->d_epv                      = s_old_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv   = s_old_epv__sidl_baseexception;
  s1->d_sidl_io_serializable.d_epv = s_old_epv__sidl_io_serializable;
  s1->d_epv                        = s_old_epv__sidl_sidlexception;

  sidl_SIDLException__fini(s1, _ex); SIDL_CHECK(*_ex);
  EXIT:
  return;
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidl_CastException__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidl_CastException__external
s_externalEntryPoints = {
  sidl_CastException__new,
  sidl_CastException__super,
  1, 
  0
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidl_CastException__external*
sidl_CastException__externals(void)
{
  return &s_externalEntryPoints;
}

