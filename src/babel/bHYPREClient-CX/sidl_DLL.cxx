// 
// File:          sidl_DLL.cxx
// Symbol:        sidl.DLL-v0.9.15
// Symbol Type:   class
// Babel Version: 1.0.0
// Release:       $Name: V1-14-0b $
// Revision:      @(#) $Id: sidl_DLL.cxx,v 1.2 2006/09/14 21:52:15 painter Exp $
// Description:   Client-side glue code for sidl.DLL
// 
// Copyright (c) 2000-2002, The Regents of the University of California.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Components Team <components@llnl.gov>
// All rights reserved.
// 
// This file is part of Babel. For more information, see
// http://www.llnl.gov/CASC/components/. Please read the COPYRIGHT file
// for Our Notice and the LICENSE file for the GNU Lesser General Public
// License.
// 
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License (as published by
// the Free Software Foundation) version 2.1 dated February 1999.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// conditions of the GNU Lesser General Public License for more details.
// 
// You should have recieved a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_sidl_DLL_hxx
#include "sidl_DLL.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseException_hxx
#include "sidl_BaseException.hxx"
#endif
#ifndef included_sidl_LangSpecificException_hxx
#include "sidl_LangSpecificException.hxx"
#endif
#ifndef included_sidl_RuntimeException_hxx
#include "sidl_RuntimeException.hxx"
#endif
#ifndef included_sidl_rmi_Call_hxx
#include "sidl_rmi_Call.hxx"
#endif
#ifndef included_sidl_rmi_Return_hxx
#include "sidl_rmi_Return.hxx"
#endif
#ifndef included_sidl_rmi_Ticket_hxx
#include "sidl_rmi_Ticket.hxx"
#endif
#ifndef included_sidl_rmi_InstanceHandle_hxx
#include "sidl_rmi_InstanceHandle.hxx"
#endif
#include "sidl_String.h"
#include "sidl_rmi_ConnectRegistry.h"
// 
// Includes for all method dependencies.
// 
#ifndef included_sidl_BaseClass_hxx
#include "sidl_BaseClass.hxx"
#endif
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif

#define LANG_SPECIFIC_INIT()
// 
// connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
// 
static int connect_loaded = 0;
extern "C" {
#include <stdlib.h>
#include <string.h>
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_rmi_ProtocolFactory_h
#include "sidl_rmi_ProtocolFactory.h"
#endif
#ifndef included_sidl_rmi_InstanceRegistry_h
#include "sidl_rmi_InstanceRegistry.h"
#endif
#ifndef included_sidl_rmi_InstanceHandle_h
#include "sidl_rmi_InstanceHandle.h"
#endif
#ifndef included_sidl_rmi_Invocation_h
#include "sidl_rmi_Invocation.h"
#endif
#ifndef included_sidl_rmi_Response_h
#include "sidl_rmi_Response.h"
#endif
#ifndef included_sidl_rmi_ServerRegistry_h
#include "sidl_rmi_ServerRegistry.h"
#endif
#ifndef included_sidl_rmi_ConnectRegistry_h
#include "sidl_rmi_ConnectRegistry.h"
#endif
#ifndef included_sidl_io_Serializable_h
#include "sidl_io_Serializable.h"
#endif
#include "sidl_Exception.h"

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidl_DLL__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_DLL__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_DLL__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_DLL__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

  // Static variables to hold version of IOR
  static const int32_t s_IOR_MAJOR_VERSION = 0;
  static const int32_t s_IOR_MINOR_VERSION = 10;

  // Static variables for managing EPV initialization.
  static int s_remote_initialized = 0;

  static struct sidl_DLL__epv s_rem_epv__sidl_dll;

  static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

  static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_sidl_DLL__cast(
    struct sidl_DLL__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp0 = strcmp(name, "sidl.BaseInterface");
    if (!cmp0) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
    else if (cmp0 < 0) {
      cmp1 = strcmp(name, "sidl.BaseClass");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
    else if (cmp0 > 0) {
      cmp1 = strcmp(name, "sidl.DLL");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = self;
        return cast;
      }
    }
    if ((*self->d_epv->f_isType)(self,name, _ex)) {
      void* (*func)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**) = 
        (void* (*)(struct sidl_rmi_InstanceHandle__object*,
          struct sidl_BaseInterface__object**)) 
        sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
      cast =  (*func)(((struct sidl_DLL__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_sidl_DLL__delete(
    struct sidl_DLL__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_sidl_DLL__getURL(
    struct sidl_DLL__object* self, sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      sidl_DLL__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_sidl_DLL__raddRef(
    struct sidl_DLL__object* self,sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      sidl_DLL__remote*)self->d_data)->d_ih;
    sidl_rmi_Response _rsvp = NULL;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn,
      "addRef", _ex ); SIDL_CHECK(*_ex);
    // send actual RMI request
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
    // Check for exceptions
    netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
    if(netex != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
        &throwaway_exception);
      return;
    }

    // cleanup and return
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
    return;
  }

  // REMOTE ISREMOTE: returns true if this object is Remote (it is).
  static sidl_bool
  remote_sidl_DLL__isRemote(
      struct sidl_DLL__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_sidl_DLL__set_hooks(
    /* in */ struct sidl_DLL__object* self ,
    /* in */ sidl_bool on,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL._set_hooks.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE EXEC: call the exec function for the object.
  static void remote_sidl_DLL__exec(
    struct sidl_DLL__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_sidl_DLL_addRef(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_DLL__remote* r_obj = (struct sidl_DLL__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_sidl_DLL_deleteRef(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_DLL__remote* r_obj = (struct sidl_DLL__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount--;
      if(r_obj->d_refcount == 0) {
        sidl_rmi_InstanceHandle_deleteRef(r_obj->d_ih, _ex);
        free(r_obj);
        free(self);
      }
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:isSame
  static sidl_bool
  remote_sidl_DLL_isSame(
    /* in */ struct sidl_DLL__object* self ,
    /* in */ struct sidl_BaseInterface__object* iobj,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isSame", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(iobj){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj,
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "iobj", _url,
          _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "iobj", NULL,
          _ex);SIDL_CHECK(*_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.isSame.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:isType
  static sidl_bool
  remote_sidl_DLL_isType(
    /* in */ struct sidl_DLL__object* self ,
    /* in */ const char* name,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.isType.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:getClassInfo
  static struct sidl_ClassInfo__object*
  remote_sidl_DLL_getClassInfo(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char*_retval_str = NULL;
      struct sidl_ClassInfo__object* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.getClassInfo.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
        _ex);SIDL_CHECK(*_ex);
      _retval = sidl_ClassInfo__connectI(_retval_str, FALSE,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:loadLibrary
  static sidl_bool
  remote_sidl_DLL_loadLibrary(
    /* in */ struct sidl_DLL__object* self ,
    /* in */ const char* uri,
    /* in */ sidl_bool loadGlobally,
    /* in */ sidl_bool loadLazy,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "loadLibrary", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "uri", uri, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "loadGlobally", loadGlobally,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "loadLazy", loadLazy,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.loadLibrary.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:getName
  static char*
  remote_sidl_DLL_getName(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getName", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.getName.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:isGlobal
  static sidl_bool
  remote_sidl_DLL_isGlobal(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isGlobal", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.isGlobal.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:isLazy
  static sidl_bool
  remote_sidl_DLL_isLazy(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      sidl_bool _retval = FALSE;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isLazy", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.isLazy.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:unloadLibrary
  static void
  remote_sidl_DLL_unloadLibrary(
    /* in */ struct sidl_DLL__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unloadLibrary", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.unloadLibrary.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:lookupSymbol
  static void*
  remote_sidl_DLL_lookupSymbol(
    /* in */ struct sidl_DLL__object* self ,
    /* in */ const char* linker_name,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      void* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "lookupSymbol", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "linker_name", linker_name,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.lookupSymbol.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackOpaque( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:createClass
  static struct sidl_BaseClass__object*
  remote_sidl_DLL_createClass(
    /* in */ struct sidl_DLL__object* self ,
    /* in */ const char* sidl_name,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char*_retval_str = NULL;
      struct sidl_BaseClass__object* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_DLL__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "createClass", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "sidl_name", sidl_name,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.DLL.createClass.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
        _ex);SIDL_CHECK(*_ex);
      _retval = sidl_BaseClass__connectI(_retval_str, FALSE,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void sidl_DLL__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct sidl_DLL__epv*           epv = &s_rem_epv__sidl_dll;
    struct sidl_BaseClass__epv*     e0  = &s_rem_epv__sidl_baseclass;
    struct sidl_BaseInterface__epv* e1  = &s_rem_epv__sidl_baseinterface;

    epv->f__cast              = remote_sidl_DLL__cast;
    epv->f__delete            = remote_sidl_DLL__delete;
    epv->f__exec              = remote_sidl_DLL__exec;
    epv->f__getURL            = remote_sidl_DLL__getURL;
    epv->f__raddRef           = remote_sidl_DLL__raddRef;
    epv->f__isRemote          = remote_sidl_DLL__isRemote;
    epv->f__set_hooks         = remote_sidl_DLL__set_hooks;
    epv->f__ctor              = NULL;
    epv->f__ctor2             = NULL;
    epv->f__dtor              = NULL;
    epv->f_addRef             = remote_sidl_DLL_addRef;
    epv->f_deleteRef          = remote_sidl_DLL_deleteRef;
    epv->f_isSame             = remote_sidl_DLL_isSame;
    epv->f_isType             = remote_sidl_DLL_isType;
    epv->f_getClassInfo       = remote_sidl_DLL_getClassInfo;
    epv->f_loadLibrary        = remote_sidl_DLL_loadLibrary;
    epv->f_getName            = remote_sidl_DLL_getName;
    epv->f_isGlobal           = remote_sidl_DLL_isGlobal;
    epv->f_isLazy             = remote_sidl_DLL_isLazy;
    epv->f_unloadLibrary      = remote_sidl_DLL_unloadLibrary;
    epv->f_lookupSymbol       = remote_sidl_DLL_lookupSymbol;
    epv->f_createClass        = remote_sidl_DLL_createClass;

    e0->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e0->f__delete      = (void (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__delete;
    e0->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__getURL;
    e0->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e0->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e0->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e0->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,
      const char*,struct sidl_BaseInterface__object **)) epv->f_isType;
    e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
      sidl_BaseClass__object*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e1->f__cast        = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e1->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
    e1->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
    e1->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
    e1->f__isRemote    = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e1->f__set_hooks   = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e1->f__exec        = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_addRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e1->f_deleteRef    = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e1->f_isSame       = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e1->f_isType       = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct sidl_DLL__object*
  sidl_DLL__remoteConnect(const char *url, sidl_bool ar,
    sidl_BaseInterface *_ex)
  {
    struct sidl_DLL__object* self;

    struct sidl_DLL__object* s0;
    struct sidl_BaseClass__object* s1;

    struct sidl_DLL__remote* r_obj;
    sidl_rmi_InstanceHandle instance = NULL;
    char* objectID = NULL;
    objectID = NULL;
    *_ex = NULL;
    if(url == NULL) {return NULL;}
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      sidl_BaseInterface bi = 
        (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
        objectID, _ex); SIDL_CHECK(*_ex);
      return sidl_DLL__rmicast(bi,_ex);SIDL_CHECK(*_ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar,
      _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct sidl_DLL__object*) malloc(
        sizeof(struct sidl_DLL__object));

    r_obj =
      (struct sidl_DLL__remote*) malloc(
        sizeof(struct sidl_DLL__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                          self;
    s1 =                          &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_DLL__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_dll;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    return NULL;
  }
  // Create an instance that uses an already existing 
  // InstanceHandle to connect to an existing remote object.
  static struct sidl_DLL__object*
  sidl_DLL__IHConnect(sidl_rmi_InstanceHandle instance, sidl_BaseInterface *_ex)
  {
    struct sidl_DLL__object* self;

    struct sidl_DLL__object* s0;
    struct sidl_BaseClass__object* s1;

    struct sidl_DLL__remote* r_obj;
    self =
      (struct sidl_DLL__object*) malloc(
        sizeof(struct sidl_DLL__object));

    r_obj =
      (struct sidl_DLL__remote*) malloc(
        sizeof(struct sidl_DLL__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                          self;
    s1 =                          &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_DLL__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_dll;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
    return self;
    EXIT:
    return NULL;
  }
  // REMOTE: generate remote instance given URL string.
  static struct sidl_DLL__object*
  sidl_DLL__remoteCreate(const char *url, sidl_BaseInterface *_ex)
  {
    sidl_BaseInterface _throwaway_exception = NULL;
    struct sidl_DLL__object* self;

    struct sidl_DLL__object* s0;
    struct sidl_BaseClass__object* s1;

    struct sidl_DLL__remote* r_obj;
    sidl_rmi_InstanceHandle instance = 
      sidl_rmi_ProtocolFactory_createInstance(url, "sidl.DLL",
      _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct sidl_DLL__object*) malloc(
        sizeof(struct sidl_DLL__object));

    r_obj =
      (struct sidl_DLL__remote*) malloc(
        sizeof(struct sidl_DLL__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                          self;
    s1 =                          &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_DLL__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_dll;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    if(instance) { sidl_rmi_InstanceHandle_deleteRef(instance,
      &_throwaway_exception); }
    return NULL;
  }
  // 
  // Cast method for interface and class type conversions.
  // 
  struct sidl_DLL__object*
  sidl_DLL__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct sidl_DLL__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("sidl.DLL",
        (void*)sidl_DLL__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct sidl_DLL__object*) (*base->d_epv->f__cast)(
        base->d_object,
        "sidl.DLL", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct sidl_DLL__object*
  sidl_DLL__connectI(const char* url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex)
  {
    return sidl_DLL__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
sidl::DLL::throwException0(
  struct sidl_BaseInterface__object *_exception
)
  // throws:
{
  void * _p = 0;
  struct sidl_BaseInterface__object *throwaway_exception;

  if ( (_p=(*(_exception->d_epv->f__cast))(_exception->d_object,
    "sidl.RuntimeException", &throwaway_exception)) != 0 ) {
    struct sidl_RuntimeException__object * _realtype = reinterpret_cast< struct 
      sidl_RuntimeException__object*>(_p);
    (*_exception->d_epv->f_deleteRef)(_exception->d_object,
      &throwaway_exception);
    // Note: alternate constructor does not increment refcount.
    ::sidl::RuntimeException _resolved_exception = ::sidl::RuntimeException( 
      _realtype, false );
    _resolved_exception.add(__FILE__,__LINE__, "C++ stub.");
    throw _resolved_exception;
  }
  // Any unresolved exception is treated as LangSpecificException
  ::sidl::LangSpecificException _unexpected = 
    ::sidl::LangSpecificException::_create();
  _unexpected.add(__FILE__,__LINE__, "Unknown method");
  _unexpected.setNote("Unexpected exception received by C++ stub.");
  throw _unexpected;
}

//////////////////////////////////////////////////
// 
// User Defined Methods
// 


/**
 * Load a dynamic link library using the specified URI.  The
 * URI may be of the form "main:", "lib:", "file:", "ftp:", or
 * "http:".  A URI that starts with any other protocol string
 * is assumed to be a file name.  The "main:" URI creates a
 * library that allows access to global symbols in the running
 * program's main address space.  The "lib:X" URI converts the
 * library "X" into a platform-specific name (e.g., libX.so) and
 * loads that library.  The "file:" URI opens the DLL from the
 * specified file path.  The "ftp:" and "http:" URIs copy the
 * specified library from the remote site into a local temporary
 * file and open that file.  This method returns true if the
 * DLL was loaded successfully and false otherwise.  Note that
 * the "ftp:" and "http:" protocols are valid only if the W3C
 * WWW library is available.
 * 
 * @param uri          the URI to load. This can be a .la file
 * (a metadata file produced by libtool) or
 * a shared library binary (i.e., .so,
 * .dll or whatever is appropriate for your
 * OS)
 * @param loadGlobally <code>true</code> means that the shared
 * library symbols will be loaded into the
 * global namespace; <code>false</code> 
 * means they will be loaded into a 
 * private namespace. Some operating systems
 * may not be able to honor the value presented
 * here.
 * @param loadLazy     <code>true</code> instructs the loader to
 * that symbols can be resolved as needed (lazy)
 * instead of requiring everything to be resolved
 * now (at load time).
 */
bool
sidl::DLL::loadLibrary( /* in */const ::std::string& uri,
  /* in */bool loadGlobally, /* in */bool loadLazy )

{
  bool _result;
  ior_t* const loc_self = _get_ior();
  sidl_bool _local_result;
  sidl_bool _local_loadGlobally = loadGlobally;
  sidl_bool _local_loadLazy = loadLazy;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_loadLibrary))(loc_self,
    /* in */ uri.c_str(), /* in */ _local_loadGlobally,
    /* in */ _local_loadLazy, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Get the library name.  This is the name used to load the
 * library in <code>loadLibrary</code> except that all file names
 * contain the "file:" protocol.
 */
::std::string
sidl::DLL::getName(  )

{
  ::std::string _result;
  ior_t* const loc_self = _get_ior();
  char * _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_getName))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  if (_local_result) {
    _result = _local_result;
    ::sidl_String_free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if the library was loaded into the global namespace.
 */
bool
sidl::DLL::isGlobal(  )

{
  bool _result;
  ior_t* const loc_self = _get_ior();
  sidl_bool _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isGlobal))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Return true if the library was loaded using lazy symbol resolution.
 */
bool
sidl::DLL::isLazy(  )

{
  bool _result;
  ior_t* const loc_self = _get_ior();
  sidl_bool _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f_isLazy))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  _result = _local_result;
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Unload the dynamic link library.  The library may no longer
 * be used to access symbol names.  When the library is actually
 * unloaded from the memory image depends on details of the operating
 * system.
 */
void
sidl::DLL::unloadLibrary(  )

{

  ior_t* const loc_self = _get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unloadLibrary))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}



/**
 * Lookup a symbol from the DLL and return the associated pointer.
 * A null value is returned if the name does not exist.
 */
void*
sidl::DLL::lookupSymbol( /* in */const ::std::string& linker_name )

{
  void* _result;
  ior_t* const loc_self = _get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_lookupSymbol))(loc_self,
    /* in */ linker_name.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Create an instance of the sidl class.  If the class constructor
 * is not defined in this DLL, then return null.
 */
::sidl::BaseClass
sidl::DLL::createClass( /* in */const ::std::string& sidl_name )

{
  ::sidl::BaseClass _result;
  ior_t* const loc_self = _get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = ::sidl::BaseClass( (*(loc_self->d_epv->f_createClass))(loc_self,
    /* in */ sidl_name.c_str(), &_exception ), false);
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// static constructor
::sidl::DLL
sidl::DLL::_create() {
  struct sidl_BaseInterface__object * _exception, *_throwaway;
  ::sidl::DLL self( (*_get_ext()->createObject)(NULL,&_exception), false );
  if (_exception) {
    void *_p;
    if ( (_p = (*(_exception->d_epv->f__cast))(_exception->d_object,
      "sidl.RuntimeException", &_throwaway)) != 0) {
    ::sidl::RuntimeException _resolved(reinterpret_cast< struct 
      sidl_RuntimeException__object * >(_p), false);
    (*(_exception->d_epv->f_deleteRef))(_exception->d_object, &_throwaway);
    _resolved.add(__FILE__,__LINE__,"C++ ctor.");
    throw _resolved;
  }
}
return self;
}

// remote constructor
::sidl::DLL
sidl::DLL::_create(const std::string& url) {
  ior_t* ior_self;
  sidl_BaseInterface__object* _ex = 0;
  ior_self = sidl_DLL__remoteCreate( url.c_str(), &_ex );
  if (_ex != 0 ) {
    ; //TODO: handle exception
  }
  return ::sidl::DLL( ior_self, false );
}

// remote connector 2
::sidl::DLL
sidl::DLL::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  sidl_BaseInterface__object* _ex = 0;
  ior_self = sidl_DLL__remoteConnect( url.c_str(), ar?TRUE:FALSE, &_ex );
  if (_ex != 0 ) {
    ; //TODO: handle exception
  }
  return ::sidl::DLL( ior_self, false );
}

// copy constructor
sidl::DLL::DLL ( const ::sidl::DLL& original ) {
  d_self = ::sidl::DLL::_cast(original._get_ior());
  d_weak_reference = false;
}

// assignment operator
::sidl::DLL&
sidl::DLL::operator=( const ::sidl::DLL& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = ::sidl::DLL::_cast(rhs._get_ior());
    // note _cast incremements the reference count
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
sidl::DLL::DLL ( ::sidl::DLL::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
sidl::DLL::DLL ( ::sidl::DLL::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// exec has special argument passing to avoid #include circularities
void ::sidl::DLL::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::sidl::DLL::ior_t* const loc_self = _get_ior();
  struct sidl_BaseInterface__object *throwaway_exception;
  (*loc_self->d_epv->f__exec)(loc_self,
                                methodName.c_str(),
                                inArgs._get_ior(),
                                outArgs._get_ior(),
                                &throwaway_exception);
}


/**
 * Get the URL of the Implementation of this object (for RMI)
 */
::std::string
sidl::DLL::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = _get_ior();
  char * _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f__getURL))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  if (_local_result) {
    _result = _local_result;
    ::sidl_String_free( _local_result );
  }
  /*unpack results and cleanup*/
  return _result;
}


/**
 * Method to set whether or not method hooks should be invoked.
 */
void
sidl::DLL::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = _get_ior();
  sidl_bool _local_on = on;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self, /* in */ _local_on,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}

// protected method that implements casting
struct sidl_DLL__object* sidl::DLL::_cast(const void* src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("sidl.DLL",
      (void*)sidl_DLL__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object,
      "sidl.DLL", &throwaway_exception));
  }
  return cast;
}

// Static data type
const ::sidl::DLL::ext_t * sidl::DLL::s_ext = 0;

// private static method to get static data type
const ::sidl::DLL::ext_t *
sidl::DLL::_get_ext()
  throw ( ::sidl::NullIORException)
{
  if (! s_ext ) {
    s_ext = sidl_DLL__externals();
  }
  return s_ext;
}

