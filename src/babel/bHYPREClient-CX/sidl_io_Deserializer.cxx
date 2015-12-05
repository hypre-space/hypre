// 
// File:          sidl_io_Deserializer.cxx
// Symbol:        sidl.io.Deserializer-v0.9.15
// Symbol Type:   interface
// Babel Version: 1.0.0
// Release:       $Name: V1-14-0b $
// Revision:      @(#) $Id: sidl_io_Deserializer.cxx,v 1.2 2006/09/14 21:52:15 painter Exp $
// Description:   Client-side glue code for sidl.io.Deserializer
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

#ifndef included_sidl_io_Deserializer_hxx
#include "sidl_io_Deserializer.hxx"
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
#ifndef included_sidl_BaseInterface_hxx
#include "sidl_BaseInterface.hxx"
#endif
#ifndef included_sidl_ClassInfo_hxx
#include "sidl_ClassInfo.hxx"
#endif
#ifndef included_sidl_io_Serializable_hxx
#include "sidl_io_Serializable.hxx"
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
static struct sidl_recursive_mutex_t sidl_io__Deserializer__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_io__Deserializer__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_io__Deserializer__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_io__Deserializer__mutex )==EDEADLOCK) */
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

  static struct sidl_io__Deserializer__epv s_rem_epv__sidl_io__deserializer;

  static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

  static struct sidl_io_Deserializer__epv s_rem_epv__sidl_io_deserializer;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_sidl_io__Deserializer__cast(
    struct sidl_io__Deserializer__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1;
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
        cast = &((*self).d_sidl_baseinterface);
        return cast;
      }
    }
    else if (cmp0 > 0) {
      cmp1 = strcmp(name, "sidl.io._Deserializer");
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
      cast =  (*func)(((struct 
        sidl_io__Deserializer__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_sidl_io__Deserializer__delete(
    struct sidl_io__Deserializer__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_sidl_io__Deserializer__getURL(
    struct sidl_io__Deserializer__object* self, sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      sidl_io__Deserializer__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_sidl_io__Deserializer__raddRef(
    struct sidl_io__Deserializer__object* self,sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      sidl_io__Deserializer__remote*)self->d_data)->d_ih;
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
  remote_sidl_io__Deserializer__isRemote(
      struct sidl_io__Deserializer__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_sidl_io__Deserializer__set_hooks(
    /* in */ struct sidl_io__Deserializer__object* self ,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer._set_hooks.", &throwaway_exception);
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
  static void remote_sidl_io__Deserializer__exec(
    struct sidl_io__Deserializer__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_sidl_io__Deserializer_addRef(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_io__Deserializer__remote* r_obj = (struct 
        sidl_io__Deserializer__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_sidl_io__Deserializer_deleteRef(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_io__Deserializer__remote* r_obj = (struct 
        sidl_io__Deserializer__remote*)self->d_data;
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
  remote_sidl_io__Deserializer_isSame(
    /* in */ struct sidl_io__Deserializer__object* self ,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.isSame.", &throwaway_exception);
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
  remote_sidl_io__Deserializer_isType(
    /* in */ struct sidl_io__Deserializer__object* self ,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.isType.", &throwaway_exception);
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
  remote_sidl_io__Deserializer_getClassInfo(
    /* in */ struct sidl_io__Deserializer__object* self ,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.getClassInfo.", &throwaway_exception);
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

  // REMOTE METHOD STUB:unpackBool
  static void
  remote_sidl_io__Deserializer_unpackBool(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ sidl_bool* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackBool", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackBool.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackBool( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackChar
  static void
  remote_sidl_io__Deserializer_unpackChar(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ char* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackChar", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackChar.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackChar( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackInt
  static void
  remote_sidl_io__Deserializer_unpackInt(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ int32_t* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackInt", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackInt.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackLong
  static void
  remote_sidl_io__Deserializer_unpackLong(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ int64_t* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackLong", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackLong.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackLong( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackOpaque
  static void
  remote_sidl_io__Deserializer_unpackOpaque(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ void** value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackOpaque", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackOpaque.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackOpaque( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackFloat
  static void
  remote_sidl_io__Deserializer_unpackFloat(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ float* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackFloat", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackFloat.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackFloat( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackDouble
  static void
  remote_sidl_io__Deserializer_unpackDouble(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ double* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackDouble", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackDouble.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDouble( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackFcomplex
  static void
  remote_sidl_io__Deserializer_unpackFcomplex(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ struct sidl_fcomplex* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackFcomplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackFcomplex.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackFcomplex( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackDcomplex
  static void
  remote_sidl_io__Deserializer_unpackDcomplex(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ struct sidl_dcomplex* value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackDcomplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackDcomplex.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDcomplex( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackString
  static void
  remote_sidl_io__Deserializer_unpackString(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ char** value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackString", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackString.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackString( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackSerializable
  static void
  remote_sidl_io__Deserializer_unpackSerializable(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out */ struct sidl_io_Serializable__object** value,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char* value_str= NULL;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackSerializable", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackSerializable.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackString( _rsvp, "value", &value_str,
        _ex);SIDL_CHECK(*_ex);
      *value = sidl_io_Serializable__connectI(value_str, FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackBoolArray
  static void
  remote_sidl_io__Deserializer_unpackBoolArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<bool> */ struct sidl_bool__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackBoolArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackBoolArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackBoolArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackCharArray
  static void
  remote_sidl_io__Deserializer_unpackCharArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<char> */ struct sidl_char__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackCharArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackCharArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackCharArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackIntArray
  static void
  remote_sidl_io__Deserializer_unpackIntArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<int> */ struct sidl_int__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackIntArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackIntArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackIntArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackLongArray
  static void
  remote_sidl_io__Deserializer_unpackLongArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<long> */ struct sidl_long__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackLongArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackLongArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackLongArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackOpaqueArray
  static void
  remote_sidl_io__Deserializer_unpackOpaqueArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<opaque> */ struct sidl_opaque__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackOpaqueArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackOpaqueArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackOpaqueArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackFloatArray
  static void
  remote_sidl_io__Deserializer_unpackFloatArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<float> */ struct sidl_float__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackFloatArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackFloatArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackFloatArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackDoubleArray
  static void
  remote_sidl_io__Deserializer_unpackDoubleArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<double> */ struct sidl_double__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackDoubleArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackDoubleArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDoubleArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackFcomplexArray
  static void
  remote_sidl_io__Deserializer_unpackFcomplexArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<fcomplex> */ struct sidl_fcomplex__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackFcomplexArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackFcomplexArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackFcomplexArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackDcomplexArray
  static void
  remote_sidl_io__Deserializer_unpackDcomplexArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<dcomplex> */ struct sidl_dcomplex__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackDcomplexArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackDcomplexArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDcomplexArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackStringArray
  static void
  remote_sidl_io__Deserializer_unpackStringArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<string> */ struct sidl_string__array** value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackStringArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackStringArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackStringArray( _rsvp, "value", value,0,0,FALSE,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackGenericArray
  static void
  remote_sidl_io__Deserializer_unpackGenericArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<> */ struct sidl__array** value,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackGenericArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackGenericArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackGenericArray( _rsvp, "value", value,
        _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE METHOD STUB:unpackSerializableArray
  static void
  remote_sidl_io__Deserializer_unpackSerializableArray(
    /* in */ struct sidl_io__Deserializer__object* self ,
    /* in */ const char* key,
    /* out array<sidl.io.Serializable> */ struct sidl_io_Serializable__array** 
      value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool isRarray,
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
        sidl_io__Deserializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackSerializableArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "isRarray", isRarray,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Deserializer.unpackSerializableArray.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // unpack out and inout arguments
      sidl_rmi_Response_unpackSerializableArray( _rsvp, "value", value,0,0,
        FALSE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return;
    }
  }

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void sidl_io__Deserializer__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct sidl_io__Deserializer__epv* epv = &s_rem_epv__sidl_io__deserializer;
    struct sidl_BaseInterface__epv*    e0  = &s_rem_epv__sidl_baseinterface;
    struct sidl_io_Deserializer__epv*  e1  = &s_rem_epv__sidl_io_deserializer;

    epv->f__cast                        = remote_sidl_io__Deserializer__cast;
    epv->f__delete                      = remote_sidl_io__Deserializer__delete;
    epv->f__exec                        = remote_sidl_io__Deserializer__exec;
    epv->f__getURL                      = remote_sidl_io__Deserializer__getURL;
    epv->f__raddRef                     = remote_sidl_io__Deserializer__raddRef;
    epv->f__isRemote                    = 
      remote_sidl_io__Deserializer__isRemote;
    epv->f__set_hooks                   = 
      remote_sidl_io__Deserializer__set_hooks;
    epv->f__ctor                        = NULL;
    epv->f__ctor2                       = NULL;
    epv->f__dtor                        = NULL;
    epv->f_addRef                       = remote_sidl_io__Deserializer_addRef;
    epv->f_deleteRef                    = 
      remote_sidl_io__Deserializer_deleteRef;
    epv->f_isSame                       = remote_sidl_io__Deserializer_isSame;
    epv->f_isType                       = remote_sidl_io__Deserializer_isType;
    epv->f_getClassInfo                 = 
      remote_sidl_io__Deserializer_getClassInfo;
    epv->f_unpackBool                   = 
      remote_sidl_io__Deserializer_unpackBool;
    epv->f_unpackChar                   = 
      remote_sidl_io__Deserializer_unpackChar;
    epv->f_unpackInt                    = 
      remote_sidl_io__Deserializer_unpackInt;
    epv->f_unpackLong                   = 
      remote_sidl_io__Deserializer_unpackLong;
    epv->f_unpackOpaque                 = 
      remote_sidl_io__Deserializer_unpackOpaque;
    epv->f_unpackFloat                  = 
      remote_sidl_io__Deserializer_unpackFloat;
    epv->f_unpackDouble                 = 
      remote_sidl_io__Deserializer_unpackDouble;
    epv->f_unpackFcomplex               = 
      remote_sidl_io__Deserializer_unpackFcomplex;
    epv->f_unpackDcomplex               = 
      remote_sidl_io__Deserializer_unpackDcomplex;
    epv->f_unpackString                 = 
      remote_sidl_io__Deserializer_unpackString;
    epv->f_unpackSerializable           = 
      remote_sidl_io__Deserializer_unpackSerializable;
    epv->f_unpackBoolArray              = 
      remote_sidl_io__Deserializer_unpackBoolArray;
    epv->f_unpackCharArray              = 
      remote_sidl_io__Deserializer_unpackCharArray;
    epv->f_unpackIntArray               = 
      remote_sidl_io__Deserializer_unpackIntArray;
    epv->f_unpackLongArray              = 
      remote_sidl_io__Deserializer_unpackLongArray;
    epv->f_unpackOpaqueArray            = 
      remote_sidl_io__Deserializer_unpackOpaqueArray;
    epv->f_unpackFloatArray             = 
      remote_sidl_io__Deserializer_unpackFloatArray;
    epv->f_unpackDoubleArray            = 
      remote_sidl_io__Deserializer_unpackDoubleArray;
    epv->f_unpackFcomplexArray          = 
      remote_sidl_io__Deserializer_unpackFcomplexArray;
    epv->f_unpackDcomplexArray          = 
      remote_sidl_io__Deserializer_unpackDcomplexArray;
    epv->f_unpackStringArray            = 
      remote_sidl_io__Deserializer_unpackStringArray;
    epv->f_unpackGenericArray           = 
      remote_sidl_io__Deserializer_unpackGenericArray;
    epv->f_unpackSerializableArray      = 
      remote_sidl_io__Deserializer_unpackSerializableArray;

    e0->f__cast        = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e0->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
    e0->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
    e0->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
    e0->f__isRemote    = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e0->f__set_hooks   = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e0->f__exec        = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e0->f_addRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e0->f_deleteRef    = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e0->f_isSame       = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e0->f_isType       = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e1->f__cast                   = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e1->f__delete                 = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__delete;
    e1->f__getURL                 = (char* (*)(void*,
      sidl_BaseInterface*)) epv->f__getURL;
    e1->f__raddRef                = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e1->f__isRemote               = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e1->f__set_hooks              = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e1->f__exec                   = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_addRef                  = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e1->f_deleteRef               = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e1->f_isSame                  = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e1->f_isType                  = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo            = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
    e1->f_unpackBool              = (void (*)(void*,const char*,sidl_bool*,
      struct sidl_BaseInterface__object **)) epv->f_unpackBool;
    e1->f_unpackChar              = (void (*)(void*,const char*,char*,
      struct sidl_BaseInterface__object **)) epv->f_unpackChar;
    e1->f_unpackInt               = (void (*)(void*,const char*,int32_t*,
      struct sidl_BaseInterface__object **)) epv->f_unpackInt;
    e1->f_unpackLong              = (void (*)(void*,const char*,int64_t*,
      struct sidl_BaseInterface__object **)) epv->f_unpackLong;
    e1->f_unpackOpaque            = (void (*)(void*,const char*,void**,
      struct sidl_BaseInterface__object **)) epv->f_unpackOpaque;
    e1->f_unpackFloat             = (void (*)(void*,const char*,float*,
      struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
    e1->f_unpackDouble            = (void (*)(void*,const char*,double*,
      struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
    e1->f_unpackFcomplex          = (void (*)(void*,const char*,
      struct sidl_fcomplex*,
      struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
    e1->f_unpackDcomplex          = (void (*)(void*,const char*,
      struct sidl_dcomplex*,
      struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
    e1->f_unpackString            = (void (*)(void*,const char*,char**,
      struct sidl_BaseInterface__object **)) epv->f_unpackString;
    e1->f_unpackSerializable      = (void (*)(void*,const char*,
      struct sidl_io_Serializable__object**,
      struct sidl_BaseInterface__object **)) epv->f_unpackSerializable;
    e1->f_unpackBoolArray         = (void (*)(void*,const char*,
      struct sidl_bool__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackBoolArray;
    e1->f_unpackCharArray         = (void (*)(void*,const char*,
      struct sidl_char__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackCharArray;
    e1->f_unpackIntArray          = (void (*)(void*,const char*,
      struct sidl_int__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackIntArray;
    e1->f_unpackLongArray         = (void (*)(void*,const char*,
      struct sidl_long__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackLongArray;
    e1->f_unpackOpaqueArray       = (void (*)(void*,const char*,
      struct sidl_opaque__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackOpaqueArray;
    e1->f_unpackFloatArray        = (void (*)(void*,const char*,
      struct sidl_float__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackFloatArray;
    e1->f_unpackDoubleArray       = (void (*)(void*,const char*,
      struct sidl_double__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackDoubleArray;
    e1->f_unpackFcomplexArray     = (void (*)(void*,const char*,
      struct sidl_fcomplex__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackFcomplexArray;
    e1->f_unpackDcomplexArray     = (void (*)(void*,const char*,
      struct sidl_dcomplex__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackDcomplexArray;
    e1->f_unpackStringArray       = (void (*)(void*,const char*,
      struct sidl_string__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackStringArray;
    e1->f_unpackGenericArray      = (void (*)(void*,const char*,
      struct sidl__array**,
      struct sidl_BaseInterface__object **)) epv->f_unpackGenericArray;
    e1->f_unpackSerializableArray = (void (*)(void*,const char*,
      struct sidl_io_Serializable__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackSerializableArray;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct sidl_io_Deserializer__object*
  sidl_io_Deserializer__remoteConnect(const char *url, sidl_bool ar,
    sidl_BaseInterface *_ex)
  {
    struct sidl_io__Deserializer__object* self;

    struct sidl_io__Deserializer__object* s0;

    struct sidl_io__Deserializer__remote* r_obj;
    sidl_rmi_InstanceHandle instance = NULL;
    char* objectID = NULL;
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      sidl_BaseInterface bi = 
        (sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
        objectID, _ex);
      if(ar) {
        sidl_BaseInterface_addRef(bi, _ex);
      }
      return sidl_io_Deserializer__rmicast(bi, _ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
    if ( instance == NULL) { return NULL; }
    self =
      (struct sidl_io__Deserializer__object*) malloc(
        sizeof(struct sidl_io__Deserializer__object));

    r_obj =
      (struct sidl_io__Deserializer__remote*) malloc(
        sizeof(struct sidl_io__Deserializer__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                 self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_io__Deserializer__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
    s0->d_sidl_io_deserializer.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_io__deserializer;

    self->d_data = (void*) r_obj;

    return sidl_io_Deserializer__rmicast(self, _ex);
  }
  // Create an instance that uses an already existing 
  // InstanceHandel to connect to an existing remote object.
  static struct sidl_io_Deserializer__object*
  sidl_io_Deserializer__IHConnect(sidl_rmi_InstanceHandle instance,
    sidl_BaseInterface *_ex)
  {
    struct sidl_io__Deserializer__object* self;

    struct sidl_io__Deserializer__object* s0;

    struct sidl_io__Deserializer__remote* r_obj;
    self =
      (struct sidl_io__Deserializer__object*) malloc(
        sizeof(struct sidl_io__Deserializer__object));

    r_obj =
      (struct sidl_io__Deserializer__remote*) malloc(
        sizeof(struct sidl_io__Deserializer__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                 self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_io__Deserializer__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
    s0->d_sidl_io_deserializer.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_io__deserializer;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance, _ex);
    return sidl_io_Deserializer__rmicast(self, _ex);
  }
  // 
  // Cast method for interface and class type conversions.
  // 
  struct sidl_io_Deserializer__object*
  sidl_io_Deserializer__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct sidl_io_Deserializer__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Deserializer",
        (void*)sidl_io_Deserializer__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct sidl_io_Deserializer__object*) (*base->d_epv->f__cast)(
        base->d_object,
        "sidl.io.Deserializer", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct sidl_io_Deserializer__object*
  sidl_io_Deserializer__connectI(const char* url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex)
  {
    return sidl_io_Deserializer__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
sidl::io::Deserializer::throwException0(
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
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackBool( /* in */const ::std::string& key,
  /* out */bool& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_bool _local_value;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackBool))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &_local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value = _local_value;
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackChar( /* in */const ::std::string& key,
  /* out */char& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackChar))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackInt( /* in */const ::std::string& key,
  /* out */int32_t& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackInt))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackLong( /* in */const ::std::string& key,
  /* out */int64_t& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackLong))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackOpaque( /* in */const ::std::string& key,
  /* out */void*& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackOpaque))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackFloat( /* in */const ::std::string& key,
  /* out */float& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackFloat))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackDouble( /* in */const ::std::string& key,
  /* out */double& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackDouble))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackFcomplex( /* in */const ::std::string& key,
  /* out */::std::complex<float>& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_fcomplex _local_value; 
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackFcomplex))(loc_self->d_object,
    /* in */ key.c_str(), /* out */ &_local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value = ::std::complex<float>(_local_value.real, _local_value.imaginary);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackDcomplex( /* in */const ::std::string& key,
  /* out */::std::complex<double>& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_dcomplex _local_value; 
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackDcomplex))(loc_self->d_object,
    /* in */ key.c_str(), /* out */ &_local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value = ::std::complex<double>(_local_value.real, _local_value.imaginary);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackString( /* in */const ::std::string& key,
  /* out */::std::string& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  char * _local_value = 0;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackString))(loc_self->d_object, /* in */ key.c_str(),
    /* out */ &_local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  if (_local_value) {
    value = _local_value;
    ::sidl_String_free( _local_value);
  } else {
    value = "";
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackSerializable( /* in */const ::std::string& key,
  /* out */::sidl::io::Serializable& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_io_Serializable__object* _local_value;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackSerializable))(loc_self->d_object,
    /* in */ key.c_str(), /* out */ &_local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  if ( value._not_nil() ) {
    value.deleteRef();
  }
  value._set_ior( _local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}



/**
 *  unpack arrays of values 
 * It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The rarray flag should be set if the array being
 * passed in is actually an rarray.  The semantics are slightly
 * different for rarrays.  The passed in array MUST be reused,
 * even if the array has changed bounds.
 */
void
sidl::io::Deserializer::unpackBoolArray( /* in */const ::std::string& key,
  /* out array<bool> */::sidl::array<bool>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_bool__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackBoolArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<bool> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackCharArray( /* in */const ::std::string& key,
  /* out array<char> */::sidl::array<char>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_char__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackCharArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<char> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackIntArray( /* in */const ::std::string& key,
  /* out array<int> */::sidl::array<int32_t>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_int__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackIntArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<int> */ &_local_value, /* in */ ordering,
    /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackLongArray( /* in */const ::std::string& key,
  /* out array<long> */::sidl::array<int64_t>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_long__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackLongArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<long> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackOpaqueArray( /* in */const ::std::string& key,
  /* out array<opaque> */::sidl::array<void*>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_opaque__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackOpaqueArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<opaque> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackFloatArray( /* in */const ::std::string& key,
  /* out array<float> */::sidl::array<float>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_float__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackFloatArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<float> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackDoubleArray( /* in */const ::std::string& key,
  /* out array<double> */::sidl::array<double>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_double__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackDoubleArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<double> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackFcomplexArray( /* in */const ::std::string& key,
  /* out array<fcomplex> */::sidl::array< ::sidl::fcomplex>& value,
  /* in */int32_t ordering, /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_fcomplex__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackFcomplexArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<fcomplex> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackDcomplexArray( /* in */const ::std::string& key,
  /* out array<dcomplex> */::sidl::array< ::sidl::dcomplex>& value,
  /* in */int32_t ordering, /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_dcomplex__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackDcomplexArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<dcomplex> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackStringArray( /* in */const ::std::string& key,
  /* out array<string> */::sidl::array< ::std::string>& value,
  /* in */int32_t ordering, /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_string__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackStringArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<string> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackGenericArray( /* in */const ::std::string& key,
  /* out array<> */::sidl::basearray& value )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl__array* _local_value;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackGenericArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<> */ &_local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Deserializer::unpackSerializableArray( /* in */const ::std::string& 
  key,
  /* out array<sidl.io.Serializable> */::sidl::array< 
  ::sidl::io::Serializable>& value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool isRarray )

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  struct sidl_io_Serializable__array* _local_value;
  sidl_bool _local_isRarray = isRarray;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_unpackSerializableArray))(loc_self->d_object,
    /* in */ key.c_str(), /* out array<sidl.io.Serializable> */ &_local_value,
    /* in */ ordering, /* in */ dimen, /* in */ _local_isRarray, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  value._set_ior(_local_value);
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// remote connector 2
::sidl::io::Deserializer
sidl::io::Deserializer::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  sidl_BaseInterface__object* _ex = 0;
  ior_self = sidl_io_Deserializer__remoteConnect( url.c_str(), ar?TRUE:FALSE,
    &_ex );
  if (_ex != 0 ) {
    ; //TODO: handle exception
  }
  return ::sidl::io::Deserializer( ior_self, false );
}

// copy constructor
sidl::io::Deserializer::Deserializer ( const ::sidl::io::Deserializer& original 
  ) {
  d_self = ::sidl::io::Deserializer::_cast(original._get_ior());
  d_weak_reference = false;
}

// assignment operator
::sidl::io::Deserializer&
sidl::io::Deserializer::operator=( const ::sidl::io::Deserializer& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = ::sidl::io::Deserializer::_cast(rhs._get_ior());
    // note _cast incremements the reference count
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
sidl::io::Deserializer::Deserializer ( ::sidl::io::Deserializer::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
sidl::io::Deserializer::Deserializer ( ::sidl::io::Deserializer::ior_t* ior,
  bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// exec has special argument passing to avoid #include circularities
void ::sidl::io::Deserializer::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::sidl::io::Deserializer::ior_t* const loc_self = _get_ior();
  struct sidl_BaseInterface__object *throwaway_exception;
  (*loc_self->d_epv->f__exec)(loc_self->d_object,
                                methodName.c_str(),
                                inArgs._get_ior(),
                                outArgs._get_ior(),
                                &throwaway_exception);
}


/**
 * Get the URL of the Implementation of this object (for RMI)
 */
::std::string
sidl::io::Deserializer::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  char * _local_result;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _local_result = (*(loc_self->d_epv->f__getURL))(loc_self->d_object,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  if (_local_result) {
    _result = _local_result;
    ::sidl_String_free( _local_result );
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}


/**
 * Method to set whether or not method hooks should be invoked.
 */
void
sidl::io::Deserializer::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::io::Deserializer::_cast((void*)(_get_ior()));
  sidl_bool _local_on = on;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self->d_object, /* in */ _local_on,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
}

// protected method that implements casting
struct sidl_io_Deserializer__object* sidl::io::Deserializer::_cast(const void* 
  src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Deserializer",
      (void*)sidl_io_Deserializer__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object,
      "sidl.io.Deserializer", &throwaway_exception));
  }
  return cast;
}

