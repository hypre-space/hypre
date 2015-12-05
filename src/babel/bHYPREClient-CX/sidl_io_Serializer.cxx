// 
// File:          sidl_io_Serializer.cxx
// Symbol:        sidl.io.Serializer-v0.9.15
// Symbol Type:   interface
// Babel Version: 1.0.4
// Release:       $Name: V2-4-0b $
// Revision:      @(#) $Id: sidl_io_Serializer.cxx,v 1.4 2007/09/27 19:55:46 painter Exp $
// Description:   Client-side glue code for sidl.io.Serializer
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

#ifndef included_sidl_io_Serializer_hxx
#include "sidl_io_Serializer.hxx"
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
static struct sidl_recursive_mutex_t sidl_io__Serializer__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_io__Serializer__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_io__Serializer__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_io__Serializer__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

  // Static variables to hold version of IOR
  static const int32_t s_IOR_MAJOR_VERSION = 1;
  static const int32_t s_IOR_MINOR_VERSION = 0;

  // Static variables for managing EPV initialization.
  static int s_remote_initialized = 0;

  static struct sidl_io__Serializer__epv s_rem_epv__sidl_io__serializer;

  static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

  static struct sidl_io_Serializer__epv s_rem_epv__sidl_io_serializer;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_sidl_io__Serializer__cast(
    struct sidl_io__Serializer__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp0 = strcmp(name, "sidl.io.Serializer");
    if (!cmp0) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_io_serializer);
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
      cmp1 = strcmp(name, "sidl.io._Serializer");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct sidl_io__Serializer__object*)self);
        return cast;
      }
    }
    if ((*self->d_epv->f_isType)(self,name, _ex)) {
      void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**) = 
        (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
          sidl_BaseInterface__object**)) 
        sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
      cast =  (*func)(((struct sidl_io__Serializer__remote*)self->d_data)->d_ih,
        _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_sidl_io__Serializer__delete(
    struct sidl_io__Serializer__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_sidl_io__Serializer__getURL(
    struct sidl_io__Serializer__object* self, sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_sidl_io__Serializer__raddRef(
    struct sidl_io__Serializer__object* self,sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      sidl_io__Serializer__remote*)self->d_data)->d_ih;
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
  remote_sidl_io__Serializer__isRemote(
      struct sidl_io__Serializer__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_sidl_io__Serializer__set_hooks(
    /* in */ struct sidl_io__Serializer__object* self ,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer._set_hooks.", &throwaway_exception);
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
  static void remote_sidl_io__Serializer__exec(
    struct sidl_io__Serializer__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_sidl_io__Serializer_addRef(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_io__Serializer__remote* r_obj = (struct 
        sidl_io__Serializer__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_sidl_io__Serializer_deleteRef(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_io__Serializer__remote* r_obj = (struct 
        sidl_io__Serializer__remote*)self->d_data;
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
  remote_sidl_io__Serializer_isSame(
    /* in */ struct sidl_io__Serializer__object* self ,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isSame", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(iobj){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.isSame.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_sidl_io__Serializer_isType(
    /* in */ struct sidl_io__Serializer__object* self ,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.isType.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_sidl_io__Serializer_getClassInfo(
    /* in */ struct sidl_io__Serializer__object* self ,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.getClassInfo.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str, 
        _ex);SIDL_CHECK(*_ex);
      _retval = sidl_ClassInfo__connectI(_retval_str, FALSE, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:packBool
  static void
  remote_sidl_io__Serializer_packBool(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ sidl_bool value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packBool", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packBool.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packChar
  static void
  remote_sidl_io__Serializer_packChar(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ char value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packChar", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packChar( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packChar.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packInt
  static void
  remote_sidl_io__Serializer_packInt(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ int32_t value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packInt", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packInt.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packLong
  static void
  remote_sidl_io__Serializer_packLong(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ int64_t value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packLong", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packLong( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packLong.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packOpaque
  static void
  remote_sidl_io__Serializer_packOpaque(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ void* value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packOpaque", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packOpaque( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packOpaque.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packFloat
  static void
  remote_sidl_io__Serializer_packFloat(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ float value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packFloat", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packFloat( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFloat.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packDouble
  static void
  remote_sidl_io__Serializer_packDouble(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ double value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packDouble", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDouble.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packFcomplex
  static void
  remote_sidl_io__Serializer_packFcomplex(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ struct sidl_fcomplex value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packFcomplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packFcomplex( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFcomplex.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packDcomplex
  static void
  remote_sidl_io__Serializer_packDcomplex(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ struct sidl_dcomplex value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packDcomplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDcomplex( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDcomplex.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packString
  static void
  remote_sidl_io__Serializer_packString(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ const char* value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packString", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packString.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packSerializable
  static void
  remote_sidl_io__Serializer_packSerializable(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in */ struct sidl_io_Serializable__object* value,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packSerializable", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      if(value){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)value, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "value", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "value", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packSerializable.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packBoolArray
  static void
  remote_sidl_io__Serializer_packBoolArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<bool> */ struct sidl_bool__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packBoolArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBoolArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packBoolArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packCharArray
  static void
  remote_sidl_io__Serializer_packCharArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<char> */ struct sidl_char__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packCharArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packCharArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packCharArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packIntArray
  static void
  remote_sidl_io__Serializer_packIntArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<int> */ struct sidl_int__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packIntArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packIntArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packLongArray
  static void
  remote_sidl_io__Serializer_packLongArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<long> */ struct sidl_long__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packLongArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packLongArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packLongArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packOpaqueArray
  static void
  remote_sidl_io__Serializer_packOpaqueArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<opaque> */ struct sidl_opaque__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packOpaqueArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packOpaqueArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packOpaqueArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packFloatArray
  static void
  remote_sidl_io__Serializer_packFloatArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<float> */ struct sidl_float__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packFloatArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packFloatArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFloatArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packDoubleArray
  static void
  remote_sidl_io__Serializer_packDoubleArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<double> */ struct sidl_double__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packDoubleArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDoubleArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packFcomplexArray
  static void
  remote_sidl_io__Serializer_packFcomplexArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<fcomplex> */ struct sidl_fcomplex__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packFcomplexArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packFcomplexArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packFcomplexArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packDcomplexArray
  static void
  remote_sidl_io__Serializer_packDcomplexArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<dcomplex> */ struct sidl_dcomplex__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packDcomplexArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDcomplexArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packDcomplexArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packStringArray
  static void
  remote_sidl_io__Serializer_packStringArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<string> */ struct sidl_string__array* value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packStringArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packStringArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packStringArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packGenericArray
  static void
  remote_sidl_io__Serializer_packGenericArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<> */ struct sidl__array* value,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packGenericArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packGenericArray( _inv, "value", value,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packGenericArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:packSerializableArray
  static void
  remote_sidl_io__Serializer_packSerializableArray(
    /* in */ struct sidl_io__Serializer__object* self ,
    /* in */ const char* key,
    /* in array<sidl.io.Serializable> */ struct sidl_io_Serializable__array* 
      value,
    /* in */ int32_t ordering,
    /* in */ int32_t dimen,
    /* in */ sidl_bool reuse_array,
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
        sidl_io__Serializer__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "packSerializableArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packSerializableArray( _inv, "value", value,0,0,0, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "ordering", ordering, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "dimen", dimen, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packBool( _inv, "reuse_array", reuse_array, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.io._Serializer.packSerializableArray.", &throwaway_exception);
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

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void sidl_io__Serializer__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct sidl_io__Serializer__epv* epv = &s_rem_epv__sidl_io__serializer;
    struct sidl_BaseInterface__epv*  e0  = &s_rem_epv__sidl_baseinterface;
    struct sidl_io_Serializer__epv*  e1  = &s_rem_epv__sidl_io_serializer;

    epv->f__cast                      = remote_sidl_io__Serializer__cast;
    epv->f__delete                    = remote_sidl_io__Serializer__delete;
    epv->f__exec                      = remote_sidl_io__Serializer__exec;
    epv->f__getURL                    = remote_sidl_io__Serializer__getURL;
    epv->f__raddRef                   = remote_sidl_io__Serializer__raddRef;
    epv->f__isRemote                  = remote_sidl_io__Serializer__isRemote;
    epv->f__set_hooks                 = remote_sidl_io__Serializer__set_hooks;
    epv->f__ctor                      = NULL;
    epv->f__ctor2                     = NULL;
    epv->f__dtor                      = NULL;
    epv->f_addRef                     = remote_sidl_io__Serializer_addRef;
    epv->f_deleteRef                  = remote_sidl_io__Serializer_deleteRef;
    epv->f_isSame                     = remote_sidl_io__Serializer_isSame;
    epv->f_isType                     = remote_sidl_io__Serializer_isType;
    epv->f_getClassInfo               = remote_sidl_io__Serializer_getClassInfo;
    epv->f_packBool                   = remote_sidl_io__Serializer_packBool;
    epv->f_packChar                   = remote_sidl_io__Serializer_packChar;
    epv->f_packInt                    = remote_sidl_io__Serializer_packInt;
    epv->f_packLong                   = remote_sidl_io__Serializer_packLong;
    epv->f_packOpaque                 = remote_sidl_io__Serializer_packOpaque;
    epv->f_packFloat                  = remote_sidl_io__Serializer_packFloat;
    epv->f_packDouble                 = remote_sidl_io__Serializer_packDouble;
    epv->f_packFcomplex               = remote_sidl_io__Serializer_packFcomplex;
    epv->f_packDcomplex               = remote_sidl_io__Serializer_packDcomplex;
    epv->f_packString                 = remote_sidl_io__Serializer_packString;
    epv->f_packSerializable           = 
      remote_sidl_io__Serializer_packSerializable;
    epv->f_packBoolArray              = 
      remote_sidl_io__Serializer_packBoolArray;
    epv->f_packCharArray              = 
      remote_sidl_io__Serializer_packCharArray;
    epv->f_packIntArray               = remote_sidl_io__Serializer_packIntArray;
    epv->f_packLongArray              = 
      remote_sidl_io__Serializer_packLongArray;
    epv->f_packOpaqueArray            = 
      remote_sidl_io__Serializer_packOpaqueArray;
    epv->f_packFloatArray             = 
      remote_sidl_io__Serializer_packFloatArray;
    epv->f_packDoubleArray            = 
      remote_sidl_io__Serializer_packDoubleArray;
    epv->f_packFcomplexArray          = 
      remote_sidl_io__Serializer_packFcomplexArray;
    epv->f_packDcomplexArray          = 
      remote_sidl_io__Serializer_packDcomplexArray;
    epv->f_packStringArray            = 
      remote_sidl_io__Serializer_packStringArray;
    epv->f_packGenericArray           = 
      remote_sidl_io__Serializer_packGenericArray;
    epv->f_packSerializableArray      = 
      remote_sidl_io__Serializer_packSerializableArray;

    e0->f__cast        = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
      epv->f__cast;
    e0->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
    e0->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
    e0->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
    e0->f__isRemote    = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e0->f__set_hooks   = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
      epv->f__set_hooks;
    e0->f__exec        = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e0->f_addRef       = (void (*)(void*,struct sidl_BaseInterface__object **)) 
      epv->f_addRef;
    e0->f_deleteRef    = (void (*)(void*,struct sidl_BaseInterface__object **)) 
      epv->f_deleteRef;
    e0->f_isSame       = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e0->f_isType       = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e1->f__cast                 = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e1->f__delete               = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__delete;
    e1->f__getURL               = (char* (*)(void*,sidl_BaseInterface*)) 
      epv->f__getURL;
    e1->f__raddRef              = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__raddRef;
    e1->f__isRemote             = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e1->f__set_hooks            = (void (*)(void*,int32_t, 
      sidl_BaseInterface*)) epv->f__set_hooks;
    e1->f__exec                 = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_addRef                = (void (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_addRef;
    e1->f_deleteRef             = (void (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_deleteRef;
    e1->f_isSame                = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e1->f_isType                = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo          = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
    e1->f_packBool              = (void (*)(void*,const char*,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packBool;
    e1->f_packChar              = (void (*)(void*,const char*,char,struct 
      sidl_BaseInterface__object **)) epv->f_packChar;
    e1->f_packInt               = (void (*)(void*,const char*,int32_t,struct 
      sidl_BaseInterface__object **)) epv->f_packInt;
    e1->f_packLong              = (void (*)(void*,const char*,int64_t,struct 
      sidl_BaseInterface__object **)) epv->f_packLong;
    e1->f_packOpaque            = (void (*)(void*,const char*,void*,struct 
      sidl_BaseInterface__object **)) epv->f_packOpaque;
    e1->f_packFloat             = (void (*)(void*,const char*,float,struct 
      sidl_BaseInterface__object **)) epv->f_packFloat;
    e1->f_packDouble            = (void (*)(void*,const char*,double,struct 
      sidl_BaseInterface__object **)) epv->f_packDouble;
    e1->f_packFcomplex          = (void (*)(void*,const char*,struct 
      sidl_fcomplex,struct sidl_BaseInterface__object **)) epv->f_packFcomplex;
    e1->f_packDcomplex          = (void (*)(void*,const char*,struct 
      sidl_dcomplex,struct sidl_BaseInterface__object **)) epv->f_packDcomplex;
    e1->f_packString            = (void (*)(void*,const char*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_packString;
    e1->f_packSerializable      = (void (*)(void*,const char*,struct 
      sidl_io_Serializable__object*,struct sidl_BaseInterface__object **)) 
      epv->f_packSerializable;
    e1->f_packBoolArray         = (void (*)(void*,const char*,struct 
      sidl_bool__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packBoolArray;
    e1->f_packCharArray         = (void (*)(void*,const char*,struct 
      sidl_char__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packCharArray;
    e1->f_packIntArray          = (void (*)(void*,const char*,struct 
      sidl_int__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packIntArray;
    e1->f_packLongArray         = (void (*)(void*,const char*,struct 
      sidl_long__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packLongArray;
    e1->f_packOpaqueArray       = (void (*)(void*,const char*,struct 
      sidl_opaque__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packOpaqueArray;
    e1->f_packFloatArray        = (void (*)(void*,const char*,struct 
      sidl_float__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packFloatArray;
    e1->f_packDoubleArray       = (void (*)(void*,const char*,struct 
      sidl_double__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packDoubleArray;
    e1->f_packFcomplexArray     = (void (*)(void*,const char*,struct 
      sidl_fcomplex__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packFcomplexArray;
    e1->f_packDcomplexArray     = (void (*)(void*,const char*,struct 
      sidl_dcomplex__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packDcomplexArray;
    e1->f_packStringArray       = (void (*)(void*,const char*,struct 
      sidl_string__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packStringArray;
    e1->f_packGenericArray      = (void (*)(void*,const char*,struct 
      sidl__array*,sidl_bool,struct sidl_BaseInterface__object **)) 
      epv->f_packGenericArray;
    e1->f_packSerializableArray = (void (*)(void*,const char*,struct 
      sidl_io_Serializable__array*,int32_t,int32_t,sidl_bool,struct 
      sidl_BaseInterface__object **)) epv->f_packSerializableArray;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct sidl_io_Serializer__object*
  sidl_io_Serializer__remoteConnect(const char *url, sidl_bool ar, 
    sidl_BaseInterface *_ex)
  {
    struct sidl_io__Serializer__object* self;

    struct sidl_io__Serializer__object* s0;

    struct sidl_io__Serializer__remote* r_obj;
    sidl_rmi_InstanceHandle instance = NULL;
    char* objectID = NULL;
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      sidl_BaseInterface bi = (
        sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
        objectID, _ex);
      if(ar) {
        sidl_BaseInterface_addRef(bi, _ex);
      }
      return sidl_io_Serializer__rmicast(bi, _ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
    if ( instance == NULL) { return NULL; }
    self =
      (struct sidl_io__Serializer__object*) malloc(
        sizeof(struct sidl_io__Serializer__object));

    r_obj =
      (struct sidl_io__Serializer__remote*) malloc(
        sizeof(struct sidl_io__Serializer__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                               self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_io__Serializer__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_sidl_io_serializer.d_epv    = &s_rem_epv__sidl_io_serializer;
    s0->d_sidl_io_serializer.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_io__serializer;

    self->d_data = (void*) r_obj;

    return sidl_io_Serializer__rmicast(self, _ex);
  }
  // Create an instance that uses an already existing 
  // InstanceHandel to connect to an existing remote object.
  static struct sidl_io_Serializer__object*
  sidl_io_Serializer__IHConnect(sidl_rmi_InstanceHandle instance, 
    sidl_BaseInterface *_ex)
  {
    struct sidl_io__Serializer__object* self;

    struct sidl_io__Serializer__object* s0;

    struct sidl_io__Serializer__remote* r_obj;
    self =
      (struct sidl_io__Serializer__object*) malloc(
        sizeof(struct sidl_io__Serializer__object));

    r_obj =
      (struct sidl_io__Serializer__remote*) malloc(
        sizeof(struct sidl_io__Serializer__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                               self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_io__Serializer__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_sidl_io_serializer.d_epv    = &s_rem_epv__sidl_io_serializer;
    s0->d_sidl_io_serializer.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_io__serializer;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance, _ex);
    return sidl_io_Serializer__rmicast(self, _ex);
  }
  // 
  // Cast method for interface and class type conversions.
  // 
  struct sidl_io_Serializer__object*
  sidl_io_Serializer__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct sidl_io_Serializer__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Serializer", (
        void*)sidl_io_Serializer__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct sidl_io_Serializer__object*) (*base->d_epv->f__cast)(
        base->d_object,
        "sidl.io.Serializer", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct sidl_io_Serializer__object*
  sidl_io_Serializer__connectI(const char* url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex)
  {
    return sidl_io_Serializer__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
sidl::io::Serializer::throwException0(
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
sidl::io::Serializer::packBool( /* in */const ::std::string& key, /* in */bool 
  value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_value = value;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packBool))(loc_self->d_object, /* in */ key.c_str(), /* 
    in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packChar( /* in */const ::std::string& key, /* in */char 
  value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packChar))(loc_self->d_object, /* in */ key.c_str(), /* 
    in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packInt( /* in */const ::std::string& key, /* in 
  */int32_t value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packInt))(loc_self->d_object, /* in */ key.c_str(), /* 
    in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packLong( /* in */const ::std::string& key, /* in 
  */int64_t value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packLong))(loc_self->d_object, /* in */ key.c_str(), /* 
    in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packOpaque( /* in */const ::std::string& key, /* in 
  */void* value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packOpaque))(loc_self->d_object, /* in */ key.c_str(), 
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packFloat( /* in */const ::std::string& key, /* in 
  */float value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packFloat))(loc_self->d_object, /* in */ key.c_str(), 
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packDouble( /* in */const ::std::string& key, /* in 
  */double value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packDouble))(loc_self->d_object, /* in */ key.c_str(), 
    /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packFcomplex( /* in */const ::std::string& key, /* in 
  */const ::std::complex<float>& value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  struct sidl_fcomplex _local_value = {value.real(), value.imag() } ; 
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packFcomplex))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packDcomplex( /* in */const ::std::string& key, /* in 
  */const ::std::complex<double>& value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  struct sidl_dcomplex _local_value = {value.real(), value.imag() } ; 
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packDcomplex))(loc_self->d_object, /* in */ key.c_str(),
    /* in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packString( /* in */const ::std::string& key, /* in 
  */const ::std::string& value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packString))(loc_self->d_object, /* in */ key.c_str(), 
    /* in */ value.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packSerializable( /* in */const ::std::string& key, /* in 
  */::sidl::io::Serializable value )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  struct sidl_io_Serializable__object* _local_value = (struct 
    sidl_io_Serializable__object*) value.::sidl::io::Serializable::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packSerializable))(loc_self->d_object, /* in */ 
    key.c_str(), /* in */ _local_value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}



/**
 *  
 * pack arrays of values.  It is possible to ensure an array is
 * in a certain order by passing in ordering and dimension
 * requirements.  ordering should represent a value in the
 * sidl_array_ordering enumeration in sidlArray.h If either
 * argument is 0, it means there is no restriction on that
 * aspect.  The boolean reuse_array flag is set to true if the
 * remote unserializer should try to reuse the array that is
 * passed into it or not.
 */
void
sidl::io::Serializer::packBoolArray( /* in */const ::std::string& key, /* in 
  array<bool> */::sidl::array<bool> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packBoolArray))(loc_self->d_object, /* in */ key.c_str(
    ), /* in array<bool> */ value._get_ior(), /* in */ ordering, /* in */ dimen,
    /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packCharArray( /* in */const ::std::string& key, /* in 
  array<char> */::sidl::array<char> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packCharArray))(loc_self->d_object, /* in */ key.c_str(
    ), /* in array<char> */ value._get_ior(), /* in */ ordering, /* in */ dimen,
    /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packIntArray( /* in */const ::std::string& key, /* in 
  array<int> */::sidl::array<int32_t> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packIntArray))(loc_self->d_object, /* in */ key.c_str(),
    /* in array<int> */ value._get_ior(), /* in */ ordering, /* in */ dimen, /* 
    in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packLongArray( /* in */const ::std::string& key, /* in 
  array<long> */::sidl::array<int64_t> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packLongArray))(loc_self->d_object, /* in */ key.c_str(
    ), /* in array<long> */ value._get_ior(), /* in */ ordering, /* in */ dimen,
    /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packOpaqueArray( /* in */const ::std::string& key, /* in 
  array<opaque> */::sidl::array<void*> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packOpaqueArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<opaque> */ value._get_ior(), /* in */ ordering, /* 
    in */ dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packFloatArray( /* in */const ::std::string& key, /* in 
  array<float> */::sidl::array<float> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packFloatArray))(loc_self->d_object, /* in */ key.c_str(
    ), /* in array<float> */ value._get_ior(), /* in */ ordering, /* in */ 
    dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packDoubleArray( /* in */const ::std::string& key, /* in 
  array<double> */::sidl::array<double> value, /* in */int32_t ordering, /* in 
  */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packDoubleArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<double> */ value._get_ior(), /* in */ ordering, /* 
    in */ dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packFcomplexArray( /* in */const ::std::string& key, /* 
  in array<fcomplex> */::sidl::array< ::sidl::fcomplex> value, /* in */int32_t 
  ordering, /* in */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packFcomplexArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<fcomplex> */ value._get_ior(), /* in */ ordering, 
    /* in */ dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packDcomplexArray( /* in */const ::std::string& key, /* 
  in array<dcomplex> */::sidl::array< ::sidl::dcomplex> value, /* in */int32_t 
  ordering, /* in */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packDcomplexArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<dcomplex> */ value._get_ior(), /* in */ ordering, 
    /* in */ dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packStringArray( /* in */const ::std::string& key, /* in 
  array<string> */::sidl::array< ::std::string> value, /* in */int32_t ordering,
  /* in */int32_t dimen, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packStringArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<string> */ value._get_ior(), /* in */ ordering, /* 
    in */ dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packGenericArray( /* in */const ::std::string& key, /* in 
  array<> */::sidl::basearray value, /* in */bool reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packGenericArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<> */ value._get_baseior(), /* in */ 
    _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * user defined non-static method.
 */
void
sidl::io::Serializer::packSerializableArray( /* in */const ::std::string& key, 
  /* in array<sidl.io.Serializable> */::sidl::array< ::sidl::io::Serializable> 
  value, /* in */int32_t ordering, /* in */int32_t dimen, /* in */bool 
  reuse_array )

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_reuse_array = reuse_array;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_packSerializableArray))(loc_self->d_object, /* in */ 
    key.c_str(), /* in array<sidl.io.Serializable> */ value._get_ior(), /* in 
    */ ordering, /* in */ dimen, /* in */ _local_reuse_array, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// remote connector
::sidl::io::Serializer
sidl::io::Serializer::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception;
  ior_self = sidl_io_Serializer__remoteConnect( url.c_str(), ar?TRUE:FALSE, 
    &_exception );
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  return ::sidl::io::Serializer( ior_self, false );
}

// copy constructor
sidl::io::Serializer::Serializer ( const ::sidl::io::Serializer& original ) {
  d_self = (struct sidl_io_Serializer__object*) 
    original.::sidl::io::Serializer::_get_ior();
  sidl_io_Serializer_IORCache = (ior_t*) d_self;
  if(d_self) {
    addRef();
  }
  d_weak_reference = false;
}

// assignment operator
::sidl::io::Serializer&
sidl::io::Serializer::operator=( const ::sidl::io::Serializer& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = (struct sidl_io_Serializer__object*) 
      rhs.::sidl::io::Serializer::_get_ior();
    sidl_io_Serializer_IORCache = (ior_t*) d_self;
    if(d_self) {
      addRef();
    }
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
sidl::io::Serializer::Serializer ( ::sidl::io::Serializer::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { 
  sidl_io_Serializer_IORCache = (ior_t*) d_self;
}

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
sidl::io::Serializer::Serializer ( ::sidl::io::Serializer::ior_t* ior, bool 
  isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
  sidl_io_Serializer_IORCache = (ior_t*) d_self;
}

// exec has special argument passing to avoid #include circularities
void ::sidl::io::Serializer::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::sidl::io::Serializer::ior_t* const loc_self = _get_ior();
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
sidl::io::Serializer::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
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
  /*unpack results and cleanup*/
  return _result;
}


/**
 * Method to set whether or not method hooks should be invoked.
 */
void
sidl::io::Serializer::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct sidl_io_Serializer__object*) 
    ::sidl::io::Serializer::_get_ior();
  sidl_bool _local_on = on;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self->d_object, /* in */ _local_on, 
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}

// protected method that implements casting
struct sidl_io_Serializer__object* sidl::io::Serializer::_cast(const void* src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("sidl.io.Serializer", (
      void*)sidl_io_Serializer__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object, 
      "sidl.io.Serializer", &throwaway_exception));
  }
  return cast;
}

