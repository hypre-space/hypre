// 
// File:          sidl_rmi_Response.cxx
// Symbol:        sidl.rmi.Response-v0.9.15
// Symbol Type:   interface
// Babel Version: 1.0.0
// Release:       $Name: V1-14-0b $
// Revision:      @(#) $Id: sidl_rmi_Response.cxx,v 1.2 2006/09/14 21:52:16 painter Exp $
// Description:   Client-side glue code for sidl.rmi.Response
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

#ifndef included_sidl_rmi_Response_hxx
#include "sidl_rmi_Response.hxx"
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
#ifndef included_sidl_BaseException_hxx
#include "sidl_BaseException.hxx"
#endif
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
static struct sidl_recursive_mutex_t sidl_rmi__Response__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidl_rmi__Response__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidl_rmi__Response__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidl_rmi__Response__mutex )==EDEADLOCK) */
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

  static struct sidl_rmi__Response__epv s_rem_epv__sidl_rmi__response;

  static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;

  static struct sidl_io_Deserializer__epv s_rem_epv__sidl_io_deserializer;

  static struct sidl_rmi_Response__epv s_rem_epv__sidl_rmi_response;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_sidl_rmi__Response__cast(
    struct sidl_rmi__Response__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1,
      cmp2;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp0 = strcmp(name, "sidl.rmi.Response");
    if (!cmp0) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_rmi_response);
      return cast;
    }
    else if (cmp0 < 0) {
      cmp1 = strcmp(name, "sidl.io.Deserializer");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_io_deserializer);
        return cast;
      }
      else if (cmp1 < 0) {
        cmp2 = strcmp(name, "sidl.BaseInterface");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_sidl_baseinterface);
          return cast;
        }
      }
    }
    else if (cmp0 > 0) {
      cmp1 = strcmp(name, "sidl.rmi._Response");
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
      cast =  (*func)(((struct sidl_rmi__Response__remote*)self->d_data)->d_ih,
        _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_sidl_rmi__Response__delete(
    struct sidl_rmi__Response__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_sidl_rmi__Response__getURL(
    struct sidl_rmi__Response__object* self, sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_sidl_rmi__Response__raddRef(
    struct sidl_rmi__Response__object* self,sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
  remote_sidl_rmi__Response__isRemote(
      struct sidl_rmi__Response__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_sidl_rmi__Response__set_hooks(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response._set_hooks.", &throwaway_exception);
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
  static void remote_sidl_rmi__Response__exec(
    struct sidl_rmi__Response__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_sidl_rmi__Response_addRef(
    /* in */ struct sidl_rmi__Response__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_rmi__Response__remote* r_obj = (struct 
        sidl_rmi__Response__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_sidl_rmi__Response_deleteRef(
    /* in */ struct sidl_rmi__Response__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct sidl_rmi__Response__remote* r_obj = (struct 
        sidl_rmi__Response__remote*)self->d_data;
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
  remote_sidl_rmi__Response_isSame(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.isSame.", &throwaway_exception);
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
  remote_sidl_rmi__Response_isType(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.isType.", &throwaway_exception);
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
  remote_sidl_rmi__Response_getClassInfo(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.getClassInfo.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackBool(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackBool", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackBool.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackChar(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackChar", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackChar.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackInt(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackInt", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackInt.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackLong(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackLong", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackLong.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackOpaque(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackOpaque", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackOpaque.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackFloat(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackFloat", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFloat.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackDouble(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackDouble", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDouble.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackFcomplex(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackFcomplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFcomplex.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackDcomplex(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackDcomplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDcomplex.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackString(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackString", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackString.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackSerializable(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackSerializable", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackSerializable.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackBoolArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackBoolArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackCharArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackCharArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackIntArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackIntArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackLongArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackLongArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackOpaqueArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackOpaqueArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackFloatArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFloatArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackDoubleArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDoubleArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackFcomplexArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackFcomplexArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackDcomplexArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackDcomplexArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackStringArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackStringArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackGenericArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "unpackGenericArray", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "key", key, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackGenericArray.", &throwaway_exception);
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
  remote_sidl_rmi__Response_unpackSerializableArray(
    /* in */ struct sidl_rmi__Response__object* self ,
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
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.unpackSerializableArray.", &throwaway_exception);
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

  // REMOTE METHOD STUB:getExceptionThrown
  static struct sidl_BaseException__object*
  remote_sidl_rmi__Response_getExceptionThrown(
    /* in */ struct sidl_rmi__Response__object* self ,
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
      struct sidl_BaseException__object* _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        sidl_rmi__Response__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getExceptionThrown", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from sidl.rmi._Response.getExceptionThrown.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str,
        _ex);SIDL_CHECK(*_ex);
      _retval = sidl_BaseException__connectI(_retval_str, FALSE,
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
  static void sidl_rmi__Response__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct sidl_rmi__Response__epv*   epv = &s_rem_epv__sidl_rmi__response;
    struct sidl_BaseInterface__epv*   e0  = &s_rem_epv__sidl_baseinterface;
    struct sidl_io_Deserializer__epv* e1  = &s_rem_epv__sidl_io_deserializer;
    struct sidl_rmi_Response__epv*    e2  = &s_rem_epv__sidl_rmi_response;

    epv->f__cast                        = remote_sidl_rmi__Response__cast;
    epv->f__delete                      = remote_sidl_rmi__Response__delete;
    epv->f__exec                        = remote_sidl_rmi__Response__exec;
    epv->f__getURL                      = remote_sidl_rmi__Response__getURL;
    epv->f__raddRef                     = remote_sidl_rmi__Response__raddRef;
    epv->f__isRemote                    = remote_sidl_rmi__Response__isRemote;
    epv->f__set_hooks                   = remote_sidl_rmi__Response__set_hooks;
    epv->f__ctor                        = NULL;
    epv->f__ctor2                       = NULL;
    epv->f__dtor                        = NULL;
    epv->f_addRef                       = remote_sidl_rmi__Response_addRef;
    epv->f_deleteRef                    = remote_sidl_rmi__Response_deleteRef;
    epv->f_isSame                       = remote_sidl_rmi__Response_isSame;
    epv->f_isType                       = remote_sidl_rmi__Response_isType;
    epv->f_getClassInfo                 = 
      remote_sidl_rmi__Response_getClassInfo;
    epv->f_unpackBool                   = remote_sidl_rmi__Response_unpackBool;
    epv->f_unpackChar                   = remote_sidl_rmi__Response_unpackChar;
    epv->f_unpackInt                    = remote_sidl_rmi__Response_unpackInt;
    epv->f_unpackLong                   = remote_sidl_rmi__Response_unpackLong;
    epv->f_unpackOpaque                 = 
      remote_sidl_rmi__Response_unpackOpaque;
    epv->f_unpackFloat                  = remote_sidl_rmi__Response_unpackFloat;
    epv->f_unpackDouble                 = 
      remote_sidl_rmi__Response_unpackDouble;
    epv->f_unpackFcomplex               = 
      remote_sidl_rmi__Response_unpackFcomplex;
    epv->f_unpackDcomplex               = 
      remote_sidl_rmi__Response_unpackDcomplex;
    epv->f_unpackString                 = 
      remote_sidl_rmi__Response_unpackString;
    epv->f_unpackSerializable           = 
      remote_sidl_rmi__Response_unpackSerializable;
    epv->f_unpackBoolArray              = 
      remote_sidl_rmi__Response_unpackBoolArray;
    epv->f_unpackCharArray              = 
      remote_sidl_rmi__Response_unpackCharArray;
    epv->f_unpackIntArray               = 
      remote_sidl_rmi__Response_unpackIntArray;
    epv->f_unpackLongArray              = 
      remote_sidl_rmi__Response_unpackLongArray;
    epv->f_unpackOpaqueArray            = 
      remote_sidl_rmi__Response_unpackOpaqueArray;
    epv->f_unpackFloatArray             = 
      remote_sidl_rmi__Response_unpackFloatArray;
    epv->f_unpackDoubleArray            = 
      remote_sidl_rmi__Response_unpackDoubleArray;
    epv->f_unpackFcomplexArray          = 
      remote_sidl_rmi__Response_unpackFcomplexArray;
    epv->f_unpackDcomplexArray          = 
      remote_sidl_rmi__Response_unpackDcomplexArray;
    epv->f_unpackStringArray            = 
      remote_sidl_rmi__Response_unpackStringArray;
    epv->f_unpackGenericArray           = 
      remote_sidl_rmi__Response_unpackGenericArray;
    epv->f_unpackSerializableArray      = 
      remote_sidl_rmi__Response_unpackSerializableArray;
    epv->f_getExceptionThrown           = 
      remote_sidl_rmi__Response_getExceptionThrown;

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

    e2->f__cast                   = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e2->f__delete                 = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__delete;
    e2->f__getURL                 = (char* (*)(void*,
      sidl_BaseInterface*)) epv->f__getURL;
    e2->f__raddRef                = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e2->f__isRemote               = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e2->f__set_hooks              = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e2->f__exec                   = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e2->f_addRef                  = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e2->f_deleteRef               = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e2->f_isSame                  = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e2->f_isType                  = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e2->f_getClassInfo            = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;
    e2->f_unpackBool              = (void (*)(void*,const char*,sidl_bool*,
      struct sidl_BaseInterface__object **)) epv->f_unpackBool;
    e2->f_unpackChar              = (void (*)(void*,const char*,char*,
      struct sidl_BaseInterface__object **)) epv->f_unpackChar;
    e2->f_unpackInt               = (void (*)(void*,const char*,int32_t*,
      struct sidl_BaseInterface__object **)) epv->f_unpackInt;
    e2->f_unpackLong              = (void (*)(void*,const char*,int64_t*,
      struct sidl_BaseInterface__object **)) epv->f_unpackLong;
    e2->f_unpackOpaque            = (void (*)(void*,const char*,void**,
      struct sidl_BaseInterface__object **)) epv->f_unpackOpaque;
    e2->f_unpackFloat             = (void (*)(void*,const char*,float*,
      struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
    e2->f_unpackDouble            = (void (*)(void*,const char*,double*,
      struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
    e2->f_unpackFcomplex          = (void (*)(void*,const char*,
      struct sidl_fcomplex*,
      struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
    e2->f_unpackDcomplex          = (void (*)(void*,const char*,
      struct sidl_dcomplex*,
      struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
    e2->f_unpackString            = (void (*)(void*,const char*,char**,
      struct sidl_BaseInterface__object **)) epv->f_unpackString;
    e2->f_unpackSerializable      = (void (*)(void*,const char*,
      struct sidl_io_Serializable__object**,
      struct sidl_BaseInterface__object **)) epv->f_unpackSerializable;
    e2->f_unpackBoolArray         = (void (*)(void*,const char*,
      struct sidl_bool__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackBoolArray;
    e2->f_unpackCharArray         = (void (*)(void*,const char*,
      struct sidl_char__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackCharArray;
    e2->f_unpackIntArray          = (void (*)(void*,const char*,
      struct sidl_int__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackIntArray;
    e2->f_unpackLongArray         = (void (*)(void*,const char*,
      struct sidl_long__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackLongArray;
    e2->f_unpackOpaqueArray       = (void (*)(void*,const char*,
      struct sidl_opaque__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackOpaqueArray;
    e2->f_unpackFloatArray        = (void (*)(void*,const char*,
      struct sidl_float__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackFloatArray;
    e2->f_unpackDoubleArray       = (void (*)(void*,const char*,
      struct sidl_double__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackDoubleArray;
    e2->f_unpackFcomplexArray     = (void (*)(void*,const char*,
      struct sidl_fcomplex__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackFcomplexArray;
    e2->f_unpackDcomplexArray     = (void (*)(void*,const char*,
      struct sidl_dcomplex__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackDcomplexArray;
    e2->f_unpackStringArray       = (void (*)(void*,const char*,
      struct sidl_string__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackStringArray;
    e2->f_unpackGenericArray      = (void (*)(void*,const char*,
      struct sidl__array**,
      struct sidl_BaseInterface__object **)) epv->f_unpackGenericArray;
    e2->f_unpackSerializableArray = (void (*)(void*,const char*,
      struct sidl_io_Serializable__array**,int32_t,int32_t,sidl_bool,
      struct sidl_BaseInterface__object **)) epv->f_unpackSerializableArray;
    e2->f_getExceptionThrown      = (struct sidl_BaseException__object* 
      (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getExceptionThrown;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct sidl_rmi_Response__object*
  sidl_rmi_Response__remoteConnect(const char *url, sidl_bool ar,
    sidl_BaseInterface *_ex)
  {
    struct sidl_rmi__Response__object* self;

    struct sidl_rmi__Response__object* s0;

    struct sidl_rmi__Response__remote* r_obj;
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
      return sidl_rmi_Response__rmicast(bi, _ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
    if ( instance == NULL) { return NULL; }
    self =
      (struct sidl_rmi__Response__object*) malloc(
        sizeof(struct sidl_rmi__Response__object));

    r_obj =
      (struct sidl_rmi__Response__remote*) malloc(
        sizeof(struct sidl_rmi__Response__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                              self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_rmi__Response__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
    s0->d_sidl_io_deserializer.d_object = (void*) self;

    s0->d_sidl_rmi_response.d_epv    = &s_rem_epv__sidl_rmi_response;
    s0->d_sidl_rmi_response.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_rmi__response;

    self->d_data = (void*) r_obj;

    return sidl_rmi_Response__rmicast(self, _ex);
  }
  // Create an instance that uses an already existing 
  // InstanceHandel to connect to an existing remote object.
  static struct sidl_rmi_Response__object*
  sidl_rmi_Response__IHConnect(sidl_rmi_InstanceHandle instance,
    sidl_BaseInterface *_ex)
  {
    struct sidl_rmi__Response__object* self;

    struct sidl_rmi__Response__object* s0;

    struct sidl_rmi__Response__remote* r_obj;
    self =
      (struct sidl_rmi__Response__object*) malloc(
        sizeof(struct sidl_rmi__Response__object));

    r_obj =
      (struct sidl_rmi__Response__remote*) malloc(
        sizeof(struct sidl_rmi__Response__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                              self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      sidl_rmi__Response__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_sidl_io_deserializer.d_epv    = &s_rem_epv__sidl_io_deserializer;
    s0->d_sidl_io_deserializer.d_object = (void*) self;

    s0->d_sidl_rmi_response.d_epv    = &s_rem_epv__sidl_rmi_response;
    s0->d_sidl_rmi_response.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__sidl_rmi__response;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance, _ex);
    return sidl_rmi_Response__rmicast(self, _ex);
  }
  // 
  // Cast method for interface and class type conversions.
  // 
  struct sidl_rmi_Response__object*
  sidl_rmi_Response__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_Response__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.Response",
        (void*)sidl_rmi_Response__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct sidl_rmi_Response__object*) (*base->d_epv->f__cast)(
        base->d_object,
        "sidl.rmi.Response", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct sidl_rmi_Response__object*
  sidl_rmi_Response__connectI(const char* url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex)
  {
    return sidl_rmi_Response__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
sidl::rmi::Response::throwException0(
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
 *  
 * May return a communication exception or an execption thrown
 * from the remote server.  If it returns null, then it's safe
 * to unpack arguments
 */
::sidl::BaseException
sidl::rmi::Response::getExceptionThrown(  )

{
  ::sidl::BaseException _result;
  ior_t* const loc_self = (ior_t*) 
    ::sidl::rmi::Response::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = ::sidl::BaseException( 
    (*(loc_self->d_epv->f_getExceptionThrown))(loc_self->d_object,
    &_exception ), false);
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



//////////////////////////////////////////////////
// 
// End User Defined Methods
// (everything else in this file is specific to
//  Babel's C++ bindings)
// 

// remote connector 2
::sidl::rmi::Response
sidl::rmi::Response::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  sidl_BaseInterface__object* _ex = 0;
  ior_self = sidl_rmi_Response__remoteConnect( url.c_str(), ar?TRUE:FALSE,
    &_ex );
  if (_ex != 0 ) {
    ; //TODO: handle exception
  }
  return ::sidl::rmi::Response( ior_self, false );
}

// copy constructor
sidl::rmi::Response::Response ( const ::sidl::rmi::Response& original ) {
  d_self = ::sidl::rmi::Response::_cast(original._get_ior());
  d_weak_reference = false;
}

// assignment operator
::sidl::rmi::Response&
sidl::rmi::Response::operator=( const ::sidl::rmi::Response& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = ::sidl::rmi::Response::_cast(rhs._get_ior());
    // note _cast incremements the reference count
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
sidl::rmi::Response::Response ( ::sidl::rmi::Response::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
sidl::rmi::Response::Response ( ::sidl::rmi::Response::ior_t* ior,
  bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// exec has special argument passing to avoid #include circularities
void ::sidl::rmi::Response::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::sidl::rmi::Response::ior_t* const loc_self = _get_ior();
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
sidl::rmi::Response::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (ior_t*) 
    ::sidl::rmi::Response::_cast((void*)(_get_ior()));
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
sidl::rmi::Response::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = (ior_t*) 
    ::sidl::rmi::Response::_cast((void*)(_get_ior()));
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
struct sidl_rmi_Response__object* sidl::rmi::Response::_cast(const void* src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("sidl.rmi.Response",
      (void*)sidl_rmi_Response__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object,
      "sidl.rmi.Response", &throwaway_exception));
  }
  return cast;
}

