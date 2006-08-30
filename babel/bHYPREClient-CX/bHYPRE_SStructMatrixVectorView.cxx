// 
// File:          bHYPRE_SStructMatrixVectorView.cxx
// Symbol:        bHYPRE.SStructMatrixVectorView-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructMatrixVectorView
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructMatrixVectorView_hxx
#include "bHYPRE_SStructMatrixVectorView.hxx"
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
#include "babel_config.h"
#ifdef SIDL_DYNAMIC_LIBRARY
#include <stdio.h>
#include <stdlib.h>
#include "sidl_Loader.hxx"
#include "sidl_DLL.hxx"
#endif
// 
// Includes for all method dependencies.
// 
#ifndef included_bHYPRE_MPICommunicator_hxx
#include "bHYPRE_MPICommunicator.hxx"
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
static struct sidl_recursive_mutex_t bHYPRE__SStructMatrixVectorView__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE__SStructMatrixVectorView__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE__SStructMatrixVectorView__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE__SStructMatrixVectorView__mutex )==EDEADLOCK) */
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

  static struct bHYPRE__SStructMatrixVectorView__epv 
    s_rem_epv__bhypre__sstructmatrixvectorview;

  static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

  static struct bHYPRE_ProblemDefinition__epv 
    s_rem_epv__bhypre_problemdefinition;

  static struct bHYPRE_SStructMatrixVectorView__epv 
    s_rem_epv__bhypre_sstructmatrixvectorview;

  static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_bHYPRE__SStructMatrixVectorView__cast(
    struct bHYPRE__SStructMatrixVectorView__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1,
      cmp2;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp0 = strcmp(name, "bHYPRE.SStructMatrixVectorView");
    if (!cmp0) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_sstructmatrixvectorview);
      return cast;
    }
    else if (cmp0 < 0) {
      cmp1 = strcmp(name, "bHYPRE.ProblemDefinition");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_problemdefinition);
        return cast;
      }
      else if (cmp1 < 0) {
        cmp2 = strcmp(name, "bHYPRE.MatrixVectorView");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_matrixvectorview);
          return cast;
        }
      }
    }
    else if (cmp0 > 0) {
      cmp1 = strcmp(name, "sidl.BaseInterface");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_sidl_baseinterface);
        return cast;
      }
      else if (cmp1 < 0) {
        cmp2 = strcmp(name, "bHYPRE._SStructMatrixVectorView");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = self;
          return cast;
        }
      }
    }
    if ((*self->d_epv->f_isType)(self,name, _ex)) {
      void* (*func)(struct sidl_rmi_InstanceHandle__object*,
        struct sidl_BaseInterface__object**) = 
        (void* (*)(struct sidl_rmi_InstanceHandle__object*,
          struct sidl_BaseInterface__object**)) 
        sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
      cast =  (*func)(((struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_bHYPRE__SStructMatrixVectorView__delete(
    struct bHYPRE__SStructMatrixVectorView__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_bHYPRE__SStructMatrixVectorView__getURL(
    struct bHYPRE__SStructMatrixVectorView__object* self,
      sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_bHYPRE__SStructMatrixVectorView__raddRef(
    struct bHYPRE__SStructMatrixVectorView__object* self,
      sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
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
  remote_bHYPRE__SStructMatrixVectorView__isRemote(
      struct bHYPRE__SStructMatrixVectorView__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_bHYPRE__SStructMatrixVectorView__set_hooks(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
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
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView._set_hooks.", &throwaway_exception);
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
  static void remote_bHYPRE__SStructMatrixVectorView__exec(
    struct bHYPRE__SStructMatrixVectorView__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:GetObject
  static int32_t
  remote_bHYPRE__SStructMatrixVectorView_GetObject(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
    /* out */ struct sidl_BaseInterface__object** A,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char* A_str= NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetObject", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.GetObject.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackString( _rsvp, "A", &A_str, _ex);SIDL_CHECK(*_ex);
      *A = sidl_BaseInterface__connectI(A_str, FALSE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetCommunicator
  static int32_t
  remote_bHYPRE__SStructMatrixVectorView_SetCommunicator(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetCommunicator", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(mpi_comm){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)mpi_comm,
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "mpi_comm", _url,
          _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "mpi_comm", NULL,
          _ex);SIDL_CHECK(*_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.SetCommunicator.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:Initialize
  static int32_t
  remote_bHYPRE__SStructMatrixVectorView_Initialize(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Initialize", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.Initialize.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:Assemble
  static int32_t
  remote_bHYPRE__SStructMatrixVectorView_Assemble(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Assemble", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.Assemble.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval,
        _ex);SIDL_CHECK(*_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:addRef
  static void
  remote_bHYPRE__SStructMatrixVectorView_addRef(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct bHYPRE__SStructMatrixVectorView__remote* r_obj = (struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_bHYPRE__SStructMatrixVectorView_deleteRef(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct bHYPRE__SStructMatrixVectorView__remote* r_obj = (struct 
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data;
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
  remote_bHYPRE__SStructMatrixVectorView_isSame(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
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
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.isSame.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixVectorView_isType(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
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
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.isType.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixVectorView_getClassInfo(
    /* in */ struct bHYPRE__SStructMatrixVectorView__object* self ,
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
        bHYPRE__SStructMatrixVectorView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixVectorView.getClassInfo.", &throwaway_exception);
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

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void bHYPRE__SStructMatrixVectorView__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct bHYPRE__SStructMatrixVectorView__epv* epv = 
      &s_rem_epv__bhypre__sstructmatrixvectorview;
    struct bHYPRE_MatrixVectorView__epv*         e0  = 
      &s_rem_epv__bhypre_matrixvectorview;
    struct bHYPRE_ProblemDefinition__epv*        e1  = 
      &s_rem_epv__bhypre_problemdefinition;
    struct bHYPRE_SStructMatrixVectorView__epv*  e2  = 
      &s_rem_epv__bhypre_sstructmatrixvectorview;
    struct sidl_BaseInterface__epv*              e3  = 
      &s_rem_epv__sidl_baseinterface;

    epv->f__cast                = remote_bHYPRE__SStructMatrixVectorView__cast;
    epv->f__delete              = 
      remote_bHYPRE__SStructMatrixVectorView__delete;
    epv->f__exec                = remote_bHYPRE__SStructMatrixVectorView__exec;
    epv->f__getURL              = 
      remote_bHYPRE__SStructMatrixVectorView__getURL;
    epv->f__raddRef             = 
      remote_bHYPRE__SStructMatrixVectorView__raddRef;
    epv->f__isRemote            = 
      remote_bHYPRE__SStructMatrixVectorView__isRemote;
    epv->f__set_hooks           = 
      remote_bHYPRE__SStructMatrixVectorView__set_hooks;
    epv->f__ctor                = NULL;
    epv->f__ctor2               = NULL;
    epv->f__dtor                = NULL;
    epv->f_GetObject            = 
      remote_bHYPRE__SStructMatrixVectorView_GetObject;
    epv->f_SetCommunicator      = 
      remote_bHYPRE__SStructMatrixVectorView_SetCommunicator;
    epv->f_Initialize           = 
      remote_bHYPRE__SStructMatrixVectorView_Initialize;
    epv->f_Assemble             = 
      remote_bHYPRE__SStructMatrixVectorView_Assemble;
    epv->f_addRef               = remote_bHYPRE__SStructMatrixVectorView_addRef;
    epv->f_deleteRef            = 
      remote_bHYPRE__SStructMatrixVectorView_deleteRef;
    epv->f_isSame               = remote_bHYPRE__SStructMatrixVectorView_isSame;
    epv->f_isType               = remote_bHYPRE__SStructMatrixVectorView_isType;
    epv->f_getClassInfo         = 
      remote_bHYPRE__SStructMatrixVectorView_getClassInfo;

    e0->f__cast           = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e0->f__delete         = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__delete;
    e0->f__getURL         = (char* (*)(void*,
      sidl_BaseInterface*)) epv->f__getURL;
    e0->f__raddRef        = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e0->f__isRemote       = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e0->f__set_hooks      = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e0->f__exec           = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e0->f_SetCommunicator = (int32_t (*)(void*,
      struct bHYPRE_MPICommunicator__object*,
      struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
    e0->f_Initialize      = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Initialize;
    e0->f_Assemble        = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Assemble;
    e0->f_addRef          = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e0->f_deleteRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e0->f_isSame          = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e0->f_isType          = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e1->f__cast           = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e1->f__delete         = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__delete;
    e1->f__getURL         = (char* (*)(void*,
      sidl_BaseInterface*)) epv->f__getURL;
    e1->f__raddRef        = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e1->f__isRemote       = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e1->f__set_hooks      = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e1->f__exec           = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_SetCommunicator = (int32_t (*)(void*,
      struct bHYPRE_MPICommunicator__object*,
      struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
    e1->f_Initialize      = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Initialize;
    e1->f_Assemble        = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Assemble;
    e1->f_addRef          = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e1->f_deleteRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e1->f_isSame          = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e1->f_isType          = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e2->f__cast           = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e2->f__delete         = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__delete;
    e2->f__getURL         = (char* (*)(void*,
      sidl_BaseInterface*)) epv->f__getURL;
    e2->f__raddRef        = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e2->f__isRemote       = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e2->f__set_hooks      = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e2->f__exec           = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e2->f_GetObject       = (int32_t (*)(void*,
      struct sidl_BaseInterface__object**,
      struct sidl_BaseInterface__object **)) epv->f_GetObject;
    e2->f_SetCommunicator = (int32_t (*)(void*,
      struct bHYPRE_MPICommunicator__object*,
      struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
    e2->f_Initialize      = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Initialize;
    e2->f_Assemble        = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Assemble;
    e2->f_addRef          = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e2->f_deleteRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e2->f_isSame          = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e2->f_isType          = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e3->f__cast        = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e3->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
    e3->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
    e3->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
    e3->f__isRemote    = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e3->f__set_hooks   = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e3->f__exec        = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e3->f_addRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e3->f_deleteRef    = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e3->f_isSame       = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e3->f_isType       = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e3->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct bHYPRE_SStructMatrixVectorView__object*
  bHYPRE_SStructMatrixVectorView__remoteConnect(const char *url, sidl_bool ar,
    sidl_BaseInterface *_ex)
  {
    struct bHYPRE__SStructMatrixVectorView__object* self;

    struct bHYPRE__SStructMatrixVectorView__object* s0;

    struct bHYPRE__SStructMatrixVectorView__remote* r_obj;
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
      return bHYPRE_SStructMatrixVectorView__rmicast(bi, _ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
    if ( instance == NULL) { return NULL; }
    self =
      (struct bHYPRE__SStructMatrixVectorView__object*) malloc(
        sizeof(struct bHYPRE__SStructMatrixVectorView__object));

    r_obj =
      (struct bHYPRE__SStructMatrixVectorView__remote*) malloc(
        sizeof(struct bHYPRE__SStructMatrixVectorView__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                           self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE__SStructMatrixVectorView__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_bhypre_matrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_matrixvectorview;
    s0->d_bhypre_matrixvectorview.d_object = (void*) self;

    s0->d_bhypre_problemdefinition.d_epv    = 
      &s_rem_epv__bhypre_problemdefinition;
    s0->d_bhypre_problemdefinition.d_object = (void*) self;

    s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_sstructmatrixvectorview;
    s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre__sstructmatrixvectorview;

    self->d_data = (void*) r_obj;

    return bHYPRE_SStructMatrixVectorView__rmicast(self, _ex);
  }
  // Create an instance that uses an already existing 
  // InstanceHandel to connect to an existing remote object.
  static struct bHYPRE_SStructMatrixVectorView__object*
  bHYPRE_SStructMatrixVectorView__IHConnect(sidl_rmi_InstanceHandle instance,
    sidl_BaseInterface *_ex)
  {
    struct bHYPRE__SStructMatrixVectorView__object* self;

    struct bHYPRE__SStructMatrixVectorView__object* s0;

    struct bHYPRE__SStructMatrixVectorView__remote* r_obj;
    self =
      (struct bHYPRE__SStructMatrixVectorView__object*) malloc(
        sizeof(struct bHYPRE__SStructMatrixVectorView__object));

    r_obj =
      (struct bHYPRE__SStructMatrixVectorView__remote*) malloc(
        sizeof(struct bHYPRE__SStructMatrixVectorView__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                           self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE__SStructMatrixVectorView__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s0->d_bhypre_matrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_matrixvectorview;
    s0->d_bhypre_matrixvectorview.d_object = (void*) self;

    s0->d_bhypre_problemdefinition.d_epv    = 
      &s_rem_epv__bhypre_problemdefinition;
    s0->d_bhypre_problemdefinition.d_object = (void*) self;

    s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_sstructmatrixvectorview;
    s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre__sstructmatrixvectorview;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance, _ex);
    return bHYPRE_SStructMatrixVectorView__rmicast(self, _ex);
  }
  // 
  // Cast method for interface and class type conversions.
  // 
  struct bHYPRE_SStructMatrixVectorView__object*
  bHYPRE_SStructMatrixVectorView__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct bHYPRE_SStructMatrixVectorView__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructMatrixVectorView",
        (void*)bHYPRE_SStructMatrixVectorView__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct bHYPRE_SStructMatrixVectorView__object*) 
        (*base->d_epv->f__cast)(
        base->d_object,
        "bHYPRE.SStructMatrixVectorView", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct bHYPRE_SStructMatrixVectorView__object*
  bHYPRE_SStructMatrixVectorView__connectI(const char* url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex)
  {
    return bHYPRE_SStructMatrixVectorView__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
bHYPRE::SStructMatrixVectorView::throwException0(
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
 * A semi-structured matrix or vector contains a Struct or IJ matrix
 * or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * A cast must be used on the returned object to convert it into a known type.
 */
int32_t
bHYPRE::SStructMatrixVectorView::GetObject( /* out */::sidl::BaseInterface& A )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixVectorView::_cast((void*)(_get_ior()));
  struct sidl_BaseInterface__object* _local_A;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetObject))(loc_self->d_object,
    /* out */ &_local_A, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  if ( A._not_nil() ) {
    A.deleteRef();
  }
  A._set_ior( _local_A);
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
::bHYPRE::SStructMatrixVectorView
bHYPRE::SStructMatrixVectorView::_connect(const std::string& url,
  const bool ar ) {
  ior_t* ior_self;
  sidl_BaseInterface__object* _ex = 0;
  ior_self = bHYPRE_SStructMatrixVectorView__remoteConnect( url.c_str(),
    ar?TRUE:FALSE, &_ex );
  if (_ex != 0 ) {
    ; //TODO: handle exception
  }
  return ::bHYPRE::SStructMatrixVectorView( ior_self, false );
}

// copy constructor
bHYPRE::SStructMatrixVectorView::SStructMatrixVectorView ( const 
  ::bHYPRE::SStructMatrixVectorView& original ) {
  d_self = ::bHYPRE::SStructMatrixVectorView::_cast(original._get_ior());
  d_weak_reference = false;
}

// assignment operator
::bHYPRE::SStructMatrixVectorView&
bHYPRE::SStructMatrixVectorView::operator=( const 
  ::bHYPRE::SStructMatrixVectorView& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = ::bHYPRE::SStructMatrixVectorView::_cast(rhs._get_ior());
    // note _cast incremements the reference count
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
bHYPRE::SStructMatrixVectorView::SStructMatrixVectorView ( 
  ::bHYPRE::SStructMatrixVectorView::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
bHYPRE::SStructMatrixVectorView::SStructMatrixVectorView ( 
  ::bHYPRE::SStructMatrixVectorView::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// exec has special argument passing to avoid #include circularities
void ::bHYPRE::SStructMatrixVectorView::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::bHYPRE::SStructMatrixVectorView::ior_t* const loc_self = _get_ior();
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
bHYPRE::SStructMatrixVectorView::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixVectorView::_cast((void*)(_get_ior()));
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
bHYPRE::SStructMatrixVectorView::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixVectorView::_cast((void*)(_get_ior()));
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
struct bHYPRE_SStructMatrixVectorView__object* 
  bHYPRE::SStructMatrixVectorView::_cast(const void* src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructMatrixVectorView",
      (void*)bHYPRE_SStructMatrixVectorView__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object,
      "bHYPRE.SStructMatrixVectorView", &throwaway_exception));
  }
  return cast;
}

