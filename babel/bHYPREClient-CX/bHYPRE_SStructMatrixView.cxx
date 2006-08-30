// 
// File:          bHYPRE_SStructMatrixView.cxx
// Symbol:        bHYPRE.SStructMatrixView-v1.0.0
// Symbol Type:   interface
// Babel Version: 1.0.0
// Description:   Client-side glue code for bHYPRE.SStructMatrixView
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_SStructMatrixView_hxx
#include "bHYPRE_SStructMatrixView.hxx"
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
#ifndef included_bHYPRE_SStructGraph_hxx
#include "bHYPRE_SStructGraph.hxx"
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
static struct sidl_recursive_mutex_t bHYPRE__SStructMatrixView__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE__SStructMatrixView__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE__SStructMatrixView__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE__SStructMatrixView__mutex )==EDEADLOCK) */
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

  static struct bHYPRE__SStructMatrixView__epv 
    s_rem_epv__bhypre__sstructmatrixview;

  static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

  static struct bHYPRE_ProblemDefinition__epv 
    s_rem_epv__bhypre_problemdefinition;

  static struct bHYPRE_SStructMatrixVectorView__epv 
    s_rem_epv__bhypre_sstructmatrixvectorview;

  static struct bHYPRE_SStructMatrixView__epv 
    s_rem_epv__bhypre_sstructmatrixview;

  static struct sidl_BaseInterface__epv s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_bHYPRE__SStructMatrixView__cast(
    struct bHYPRE__SStructMatrixView__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1,
      cmp2;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp0 = strcmp(name, "bHYPRE.SStructMatrixView");
    if (!cmp0) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_sstructmatrixview);
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
      else if (cmp1 > 0) {
        cmp2 = strcmp(name, "bHYPRE.SStructMatrixVectorView");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_sstructmatrixvectorview);
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
        cmp2 = strcmp(name, "bHYPRE._SStructMatrixView");
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_bHYPRE__SStructMatrixView__delete(
    struct bHYPRE__SStructMatrixView__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_bHYPRE__SStructMatrixView__getURL(
    struct bHYPRE__SStructMatrixView__object* self, sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_bHYPRE__SStructMatrixView__raddRef(
    struct bHYPRE__SStructMatrixView__object* self,sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
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
  remote_bHYPRE__SStructMatrixView__isRemote(
      struct bHYPRE__SStructMatrixView__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_bHYPRE__SStructMatrixView__set_hooks(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView._set_hooks.", &throwaway_exception);
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
  static void remote_bHYPRE__SStructMatrixView__exec(
    struct bHYPRE__SStructMatrixView__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:SetGraph
  static int32_t
  remote_bHYPRE__SStructMatrixView_SetGraph(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ struct bHYPRE_SStructGraph__object* graph,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetGraph", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(graph){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)graph,
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "graph", _url,
          _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "graph", NULL,
          _ex);SIDL_CHECK(*_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetGraph.", &throwaway_exception);
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

  // REMOTE METHOD STUB:SetValues
  static int32_t
  remote_bHYPRE__SStructMatrixView_SetValues(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nentries] */ struct sidl_double__array* values,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "index", index,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "entries", entries,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetValues.", &throwaway_exception);
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

  // REMOTE METHOD STUB:SetBoxValues
  static int32_t
  remote_bHYPRE__SStructMatrixView_SetBoxValues(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetBoxValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "entries", entries,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetBoxValues.", &throwaway_exception);
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

  // REMOTE METHOD STUB:AddToValues
  static int32_t
  remote_bHYPRE__SStructMatrixView_AddToValues(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* index,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nentries] */ struct sidl_double__array* values,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "AddToValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "index", index,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "entries", entries,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.AddToValues.", &throwaway_exception);
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

  // REMOTE METHOD STUB:AddToBoxValues
  static int32_t
  remote_bHYPRE__SStructMatrixView_AddToBoxValues(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ int32_t part,
    /* in rarray[dim] */ struct sidl_int__array* ilower,
    /* in rarray[dim] */ struct sidl_int__array* iupper,
    /* in */ int32_t var,
    /* in rarray[nentries] */ struct sidl_int__array* entries,
    /* in rarray[nvalues] */ struct sidl_double__array* values,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "AddToBoxValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "entries", entries,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.AddToBoxValues.", &throwaway_exception);
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

  // REMOTE METHOD STUB:SetSymmetric
  static int32_t
  remote_bHYPRE__SStructMatrixView_SetSymmetric(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ int32_t part,
    /* in */ int32_t var,
    /* in */ int32_t to_var,
    /* in */ int32_t symmetric,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetSymmetric", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "to_var", to_var,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "symmetric", symmetric,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetSymmetric.", &throwaway_exception);
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

  // REMOTE METHOD STUB:SetNSSymmetric
  static int32_t
  remote_bHYPRE__SStructMatrixView_SetNSSymmetric(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ int32_t symmetric,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetNSSymmetric", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "symmetric", symmetric,
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetNSSymmetric.", &throwaway_exception);
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

  // REMOTE METHOD STUB:SetComplex
  static int32_t
  remote_bHYPRE__SStructMatrixView_SetComplex(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetComplex", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetComplex.", &throwaway_exception);
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

  // REMOTE METHOD STUB:Print
  static int32_t
  remote_bHYPRE__SStructMatrixView_Print(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* in */ const char* filename,
    /* in */ int32_t all,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Print", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "filename", filename,
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "all", all, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.Print.", &throwaway_exception);
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

  // REMOTE METHOD STUB:GetObject
  static int32_t
  remote_bHYPRE__SStructMatrixView_GetObject(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetObject", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.GetObject.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixView_SetCommunicator(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.SetCommunicator.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixView_Initialize(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Initialize", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.Initialize.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixView_Assemble(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Assemble", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.Assemble.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixView_addRef(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct bHYPRE__SStructMatrixView__remote* r_obj = (struct 
        bHYPRE__SStructMatrixView__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_bHYPRE__SStructMatrixView_deleteRef(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct bHYPRE__SStructMatrixView__remote* r_obj = (struct 
        bHYPRE__SStructMatrixView__remote*)self->d_data;
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
  remote_bHYPRE__SStructMatrixView_isSame(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.isSame.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixView_isType(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.isType.", &throwaway_exception);
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
  remote_bHYPRE__SStructMatrixView_getClassInfo(
    /* in */ struct bHYPRE__SStructMatrixView__object* self ,
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
        bHYPRE__SStructMatrixView__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE._SStructMatrixView.getClassInfo.", &throwaway_exception);
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
  static void bHYPRE__SStructMatrixView__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct bHYPRE__SStructMatrixView__epv*      epv = 
      &s_rem_epv__bhypre__sstructmatrixview;
    struct bHYPRE_MatrixVectorView__epv*        e0  = 
      &s_rem_epv__bhypre_matrixvectorview;
    struct bHYPRE_ProblemDefinition__epv*       e1  = 
      &s_rem_epv__bhypre_problemdefinition;
    struct bHYPRE_SStructMatrixVectorView__epv* e2  = 
      &s_rem_epv__bhypre_sstructmatrixvectorview;
    struct bHYPRE_SStructMatrixView__epv*       e3  = 
      &s_rem_epv__bhypre_sstructmatrixview;
    struct sidl_BaseInterface__epv*             e4  = 
      &s_rem_epv__sidl_baseinterface;

    epv->f__cast                = remote_bHYPRE__SStructMatrixView__cast;
    epv->f__delete              = remote_bHYPRE__SStructMatrixView__delete;
    epv->f__exec                = remote_bHYPRE__SStructMatrixView__exec;
    epv->f__getURL              = remote_bHYPRE__SStructMatrixView__getURL;
    epv->f__raddRef             = remote_bHYPRE__SStructMatrixView__raddRef;
    epv->f__isRemote            = remote_bHYPRE__SStructMatrixView__isRemote;
    epv->f__set_hooks           = remote_bHYPRE__SStructMatrixView__set_hooks;
    epv->f__ctor                = NULL;
    epv->f__ctor2               = NULL;
    epv->f__dtor                = NULL;
    epv->f_SetGraph             = remote_bHYPRE__SStructMatrixView_SetGraph;
    epv->f_SetValues            = remote_bHYPRE__SStructMatrixView_SetValues;
    epv->f_SetBoxValues         = remote_bHYPRE__SStructMatrixView_SetBoxValues;
    epv->f_AddToValues          = remote_bHYPRE__SStructMatrixView_AddToValues;
    epv->f_AddToBoxValues       = 
      remote_bHYPRE__SStructMatrixView_AddToBoxValues;
    epv->f_SetSymmetric         = remote_bHYPRE__SStructMatrixView_SetSymmetric;
    epv->f_SetNSSymmetric       = 
      remote_bHYPRE__SStructMatrixView_SetNSSymmetric;
    epv->f_SetComplex           = remote_bHYPRE__SStructMatrixView_SetComplex;
    epv->f_Print                = remote_bHYPRE__SStructMatrixView_Print;
    epv->f_GetObject            = remote_bHYPRE__SStructMatrixView_GetObject;
    epv->f_SetCommunicator      = 
      remote_bHYPRE__SStructMatrixView_SetCommunicator;
    epv->f_Initialize           = remote_bHYPRE__SStructMatrixView_Initialize;
    epv->f_Assemble             = remote_bHYPRE__SStructMatrixView_Assemble;
    epv->f_addRef               = remote_bHYPRE__SStructMatrixView_addRef;
    epv->f_deleteRef            = remote_bHYPRE__SStructMatrixView_deleteRef;
    epv->f_isSame               = remote_bHYPRE__SStructMatrixView_isSame;
    epv->f_isType               = remote_bHYPRE__SStructMatrixView_isType;
    epv->f_getClassInfo         = remote_bHYPRE__SStructMatrixView_getClassInfo;

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

    e3->f__cast           = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e3->f__delete         = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__delete;
    e3->f__getURL         = (char* (*)(void*,
      sidl_BaseInterface*)) epv->f__getURL;
    e3->f__raddRef        = (void (*)(void*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e3->f__isRemote       = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e3->f__set_hooks      = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e3->f__exec           = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e3->f_SetGraph        = (int32_t (*)(void*,
      struct bHYPRE_SStructGraph__object*,
      struct sidl_BaseInterface__object **)) epv->f_SetGraph;
    e3->f_SetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
      int32_t,struct sidl_int__array*,struct sidl_double__array*,
      struct sidl_BaseInterface__object **)) epv->f_SetValues;
    e3->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
      struct sidl_int__array*,int32_t,struct sidl_int__array*,
      struct sidl_double__array*,
      struct sidl_BaseInterface__object **)) epv->f_SetBoxValues;
    e3->f_AddToValues     = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
      int32_t,struct sidl_int__array*,struct sidl_double__array*,
      struct sidl_BaseInterface__object **)) epv->f_AddToValues;
    e3->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
      struct sidl_int__array*,int32_t,struct sidl_int__array*,
      struct sidl_double__array*,
      struct sidl_BaseInterface__object **)) epv->f_AddToBoxValues;
    e3->f_SetSymmetric    = (int32_t (*)(void*,int32_t,int32_t,int32_t,int32_t,
      struct sidl_BaseInterface__object **)) epv->f_SetSymmetric;
    e3->f_SetNSSymmetric  = (int32_t (*)(void*,int32_t,
      struct sidl_BaseInterface__object **)) epv->f_SetNSSymmetric;
    e3->f_SetComplex      = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_SetComplex;
    e3->f_Print           = (int32_t (*)(void*,const char*,int32_t,
      struct sidl_BaseInterface__object **)) epv->f_Print;
    e3->f_GetObject       = (int32_t (*)(void*,
      struct sidl_BaseInterface__object**,
      struct sidl_BaseInterface__object **)) epv->f_GetObject;
    e3->f_SetCommunicator = (int32_t (*)(void*,
      struct bHYPRE_MPICommunicator__object*,
      struct sidl_BaseInterface__object **)) epv->f_SetCommunicator;
    e3->f_Initialize      = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Initialize;
    e3->f_Assemble        = (int32_t (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_Assemble;
    e3->f_addRef          = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e3->f_deleteRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e3->f_isSame          = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e3->f_isType          = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e3->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e4->f__cast        = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e4->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
    e4->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
    e4->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
    e4->f__isRemote    = (sidl_bool (*)(void*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e4->f__set_hooks   = (void (*)(void*,int32_t,
      sidl_BaseInterface*)) epv->f__set_hooks;
    e4->f__exec        = (void (*)(void*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,
      struct sidl_BaseInterface__object **)) epv->f__exec;
    e4->f_addRef       = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_addRef;
    e4->f_deleteRef    = (void (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_deleteRef;
    e4->f_isSame       = (sidl_bool (*)(void*,
      struct sidl_BaseInterface__object*,
      struct sidl_BaseInterface__object **)) epv->f_isSame;
    e4->f_isType       = (sidl_bool (*)(void*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_isType;
    e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct bHYPRE_SStructMatrixView__object*
  bHYPRE_SStructMatrixView__remoteConnect(const char *url, sidl_bool ar,
    sidl_BaseInterface *_ex)
  {
    struct bHYPRE__SStructMatrixView__object* self;

    struct bHYPRE__SStructMatrixView__object* s0;

    struct bHYPRE__SStructMatrixView__remote* r_obj;
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
      return bHYPRE_SStructMatrixView__rmicast(bi, _ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex );
    if ( instance == NULL) { return NULL; }
    self =
      (struct bHYPRE__SStructMatrixView__object*) malloc(
        sizeof(struct bHYPRE__SStructMatrixView__object));

    r_obj =
      (struct bHYPRE__SStructMatrixView__remote*) malloc(
        sizeof(struct bHYPRE__SStructMatrixView__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                     self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE__SStructMatrixView__init_remote_epv();
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

    s0->d_bhypre_sstructmatrixview.d_epv    = 
      &s_rem_epv__bhypre_sstructmatrixview;
    s0->d_bhypre_sstructmatrixview.d_object = (void*) self;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre__sstructmatrixview;

    self->d_data = (void*) r_obj;

    return bHYPRE_SStructMatrixView__rmicast(self, _ex);
  }
  // Create an instance that uses an already existing 
  // InstanceHandel to connect to an existing remote object.
  static struct bHYPRE_SStructMatrixView__object*
  bHYPRE_SStructMatrixView__IHConnect(sidl_rmi_InstanceHandle instance,
    sidl_BaseInterface *_ex)
  {
    struct bHYPRE__SStructMatrixView__object* self;

    struct bHYPRE__SStructMatrixView__object* s0;

    struct bHYPRE__SStructMatrixView__remote* r_obj;
    self =
      (struct bHYPRE__SStructMatrixView__object*) malloc(
        sizeof(struct bHYPRE__SStructMatrixView__object));

    r_obj =
      (struct bHYPRE__SStructMatrixView__remote*) malloc(
        sizeof(struct bHYPRE__SStructMatrixView__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                     self;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE__SStructMatrixView__init_remote_epv();
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

    s0->d_bhypre_sstructmatrixview.d_epv    = 
      &s_rem_epv__bhypre_sstructmatrixview;
    s0->d_bhypre_sstructmatrixview.d_object = (void*) self;

    s0->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s0->d_sidl_baseinterface.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre__sstructmatrixview;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance, _ex);
    return bHYPRE_SStructMatrixView__rmicast(self, _ex);
  }
  // 
  // Cast method for interface and class type conversions.
  // 
  struct bHYPRE_SStructMatrixView__object*
  bHYPRE_SStructMatrixView__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct bHYPRE_SStructMatrixView__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructMatrixView",
        (void*)bHYPRE_SStructMatrixView__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct bHYPRE_SStructMatrixView__object*) (*base->d_epv->f__cast)(
        base->d_object,
        "bHYPRE.SStructMatrixView", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct bHYPRE_SStructMatrixView__object*
  bHYPRE_SStructMatrixView__connectI(const char* url, sidl_bool ar,
    struct sidl_BaseInterface__object **_ex)
  {
    return bHYPRE_SStructMatrixView__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
bHYPRE::SStructMatrixView::throwException0(
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
 * Set the matrix graph.
 * DEPRECATED     Use Create
 */
int32_t
bHYPRE::SStructMatrixView::SetGraph( /* in */::bHYPRE::SStructGraph graph )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  struct bHYPRE_SStructGraph__object* _local_graph = graph._get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetGraph))(loc_self->d_object,
    /* in */ _local_graph, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::SetValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* index, /* in */int32_t dim, /* in */int32_t var,
  /* in */int32_t nentries, /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nentries] */double* values )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array index_real;
  struct sidl_int__array *index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nentries-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self->d_object, /* in */ part,
    /* in rarray[dim] */ index_tmp, /* in */ var,
    /* in rarray[nentries] */ entries_tmp, /* in rarray[nentries] */ values_tmp,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::SetValues( /* in */int32_t part,
  /* in rarray[dim] */::sidl::array<int32_t> index, /* in */int32_t var,
  /* in rarray[nentries] */::sidl::array<int32_t> entries,
  /* in rarray[nentries] */::sidl::array<double> values )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self->d_object, /* in */ part,
    /* in rarray[dim] */ index._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nentries] */ values._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::SetBoxValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* ilower, /* in rarray[dim] */int32_t* iupper,
  /* in */int32_t dim, /* in */int32_t var, /* in */int32_t nentries,
  /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nvalues] */double* values, /* in */int32_t nvalues )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array ilower_real;
  struct sidl_int__array *ilower_tmp = &ilower_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_int__array iupper_real;
  struct sidl_int__array *iupper_tmp = &iupper_real;
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetBoxValues))(loc_self->d_object,
    /* in */ part, /* in rarray[dim] */ ilower_tmp,
    /* in rarray[dim] */ iupper_tmp, /* in */ var,
    /* in rarray[nentries] */ entries_tmp, /* in rarray[nvalues] */ values_tmp,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Set matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type (there are no such restrictions for
 * non-stencil entries).
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::SetBoxValues( /* in */int32_t part,
  /* in rarray[dim] */::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::sidl::array<int32_t> iupper, /* in */int32_t var,
  /* in rarray[nentries] */::sidl::array<int32_t> entries,
  /* in rarray[nvalues] */::sidl::array<double> values )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetBoxValues))(loc_self->d_object,
    /* in */ part, /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nvalues] */ values._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::AddToValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* index, /* in */int32_t dim, /* in */int32_t var,
  /* in */int32_t nentries, /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nentries] */double* values )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  int32_t index_lower[1], index_upper[1], index_stride[1];
  struct sidl_int__array index_real;
  struct sidl_int__array *index_tmp = &index_real;
  index_upper[0] = dim-1;
  sidl_int__array_init(index, index_tmp, 1, index_lower, index_upper,
    index_stride);
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nentries-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self->d_object,
    /* in */ part, /* in rarray[dim] */ index_tmp, /* in */ var,
    /* in rarray[nentries] */ entries_tmp, /* in rarray[nentries] */ values_tmp,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of the same
 * type: either stencil or non-stencil, but not both.  Also, if
 * they are stencil entries, they must all represent couplings
 * to the same variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::AddToValues( /* in */int32_t part,
  /* in rarray[dim] */::sidl::array<int32_t> index, /* in */int32_t var,
  /* in rarray[nentries] */::sidl::array<int32_t> entries,
  /* in rarray[nentries] */::sidl::array<double> values )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self->d_object,
    /* in */ part, /* in rarray[dim] */ index._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nentries] */ values._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::AddToBoxValues( /* in */int32_t part,
  /* in rarray[dim] */int32_t* ilower, /* in rarray[dim] */int32_t* iupper,
  /* in */int32_t dim, /* in */int32_t var, /* in */int32_t nentries,
  /* in rarray[nentries] */int32_t* entries,
  /* in rarray[nvalues] */double* values, /* in */int32_t nvalues )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  int32_t ilower_lower[1], ilower_upper[1], ilower_stride[1];
  struct sidl_int__array ilower_real;
  struct sidl_int__array *ilower_tmp = &ilower_real;
  ilower_upper[0] = dim-1;
  sidl_int__array_init(ilower, ilower_tmp, 1, ilower_lower, ilower_upper,
    ilower_stride);
  int32_t iupper_lower[1], iupper_upper[1], iupper_stride[1];
  struct sidl_int__array iupper_real;
  struct sidl_int__array *iupper_tmp = &iupper_real;
  iupper_upper[0] = dim-1;
  sidl_int__array_init(iupper, iupper_tmp, 1, iupper_lower, iupper_upper,
    iupper_stride);
  int32_t entries_lower[1], entries_upper[1], entries_stride[1];
  struct sidl_int__array entries_real;
  struct sidl_int__array *entries_tmp = &entries_real;
  entries_upper[0] = nentries-1;
  sidl_int__array_init(entries, entries_tmp, 1, entries_lower, entries_upper,
    entries_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper,
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToBoxValues))(loc_self->d_object,
    /* in */ part, /* in rarray[dim] */ ilower_tmp,
    /* in rarray[dim] */ iupper_tmp, /* in */ var,
    /* in rarray[nentries] */ entries_tmp, /* in rarray[nvalues] */ values_tmp,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Add to matrix coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * NOTE: The entries in this routine must all be of stencil
 * type.  Also, they must all represent couplings to the same
 * variable type.
 * 
 * If the matrix is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */
int32_t
bHYPRE::SStructMatrixView::AddToBoxValues( /* in */int32_t part,
  /* in rarray[dim] */::sidl::array<int32_t> ilower,
  /* in rarray[dim] */::sidl::array<int32_t> iupper, /* in */int32_t var,
  /* in rarray[nentries] */::sidl::array<int32_t> entries,
  /* in rarray[nvalues] */::sidl::array<double> values )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToBoxValues))(loc_self->d_object,
    /* in */ part, /* in rarray[dim] */ ilower._get_ior(),
    /* in rarray[dim] */ iupper._get_ior(), /* in */ var,
    /* in rarray[nentries] */ entries._get_ior(),
    /* in rarray[nvalues] */ values._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Define symmetry properties for the stencil entries in the
 * matrix.  The boolean argument {\tt symmetric} is applied to
 * stencil entries on part {\tt part} that couple variable {\tt
 * var} to variable {\tt to\_var}.  A value of -1 may be used
 * for {\tt part}, {\tt var}, or {\tt to\_var} to specify
 * ``all''.  For example, if {\tt part} and {\tt to\_var} are
 * set to -1, then the boolean is applied to stencil entries on
 * all parts that couple variable {\tt var} to all other
 * variables.
 * 
 * By default, matrices are assumed to be nonsymmetric.
 * Significant storage savings can be made if the matrix is
 * symmetric.
 */
int32_t
bHYPRE::SStructMatrixView::SetSymmetric( /* in */int32_t part,
  /* in */int32_t var, /* in */int32_t to_var, /* in */int32_t symmetric )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetSymmetric))(loc_self->d_object,
    /* in */ part, /* in */ var, /* in */ to_var, /* in */ symmetric,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Define symmetry properties for all non-stencil matrix
 * entries.
 */
int32_t
bHYPRE::SStructMatrixView::SetNSSymmetric( /* in */int32_t symmetric )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetNSSymmetric))(loc_self->d_object,
    /* in */ symmetric, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Set the matrix to be complex.
 */
int32_t
bHYPRE::SStructMatrixView::SetComplex(  )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetComplex))(loc_self->d_object,
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  {  struct sidl_BaseInterface__object *throwaway_exception;  
    (*loc_self->d_epv->f_deleteRef)(loc_self->d_object, &throwaway_exception);
  }/*unpack results and cleanup*/
  return _result;
}



/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 */
int32_t
bHYPRE::SStructMatrixView::Print( /* in */const ::std::string& filename,
  /* in */int32_t all )

{
  int32_t _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Print))(loc_self->d_object,
    /* in */ filename.c_str(), /* in */ all, &_exception );
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
::bHYPRE::SStructMatrixView
bHYPRE::SStructMatrixView::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  sidl_BaseInterface__object* _ex = 0;
  ior_self = bHYPRE_SStructMatrixView__remoteConnect( url.c_str(),
    ar?TRUE:FALSE, &_ex );
  if (_ex != 0 ) {
    ; //TODO: handle exception
  }
  return ::bHYPRE::SStructMatrixView( ior_self, false );
}

// copy constructor
bHYPRE::SStructMatrixView::SStructMatrixView ( const 
  ::bHYPRE::SStructMatrixView& original ) {
  d_self = ::bHYPRE::SStructMatrixView::_cast(original._get_ior());
  d_weak_reference = false;
}

// assignment operator
::bHYPRE::SStructMatrixView&
bHYPRE::SStructMatrixView::operator=( const ::bHYPRE::SStructMatrixView& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = ::bHYPRE::SStructMatrixView::_cast(rhs._get_ior());
    // note _cast incremements the reference count
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
bHYPRE::SStructMatrixView::SStructMatrixView ( 
  ::bHYPRE::SStructMatrixView::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { }

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
bHYPRE::SStructMatrixView::SStructMatrixView ( 
  ::bHYPRE::SStructMatrixView::ior_t* ior, bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
}

// exec has special argument passing to avoid #include circularities
void ::bHYPRE::SStructMatrixView::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::bHYPRE::SStructMatrixView::ior_t* const loc_self = _get_ior();
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
bHYPRE::SStructMatrixView::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
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
bHYPRE::SStructMatrixView::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = (ior_t*) 
    ::bHYPRE::SStructMatrixView::_cast((void*)(_get_ior()));
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
struct bHYPRE_SStructMatrixView__object* bHYPRE::SStructMatrixView::_cast(const 
  void* src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructMatrixView",
      (void*)bHYPRE_SStructMatrixView__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object,
      "bHYPRE.SStructMatrixView", &throwaway_exception));
  }
  return cast;
}

