// 
// File:          bHYPRE_IJParCSRMatrix.cxx
// Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
// Symbol Type:   class
// Babel Version: 1.0.4
// Description:   Client-side glue code for bHYPRE.IJParCSRMatrix
// 
// WARNING: Automatically generated; changes will be lost
// 
// 

#ifndef included_bHYPRE_IJParCSRMatrix_hxx
#include "bHYPRE_IJParCSRMatrix.hxx"
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
#ifndef included_bHYPRE_IJParCSRMatrix_hxx
#include "bHYPRE_IJParCSRMatrix.hxx"
#endif
#ifndef included_bHYPRE_MPICommunicator_hxx
#include "bHYPRE_MPICommunicator.hxx"
#endif
#ifndef included_bHYPRE_Vector_hxx
#include "bHYPRE_Vector.hxx"
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
static struct sidl_recursive_mutex_t bHYPRE_IJParCSRMatrix__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_IJParCSRMatrix__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_IJParCSRMatrix__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_IJParCSRMatrix__mutex )==EDEADLOCK) */
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

  static struct bHYPRE_IJParCSRMatrix__epv s_rem_epv__bhypre_ijparcsrmatrix;

  static struct bHYPRE_CoefficientAccess__epv 
    s_rem_epv__bhypre_coefficientaccess;

  static struct bHYPRE_IJMatrixView__epv s_rem_epv__bhypre_ijmatrixview;

  static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

  static struct bHYPRE_Operator__epv s_rem_epv__bhypre_operator;

  static struct bHYPRE_ProblemDefinition__epv 
    s_rem_epv__bhypre_problemdefinition;

  static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

  static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


  // REMOTE CAST: dynamic type casting for remote objects.
  static void* remote_bHYPRE_IJParCSRMatrix__cast(
    struct bHYPRE_IJParCSRMatrix__object* self,
    const char* name, sidl_BaseInterface* _ex)
  {
    int
      cmp0,
      cmp1,
      cmp2,
      cmp3;
    void* cast = NULL;
    *_ex = NULL; /* default to no exception */
    cmp0 = strcmp(name, "bHYPRE.Operator");
    if (!cmp0) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_operator);
      return cast;
    }
    else if (cmp0 < 0) {
      cmp1 = strcmp(name, "bHYPRE.IJParCSRMatrix");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct bHYPRE_IJParCSRMatrix__object*)self);
        return cast;
      }
      else if (cmp1 < 0) {
        cmp2 = strcmp(name, "bHYPRE.IJMatrixView");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_ijmatrixview);
          return cast;
        }
        else if (cmp2 < 0) {
          cmp3 = strcmp(name, "bHYPRE.CoefficientAccess");
          if (!cmp3) {
            (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
            cast = &((*self).d_bhypre_coefficientaccess);
            return cast;
          }
        }
      }
      else if (cmp1 > 0) {
        cmp2 = strcmp(name, "bHYPRE.MatrixVectorView");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_matrixvectorview);
          return cast;
        }
      }
    }
    else if (cmp0 > 0) {
      cmp1 = strcmp(name, "sidl.BaseClass");
      if (!cmp1) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct sidl_BaseClass__object*)self);
        return cast;
      }
      else if (cmp1 < 0) {
        cmp2 = strcmp(name, "bHYPRE.ProblemDefinition");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_problemdefinition);
          return cast;
        }
      }
      else if (cmp1 > 0) {
        cmp2 = strcmp(name, "sidl.BaseInterface");
        if (!cmp2) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
          return cast;
        }
      }
    }
    if ((*self->d_epv->f_isType)(self,name, _ex)) {
      void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**) = 
        (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
          sidl_BaseInterface__object**)) 
        sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
      cast =  (*func)(((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih, _ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // REMOTE DELETE: call the remote destructor for the object.
  static void remote_bHYPRE_IJParCSRMatrix__delete(
    struct bHYPRE_IJParCSRMatrix__object* self,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
    free((void*) self);
  }

  // REMOTE GETURL: call the getURL function for the object.
  static char* remote_bHYPRE_IJParCSRMatrix__getURL(
    struct bHYPRE_IJParCSRMatrix__object* self, sidl_BaseInterface* _ex)
  {
    struct sidl_rmi_InstanceHandle__object *conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    *_ex = NULL;
    if(conn != NULL) {
      return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
    }
    return NULL;
  }

  // REMOTE ADDREF: For internal babel use only! Remote addRef.
  static void remote_bHYPRE_IJParCSRMatrix__raddRef(
    struct bHYPRE_IJParCSRMatrix__object* self,sidl_BaseInterface* _ex)
  {
    sidl_BaseException netex = NULL;
    // initialize a new invocation
    sidl_BaseInterface _throwaway = NULL;
    struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
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
  remote_bHYPRE_IJParCSRMatrix__isRemote(
      struct bHYPRE_IJParCSRMatrix__object* self, 
      sidl_BaseInterface *_ex) {
    *_ex = NULL;
    return TRUE;
  }

  // REMOTE METHOD STUB:_set_hooks
  static void
  remote_bHYPRE_IJParCSRMatrix__set_hooks(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "_set_hooks", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix._set_hooks.", &throwaway_exception);
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
  static void remote_bHYPRE_IJParCSRMatrix__exec(
    struct bHYPRE_IJParCSRMatrix__object* self,const char* methodName,
    sidl_rmi_Call inArgs,
    sidl_rmi_Return outArgs,
    sidl_BaseInterface* _ex)
  {
    *_ex = NULL;
  }

  // REMOTE METHOD STUB:SetDiagOffdSizes
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in rarray[local_nrows] */ struct sidl_int__array* diag_sizes,
    /* in rarray[local_nrows] */ struct sidl_int__array* offdiag_sizes,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetDiagOffdSizes", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "diag_sizes", diag_sizes,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "offdiag_sizes", offdiag_sizes,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDiagOffdSizes.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_bHYPRE_IJParCSRMatrix_addRef(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct bHYPRE_IJParCSRMatrix__remote* r_obj = (struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data;
      LOCK_STATIC_GLOBALS;
      r_obj->d_refcount++;
      UNLOCK_STATIC_GLOBALS;
    }
  }

  // REMOTE METHOD STUB:deleteRef
  static void
  remote_bHYPRE_IJParCSRMatrix_deleteRef(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      struct bHYPRE_IJParCSRMatrix__remote* r_obj = (struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data;
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
  remote_bHYPRE_IJParCSRMatrix_isSame(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.isSame.", &throwaway_exception);
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
  remote_bHYPRE_IJParCSRMatrix_isType(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "isType", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.isType.", &throwaway_exception);
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
  remote_bHYPRE_IJParCSRMatrix_getClassInfo(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "getClassInfo", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.getClassInfo.", &throwaway_exception);
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

  // REMOTE METHOD STUB:SetLocalRange
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetLocalRange(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ int32_t ilower,
    /* in */ int32_t iupper,
    /* in */ int32_t jlower,
    /* in */ int32_t jupper,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetLocalRange", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "ilower", ilower, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "iupper", iupper, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "jlower", jlower, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Invocation_packInt( _inv, "jupper", jupper, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetLocalRange.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_bHYPRE_IJParCSRMatrix_SetValues(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in rarray[nrows] */ struct sidl_int__array* ncols,
    /* in rarray[nrows] */ struct sidl_int__array* rows,
    /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
    /* in rarray[nnonzeros] */ struct sidl_double__array* values,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetValues.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_bHYPRE_IJParCSRMatrix_AddToValues(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in rarray[nrows] */ struct sidl_int__array* ncols,
    /* in rarray[nrows] */ struct sidl_int__array* rows,
    /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
    /* in rarray[nnonzeros] */ struct sidl_double__array* values,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "AddToValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.AddToValues.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:GetLocalRange
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_GetLocalRange(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* out */ int32_t* ilower,
    /* out */ int32_t* iupper,
    /* out */ int32_t* jlower,
    /* out */ int32_t* jupper,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetLocalRange", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetLocalRange.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackInt( _rsvp, "ilower", ilower, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Response_unpackInt( _rsvp, "iupper", iupper, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Response_unpackInt( _rsvp, "jlower", jlower, _ex);SIDL_CHECK(
        *_ex);
      sidl_rmi_Response_unpackInt( _rsvp, "jupper", jupper, _ex);SIDL_CHECK(
        *_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:GetRowCounts
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_GetRowCounts(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in rarray[nrows] */ struct sidl_int__array* rows,
    /* inout rarray[nrows] */ struct sidl_int__array** ncols,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetRowCounts", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "ncols", *ncols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetRowCounts.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackIntArray( _rsvp, "ncols", ncols,
        sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:GetValues
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_GetValues(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in rarray[nrows] */ struct sidl_int__array* ncols,
    /* in rarray[nrows] */ struct sidl_int__array* rows,
    /* in rarray[nnonzeros] */ struct sidl_int__array* cols,
    /* inout rarray[nnonzeros] */ struct sidl_double__array** values,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetValues", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "values", *values,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetValues.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDoubleArray( _rsvp, "values", values,
        sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetRowSizes
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetRowSizes(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in rarray[nrows] */ struct sidl_int__array* sizes,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetRowSizes", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packIntArray( _inv, "sizes", sizes,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetRowSizes.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_bHYPRE_IJParCSRMatrix_Print(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* filename,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Print", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "filename", filename, 
        _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Print.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:Read
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_Read(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* filename,
    /* in */ struct bHYPRE_MPICommunicator__object* comm,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Read", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "filename", filename, 
        _ex);SIDL_CHECK(*_ex);
      if(comm){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)comm, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "comm", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "comm", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Read.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetCommunicator
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetCommunicator(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetCommunicator", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(mpi_comm){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)mpi_comm, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "mpi_comm", _url, _ex);SIDL_CHECK(
          *_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "mpi_comm", NULL, _ex);SIDL_CHECK(
          *_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetCommunicator.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:Destroy
  static void
  remote_bHYPRE_IJParCSRMatrix_Destroy(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Destroy", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Destroy.", &throwaway_exception);
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

  // REMOTE METHOD STUB:Initialize
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_Initialize(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Initialize", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Initialize.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

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
  remote_bHYPRE_IJParCSRMatrix_Assemble(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Assemble", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Assemble.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetIntParameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetIntParameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
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
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetIntParameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packInt( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetIntParameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetDoubleParameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetDoubleParameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
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
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetDoubleParameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDoubleParameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetStringParameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetStringParameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
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
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetStringParameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetStringParameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetIntArray1Parameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
    /* in rarray[nvalues] */ struct sidl_int__array* value,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetIntArray1Parameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "value", value,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetIntArray1Parameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetIntArray2Parameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
    /* in array<int,2,column-major> */ struct sidl_int__array* value,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetIntArray2Parameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packIntArray( _inv, "value", value,
        sidl_column_major_order,2,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetIntArray2Parameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetDoubleArray1Parameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
    /* in rarray[nvalues] */ struct sidl_double__array* value,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetDoubleArray1Parameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,
        sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDoubleArray1Parameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:SetDoubleArray2Parameter
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
    /* in array<double,2,column-major> */ struct sidl_double__array* value,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "SetDoubleArray2Parameter", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,
        sidl_column_major_order,2,0, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDoubleArray2Parameter.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:GetIntValue
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_GetIntValue(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
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
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetIntValue", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetIntValue.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:GetDoubleValue
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_GetDoubleValue(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ const char* name,
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
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetDoubleValue", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetDoubleValue.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackDouble( _rsvp, "value", value, _ex);SIDL_CHECK(
        *_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:Setup
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_Setup(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Setup", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(b){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)b, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "b", _url, _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "b", NULL, _ex);SIDL_CHECK(*_ex);
      }
      if(x){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)x, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Setup.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:Apply
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_Apply(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char* x_str= NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "Apply", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(b){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)b, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "b", _url, _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "b", NULL, _ex);SIDL_CHECK(*_ex);
      }
      if(*x){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*x, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
      }
      // Transfer this reference
      if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
        SIDL_CHECK(*_ex);
        (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
          sidl_BaseInterface)*x)->d_object, _ex);SIDL_CHECK(*_ex);
        sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x, _ex);SIDL_CHECK(
          *_ex); 
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Apply.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackString( _rsvp, "x", &x_str, _ex);SIDL_CHECK(*_ex);
      *x = bHYPRE_Vector__connectI(x_str, FALSE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:ApplyAdjoint
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_ApplyAdjoint(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x,
    /* out */ struct sidl_BaseInterface__object* *_ex)
  {
    LANG_SPECIFIC_INIT();
    *_ex = NULL;
    {
      // initialize a new invocation
      sidl_BaseInterface _throwaway = NULL;
      sidl_BaseException _be = NULL;
      sidl_rmi_Response _rsvp = NULL;
      char* x_str= NULL;
      int32_t _retval = 0;
      struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "ApplyAdjoint", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      if(b){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)b, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "b", _url, _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "b", NULL, _ex);SIDL_CHECK(*_ex);
      }
      if(*x){
        char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)*x, 
          _ex);SIDL_CHECK(*_ex);
        sidl_rmi_Invocation_packString( _inv, "x", _url, _ex);SIDL_CHECK(*_ex);
        free((void*)_url);
      } else {
        sidl_rmi_Invocation_packString( _inv, "x", NULL, _ex);SIDL_CHECK(*_ex);
      }
      // Transfer this reference
      if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
        SIDL_CHECK(*_ex);
        (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
          sidl_BaseInterface)*x)->d_object, _ex);SIDL_CHECK(*_ex);
        sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x, _ex);SIDL_CHECK(
          *_ex); 
      }

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.ApplyAdjoint.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackString( _rsvp, "x", &x_str, _ex);SIDL_CHECK(*_ex);
      *x = bHYPRE_Vector__connectI(x_str, FALSE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE METHOD STUB:GetRow
  static int32_t
  remote_bHYPRE_IJParCSRMatrix_GetRow(
    /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
    /* in */ int32_t row,
    /* out */ int32_t* size,
    /* out array<int,column-major> */ struct sidl_int__array** col_ind,
    /* out array<double,column-major> */ struct sidl_double__array** values,
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
        bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
      sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( 
        _conn, "GetRow", _ex ); SIDL_CHECK(*_ex);

      // pack in and inout arguments
      sidl_rmi_Invocation_packInt( _inv, "row", row, _ex);SIDL_CHECK(*_ex);

      // send actual RMI request
      _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

      _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
      if(_be != NULL) {
        sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetRow.", &throwaway_exception);
        *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
          &throwaway_exception);
        goto EXIT;
      }

      // extract return value
      sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
        *_ex);

      // unpack out and inout arguments
      sidl_rmi_Response_unpackInt( _rsvp, "size", size, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Response_unpackIntArray( _rsvp, "col_ind", col_ind,
        sidl_column_major_order,1,FALSE, _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Response_unpackDoubleArray( _rsvp, "values", values,
        sidl_column_major_order,1,FALSE, _ex);SIDL_CHECK(*_ex);

      // cleanup and return
      EXIT:
      if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
      if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
      return _retval;
    }
  }

  // REMOTE EPV: create remote entry point vectors (EPVs).
  static void bHYPRE_IJParCSRMatrix__init_remote_epv(void)
  {
    // assert( HAVE_LOCKED_STATIC_GLOBALS );
    struct bHYPRE_IJParCSRMatrix__epv*    epv = 
      &s_rem_epv__bhypre_ijparcsrmatrix;
    struct bHYPRE_CoefficientAccess__epv* e0  = 
      &s_rem_epv__bhypre_coefficientaccess;
    struct bHYPRE_IJMatrixView__epv*      e1  = &s_rem_epv__bhypre_ijmatrixview;
    struct bHYPRE_MatrixVectorView__epv*  e2  = 
      &s_rem_epv__bhypre_matrixvectorview;
    struct bHYPRE_Operator__epv*          e3  = &s_rem_epv__bhypre_operator;
    struct bHYPRE_ProblemDefinition__epv* e4  = 
      &s_rem_epv__bhypre_problemdefinition;
    struct sidl_BaseClass__epv*           e5  = &s_rem_epv__sidl_baseclass;
    struct sidl_BaseInterface__epv*       e6  = &s_rem_epv__sidl_baseinterface;

    epv->f__cast                         = remote_bHYPRE_IJParCSRMatrix__cast;
    epv->f__delete                       = remote_bHYPRE_IJParCSRMatrix__delete;
    epv->f__exec                         = remote_bHYPRE_IJParCSRMatrix__exec;
    epv->f__getURL                       = remote_bHYPRE_IJParCSRMatrix__getURL;
    epv->f__raddRef                      = 
      remote_bHYPRE_IJParCSRMatrix__raddRef;
    epv->f__isRemote                     = 
      remote_bHYPRE_IJParCSRMatrix__isRemote;
    epv->f__set_hooks                    = 
      remote_bHYPRE_IJParCSRMatrix__set_hooks;
    epv->f__ctor                         = NULL;
    epv->f__ctor2                        = NULL;
    epv->f__dtor                         = NULL;
    epv->f_SetDiagOffdSizes              = 
      remote_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes;
    epv->f_addRef                        = remote_bHYPRE_IJParCSRMatrix_addRef;
    epv->f_deleteRef                     = 
      remote_bHYPRE_IJParCSRMatrix_deleteRef;
    epv->f_isSame                        = remote_bHYPRE_IJParCSRMatrix_isSame;
    epv->f_isType                        = remote_bHYPRE_IJParCSRMatrix_isType;
    epv->f_getClassInfo                  = 
      remote_bHYPRE_IJParCSRMatrix_getClassInfo;
    epv->f_SetLocalRange                 = 
      remote_bHYPRE_IJParCSRMatrix_SetLocalRange;
    epv->f_SetValues                     = 
      remote_bHYPRE_IJParCSRMatrix_SetValues;
    epv->f_AddToValues                   = 
      remote_bHYPRE_IJParCSRMatrix_AddToValues;
    epv->f_GetLocalRange                 = 
      remote_bHYPRE_IJParCSRMatrix_GetLocalRange;
    epv->f_GetRowCounts                  = 
      remote_bHYPRE_IJParCSRMatrix_GetRowCounts;
    epv->f_GetValues                     = 
      remote_bHYPRE_IJParCSRMatrix_GetValues;
    epv->f_SetRowSizes                   = 
      remote_bHYPRE_IJParCSRMatrix_SetRowSizes;
    epv->f_Print                         = remote_bHYPRE_IJParCSRMatrix_Print;
    epv->f_Read                          = remote_bHYPRE_IJParCSRMatrix_Read;
    epv->f_SetCommunicator               = 
      remote_bHYPRE_IJParCSRMatrix_SetCommunicator;
    epv->f_Destroy                       = remote_bHYPRE_IJParCSRMatrix_Destroy;
    epv->f_Initialize                    = 
      remote_bHYPRE_IJParCSRMatrix_Initialize;
    epv->f_Assemble                      = 
      remote_bHYPRE_IJParCSRMatrix_Assemble;
    epv->f_SetIntParameter               = 
      remote_bHYPRE_IJParCSRMatrix_SetIntParameter;
    epv->f_SetDoubleParameter            = 
      remote_bHYPRE_IJParCSRMatrix_SetDoubleParameter;
    epv->f_SetStringParameter            = 
      remote_bHYPRE_IJParCSRMatrix_SetStringParameter;
    epv->f_SetIntArray1Parameter         = 
      remote_bHYPRE_IJParCSRMatrix_SetIntArray1Parameter;
    epv->f_SetIntArray2Parameter         = 
      remote_bHYPRE_IJParCSRMatrix_SetIntArray2Parameter;
    epv->f_SetDoubleArray1Parameter      = 
      remote_bHYPRE_IJParCSRMatrix_SetDoubleArray1Parameter;
    epv->f_SetDoubleArray2Parameter      = 
      remote_bHYPRE_IJParCSRMatrix_SetDoubleArray2Parameter;
    epv->f_GetIntValue                   = 
      remote_bHYPRE_IJParCSRMatrix_GetIntValue;
    epv->f_GetDoubleValue                = 
      remote_bHYPRE_IJParCSRMatrix_GetDoubleValue;
    epv->f_Setup                         = remote_bHYPRE_IJParCSRMatrix_Setup;
    epv->f_Apply                         = remote_bHYPRE_IJParCSRMatrix_Apply;
    epv->f_ApplyAdjoint                  = 
      remote_bHYPRE_IJParCSRMatrix_ApplyAdjoint;
    epv->f_GetRow                        = remote_bHYPRE_IJParCSRMatrix_GetRow;

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
    e0->f_GetRow       = (int32_t (*)(void*,int32_t,int32_t*,struct 
      sidl_int__array**,struct sidl_double__array**,struct 
      sidl_BaseInterface__object **)) epv->f_GetRow;
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

    e1->f__cast           = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
      epv->f__cast;
    e1->f__delete         = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__delete;
    e1->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) 
      epv->f__getURL;
    e1->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__raddRef;
    e1->f__isRemote       = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e1->f__set_hooks      = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
      epv->f__set_hooks;
    e1->f__exec           = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e1->f_SetLocalRange   = (int32_t (*)(void*,int32_t,int32_t,int32_t,int32_t,
      struct sidl_BaseInterface__object **)) epv->f_SetLocalRange;
    e1->f_SetValues       = (int32_t (*)(void*,struct sidl_int__array*,struct 
      sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
      struct sidl_BaseInterface__object **)) epv->f_SetValues;
    e1->f_AddToValues     = (int32_t (*)(void*,struct sidl_int__array*,struct 
      sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,
      struct sidl_BaseInterface__object **)) epv->f_AddToValues;
    e1->f_GetLocalRange   = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
      int32_t*,struct sidl_BaseInterface__object **)) epv->f_GetLocalRange;
    e1->f_GetRowCounts    = (int32_t (*)(void*,struct sidl_int__array*,struct 
      sidl_int__array**,struct sidl_BaseInterface__object **)) 
      epv->f_GetRowCounts;
    e1->f_GetValues       = (int32_t (*)(void*,struct sidl_int__array*,struct 
      sidl_int__array*,struct sidl_int__array*,struct sidl_double__array**,
      struct sidl_BaseInterface__object **)) epv->f_GetValues;
    e1->f_SetRowSizes     = (int32_t (*)(void*,struct sidl_int__array*,struct 
      sidl_BaseInterface__object **)) epv->f_SetRowSizes;
    e1->f_Print           = (int32_t (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_Print;
    e1->f_Read            = (int32_t (*)(void*,const char*,struct 
      bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
      epv->f_Read;
    e1->f_SetCommunicator = (int32_t (*)(void*,struct 
      bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
      epv->f_SetCommunicator;
    e1->f_Destroy         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_Destroy;
    e1->f_Initialize      = (int32_t (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Initialize;
    e1->f_Assemble        = (int32_t (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Assemble;
    e1->f_addRef          = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e1->f_deleteRef       = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e1->f_isSame          = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e1->f_isType          = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e1->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e2->f__cast           = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
      epv->f__cast;
    e2->f__delete         = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__delete;
    e2->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) 
      epv->f__getURL;
    e2->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__raddRef;
    e2->f__isRemote       = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e2->f__set_hooks      = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
      epv->f__set_hooks;
    e2->f__exec           = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e2->f_SetCommunicator = (int32_t (*)(void*,struct 
      bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
      epv->f_SetCommunicator;
    e2->f_Destroy         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_Destroy;
    e2->f_Initialize      = (int32_t (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Initialize;
    e2->f_Assemble        = (int32_t (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Assemble;
    e2->f_addRef          = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e2->f_deleteRef       = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e2->f_isSame          = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e2->f_isType          = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e2->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e3->f__cast                    = (void* (*)(void*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e3->f__delete                  = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__delete;
    e3->f__getURL                  = (char* (*)(void*,sidl_BaseInterface*)) 
      epv->f__getURL;
    e3->f__raddRef                 = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__raddRef;
    e3->f__isRemote                = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e3->f__set_hooks               = (void (*)(void*,int32_t, 
      sidl_BaseInterface*)) epv->f__set_hooks;
    e3->f__exec                    = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e3->f_SetCommunicator          = (int32_t (*)(void*,struct 
      bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
      epv->f_SetCommunicator;
    e3->f_Destroy                  = (void (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Destroy;
    e3->f_SetIntParameter          = (int32_t (*)(void*,const char*,int32_t,
      struct sidl_BaseInterface__object **)) epv->f_SetIntParameter;
    e3->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,double,
      struct sidl_BaseInterface__object **)) epv->f_SetDoubleParameter;
    e3->f_SetStringParameter       = (int32_t (*)(void*,const char*,const char*,
      struct sidl_BaseInterface__object **)) epv->f_SetStringParameter;
    e3->f_SetIntArray1Parameter    = (int32_t (*)(void*,const char*,struct 
      sidl_int__array*,struct sidl_BaseInterface__object **)) 
      epv->f_SetIntArray1Parameter;
    e3->f_SetIntArray2Parameter    = (int32_t (*)(void*,const char*,struct 
      sidl_int__array*,struct sidl_BaseInterface__object **)) 
      epv->f_SetIntArray2Parameter;
    e3->f_SetDoubleArray1Parameter = (int32_t (*)(void*,const char*,struct 
      sidl_double__array*,struct sidl_BaseInterface__object **)) 
      epv->f_SetDoubleArray1Parameter;
    e3->f_SetDoubleArray2Parameter = (int32_t (*)(void*,const char*,struct 
      sidl_double__array*,struct sidl_BaseInterface__object **)) 
      epv->f_SetDoubleArray2Parameter;
    e3->f_GetIntValue              = (int32_t (*)(void*,const char*,int32_t*,
      struct sidl_BaseInterface__object **)) epv->f_GetIntValue;
    e3->f_GetDoubleValue           = (int32_t (*)(void*,const char*,double*,
      struct sidl_BaseInterface__object **)) epv->f_GetDoubleValue;
    e3->f_Setup                    = (int32_t (*)(void*,struct 
      bHYPRE_Vector__object*,struct bHYPRE_Vector__object*,struct 
      sidl_BaseInterface__object **)) epv->f_Setup;
    e3->f_Apply                    = (int32_t (*)(void*,struct 
      bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,struct 
      sidl_BaseInterface__object **)) epv->f_Apply;
    e3->f_ApplyAdjoint             = (int32_t (*)(void*,struct 
      bHYPRE_Vector__object*,struct bHYPRE_Vector__object**,struct 
      sidl_BaseInterface__object **)) epv->f_ApplyAdjoint;
    e3->f_addRef                   = (void (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_addRef;
    e3->f_deleteRef                = (void (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_deleteRef;
    e3->f_isSame                   = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e3->f_isType                   = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e3->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(void*,
      struct sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e4->f__cast           = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
      epv->f__cast;
    e4->f__delete         = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__delete;
    e4->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) 
      epv->f__getURL;
    e4->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) 
      epv->f__raddRef;
    e4->f__isRemote       = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e4->f__set_hooks      = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
      epv->f__set_hooks;
    e4->f__exec           = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e4->f_SetCommunicator = (int32_t (*)(void*,struct 
      bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
      epv->f_SetCommunicator;
    e4->f_Destroy         = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_Destroy;
    e4->f_Initialize      = (int32_t (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Initialize;
    e4->f_Assemble        = (int32_t (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_Assemble;
    e4->f_addRef          = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_addRef;
    e4->f_deleteRef       = (void (*)(void*,struct sidl_BaseInterface__object 
      **)) epv->f_deleteRef;
    e4->f_isSame          = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e4->f_isType          = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e4->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    e5->f__cast        = (void* (*)(struct sidl_BaseClass__object*,const char*,
      sidl_BaseInterface*)) epv->f__cast;
    e5->f__delete      = (void (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__delete;
    e5->f__getURL      = (char* (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__getURL;
    e5->f__raddRef     = (void (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__raddRef;
    e5->f__isRemote    = (sidl_bool (*)(struct sidl_BaseClass__object*,
      sidl_BaseInterface*)) epv->f__isRemote;
    e5->f__set_hooks   = (void (*)(struct sidl_BaseClass__object*,int32_t, 
      sidl_BaseInterface*)) epv->f__set_hooks;
    e5->f__exec        = (void (*)(struct sidl_BaseClass__object*,const char*,
      struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e5->f_addRef       = (void (*)(struct sidl_BaseClass__object*,struct 
      sidl_BaseInterface__object **)) epv->f_addRef;
    e5->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,struct 
      sidl_BaseInterface__object **)) epv->f_deleteRef;
    e5->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e5->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,const 
      char*,struct sidl_BaseInterface__object **)) epv->f_isType;
    e5->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
      sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) 
      epv->f_getClassInfo;

    e6->f__cast        = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
      epv->f__cast;
    e6->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
    e6->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
    e6->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
    e6->f__isRemote    = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
      epv->f__isRemote;
    e6->f__set_hooks   = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
      epv->f__set_hooks;
    e6->f__exec        = (void (*)(void*,const char*,struct 
      sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
      sidl_BaseInterface__object **)) epv->f__exec;
    e6->f_addRef       = (void (*)(void*,struct sidl_BaseInterface__object **)) 
      epv->f_addRef;
    e6->f_deleteRef    = (void (*)(void*,struct sidl_BaseInterface__object **)) 
      epv->f_deleteRef;
    e6->f_isSame       = (sidl_bool (*)(void*,struct 
      sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
      epv->f_isSame;
    e6->f_isType       = (sidl_bool (*)(void*,const char*,struct 
      sidl_BaseInterface__object **)) epv->f_isType;
    e6->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,struct 
      sidl_BaseInterface__object **)) epv->f_getClassInfo;

    s_remote_initialized = 1;
  }

  // Create an instance that connects to an existing remote object.
  static struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__remoteConnect(const char *url, sidl_bool ar, 
    sidl_BaseInterface *_ex)
  {
    struct bHYPRE_IJParCSRMatrix__object* self;

    struct bHYPRE_IJParCSRMatrix__object* s0;
    struct sidl_BaseClass__object* s1;

    struct bHYPRE_IJParCSRMatrix__remote* r_obj;
    sidl_rmi_InstanceHandle instance = NULL;
    char* objectID = NULL;
    objectID = NULL;
    *_ex = NULL;
    if(url == NULL) {return NULL;}
    objectID = sidl_rmi_ServerRegistry_isLocalObject(url, _ex);
    if(objectID) {
      sidl_BaseInterface bi = (
        sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(
        objectID, _ex); SIDL_CHECK(*_ex);
      return bHYPRE_IJParCSRMatrix__rmicast(bi,_ex);SIDL_CHECK(*_ex);
    }
    instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex ); 
      SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct bHYPRE_IJParCSRMatrix__object*) malloc(
        sizeof(struct bHYPRE_IJParCSRMatrix__object));

    r_obj =
      (struct bHYPRE_IJParCSRMatrix__remote*) malloc(
        sizeof(struct bHYPRE_IJParCSRMatrix__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                 self;
    s1 =                                 &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE_IJParCSRMatrix__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_bhypre_coefficientaccess.d_epv    = 
      &s_rem_epv__bhypre_coefficientaccess;
    s0->d_bhypre_coefficientaccess.d_object = (void*) self;

    s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
    s0->d_bhypre_ijmatrixview.d_object = (void*) self;

    s0->d_bhypre_matrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_matrixvectorview;
    s0->d_bhypre_matrixvectorview.d_object = (void*) self;

    s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
    s0->d_bhypre_operator.d_object = (void*) self;

    s0->d_bhypre_problemdefinition.d_epv    = 
      &s_rem_epv__bhypre_problemdefinition;
    s0->d_bhypre_problemdefinition.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre_ijparcsrmatrix;

    self->d_data = (void*) r_obj;

    return self;
    EXIT:
    return NULL;
  }
  // Create an instance that uses an already existing 
  // InstanceHandle to connect to an existing remote object.
  static struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__IHConnect(sidl_rmi_InstanceHandle instance, 
    sidl_BaseInterface *_ex)
  {
    struct bHYPRE_IJParCSRMatrix__object* self;

    struct bHYPRE_IJParCSRMatrix__object* s0;
    struct sidl_BaseClass__object* s1;

    struct bHYPRE_IJParCSRMatrix__remote* r_obj;
    self =
      (struct bHYPRE_IJParCSRMatrix__object*) malloc(
        sizeof(struct bHYPRE_IJParCSRMatrix__object));

    r_obj =
      (struct bHYPRE_IJParCSRMatrix__remote*) malloc(
        sizeof(struct bHYPRE_IJParCSRMatrix__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                 self;
    s1 =                                 &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE_IJParCSRMatrix__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_bhypre_coefficientaccess.d_epv    = 
      &s_rem_epv__bhypre_coefficientaccess;
    s0->d_bhypre_coefficientaccess.d_object = (void*) self;

    s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
    s0->d_bhypre_ijmatrixview.d_object = (void*) self;

    s0->d_bhypre_matrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_matrixvectorview;
    s0->d_bhypre_matrixvectorview.d_object = (void*) self;

    s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
    s0->d_bhypre_operator.d_object = (void*) self;

    s0->d_bhypre_problemdefinition.d_epv    = 
      &s_rem_epv__bhypre_problemdefinition;
    s0->d_bhypre_problemdefinition.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre_ijparcsrmatrix;

    self->d_data = (void*) r_obj;

    sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
    return self;
    EXIT:
    return NULL;
  }
  // REMOTE: generate remote instance given URL string.
  static struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__remoteCreate(const char *url, sidl_BaseInterface *_ex)
  {
    sidl_BaseInterface _throwaway_exception = NULL;
    struct bHYPRE_IJParCSRMatrix__object* self;

    struct bHYPRE_IJParCSRMatrix__object* s0;
    struct sidl_BaseClass__object* s1;

    struct bHYPRE_IJParCSRMatrix__remote* r_obj;
    sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
      url, "bHYPRE.IJParCSRMatrix", _ex ); SIDL_CHECK(*_ex);
    if ( instance == NULL) { return NULL; }
    self =
      (struct bHYPRE_IJParCSRMatrix__object*) malloc(
        sizeof(struct bHYPRE_IJParCSRMatrix__object));

    r_obj =
      (struct bHYPRE_IJParCSRMatrix__remote*) malloc(
        sizeof(struct bHYPRE_IJParCSRMatrix__remote));

    r_obj->d_refcount = 1;
    r_obj->d_ih = instance;
    s0 =                                 self;
    s1 =                                 &s0->d_sidl_baseclass;

    LOCK_STATIC_GLOBALS;
    if (!s_remote_initialized) {
      bHYPRE_IJParCSRMatrix__init_remote_epv();
    }
    UNLOCK_STATIC_GLOBALS;

    s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
    s1->d_sidl_baseinterface.d_object = (void*) self;

    s1->d_data = (void*) r_obj;
    s1->d_epv  = &s_rem_epv__sidl_baseclass;

    s0->d_bhypre_coefficientaccess.d_epv    = 
      &s_rem_epv__bhypre_coefficientaccess;
    s0->d_bhypre_coefficientaccess.d_object = (void*) self;

    s0->d_bhypre_ijmatrixview.d_epv    = &s_rem_epv__bhypre_ijmatrixview;
    s0->d_bhypre_ijmatrixview.d_object = (void*) self;

    s0->d_bhypre_matrixvectorview.d_epv    = 
      &s_rem_epv__bhypre_matrixvectorview;
    s0->d_bhypre_matrixvectorview.d_object = (void*) self;

    s0->d_bhypre_operator.d_epv    = &s_rem_epv__bhypre_operator;
    s0->d_bhypre_operator.d_object = (void*) self;

    s0->d_bhypre_problemdefinition.d_epv    = 
      &s_rem_epv__bhypre_problemdefinition;
    s0->d_bhypre_problemdefinition.d_object = (void*) self;

    s0->d_data = (void*) r_obj;
    s0->d_epv  = &s_rem_epv__bhypre_ijparcsrmatrix;

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
  struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__rmicast(
    void* obj,
    sidl_BaseInterface* _ex)
  {
    struct bHYPRE_IJParCSRMatrix__object* cast = NULL;

    *_ex = NULL;
    if(!connect_loaded) {
      sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.IJParCSRMatrix", (
        void*)bHYPRE_IJParCSRMatrix__IHConnect, _ex);
      connect_loaded = 1;
    }
    if (obj != NULL) {
      struct sidl_BaseInterface__object* base = (struct 
        sidl_BaseInterface__object*) obj;
      cast = (struct bHYPRE_IJParCSRMatrix__object*) (*base->d_epv->f__cast)(
        base->d_object,
        "bHYPRE.IJParCSRMatrix", _ex); SIDL_CHECK(*_ex);
    }

    return cast;
    EXIT:
    return NULL;
  }

  // 
  // RMI connector function for the class.
  // 
  struct bHYPRE_IJParCSRMatrix__object*
  bHYPRE_IJParCSRMatrix__connectI(const char* url, sidl_bool ar, struct 
    sidl_BaseInterface__object **_ex)
  {
    return bHYPRE_IJParCSRMatrix__remoteConnect(url, ar, _ex);
  }

}

//////////////////////////////////////////////////
// 
// Special methods for throwing exceptions
// 

void
bHYPRE::IJParCSRMatrix::throwException0(
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
 *  This function is the preferred way to create an IJParCSR Matrix. 
 */
::bHYPRE::IJParCSRMatrix
bHYPRE::IJParCSRMatrix::Create( /* in */::bHYPRE::MPICommunicator mpi_comm, /* 
  in */int32_t ilower, /* in */int32_t iupper, /* in */int32_t jlower, /* in 
  */int32_t jupper )

{
  ::bHYPRE::IJParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) 
    mpi_comm.::bHYPRE::MPICommunicator::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = ::bHYPRE::IJParCSRMatrix( ( _get_sepv()->f_Create)( /* in */ 
    _local_mpi_comm, /* in */ ilower, /* in */ iupper, /* in */ jlower, /* in 
    */ jupper, &_exception ), false);
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined static method
 */
::bHYPRE::IJParCSRMatrix
bHYPRE::IJParCSRMatrix::GenerateLaplacian( /* in */::bHYPRE::MPICommunicator 
  mpi_comm, /* in */int32_t nx, /* in */int32_t ny, /* in */int32_t nz, /* in 
  */int32_t Px, /* in */int32_t Py, /* in */int32_t Pz, /* in */int32_t p, /* 
  in */int32_t q, /* in */int32_t r, /* in rarray[nvalues] */double* values, /* 
  in */int32_t nvalues, /* in */int32_t discretization )

{
  ::bHYPRE::IJParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) 
    mpi_comm.::bHYPRE::MPICommunicator::_get_ior();
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nvalues-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper, 
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = ::bHYPRE::IJParCSRMatrix( ( _get_sepv()->f_GenerateLaplacian)( /* 
    in */ _local_mpi_comm, /* in */ nx, /* in */ ny, /* in */ nz, /* in */ Px, 
    /* in */ Py, /* in */ Pz, /* in */ p, /* in */ q, /* in */ r, /* in 
    rarray[nvalues] */ values_tmp, /* in */ discretization, &_exception ), 
    false);
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)values_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)values_tmp);
  /*unpack results and cleanup*/
  return _result;
}


/**
 * user defined static method
 */
::bHYPRE::IJParCSRMatrix
bHYPRE::IJParCSRMatrix::GenerateLaplacian( /* in */::bHYPRE::MPICommunicator 
  mpi_comm, /* in */int32_t nx, /* in */int32_t ny, /* in */int32_t nz, /* in 
  */int32_t Px, /* in */int32_t Py, /* in */int32_t Pz, /* in */int32_t p, /* 
  in */int32_t q, /* in */int32_t r, /* in rarray[nvalues] 
  */::sidl::array<double> values, /* in */int32_t discretization )

{
  ::bHYPRE::IJParCSRMatrix _result;
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) 
    mpi_comm.::bHYPRE::MPICommunicator::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = ::bHYPRE::IJParCSRMatrix( ( _get_sepv()->f_GenerateLaplacian)( /* 
    in */ _local_mpi_comm, /* in */ nx, /* in */ ny, /* in */ nz, /* in */ Px, 
    /* in */ Py, /* in */ Pz, /* in */ p, /* in */ q, /* in */ r, /* in 
    rarray[nvalues] */ values._get_ior(), /* in */ discretization, &_exception 
    ), false);
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetDiagOffdSizes( /* in rarray[local_nrows] */int32_t* 
  diag_sizes, /* in rarray[local_nrows] */int32_t* offdiag_sizes, /* in 
  */int32_t local_nrows )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t diag_sizes_lower[1], diag_sizes_upper[1], diag_sizes_stride[1];
  struct sidl_int__array diag_sizes_real;
  struct sidl_int__array *diag_sizes_tmp = &diag_sizes_real;
  diag_sizes_upper[0] = local_nrows-1;
  sidl_int__array_init(diag_sizes, diag_sizes_tmp, 1, diag_sizes_lower, 
    diag_sizes_upper, diag_sizes_stride);
  int32_t offdiag_sizes_lower[1], offdiag_sizes_upper[1], 
    offdiag_sizes_stride[1];
  struct sidl_int__array offdiag_sizes_real;
  struct sidl_int__array *offdiag_sizes_tmp = &offdiag_sizes_real;
  offdiag_sizes_upper[0] = local_nrows-1;
  sidl_int__array_init(offdiag_sizes, offdiag_sizes_tmp, 1, offdiag_sizes_lower,
    offdiag_sizes_upper, offdiag_sizes_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDiagOffdSizes))(loc_self, /* in 
    rarray[local_nrows] */ diag_sizes_tmp, /* in rarray[local_nrows] */ 
    offdiag_sizes_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)diag_sizes_tmp);
    sidl__array_deleteRef((struct sidl__array *)offdiag_sizes_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)diag_sizes_tmp);
  sidl__array_deleteRef((struct sidl__array *)offdiag_sizes_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row of the diagonal and off-diagonal blocks.  The diagonal
 * block is the submatrix whose column numbers correspond to
 * rows owned by this process, and the off-diagonal block is
 * everything else.  The arrays {\tt diag\_sizes} and {\tt
 * offdiag\_sizes} contain estimated sizes for each row of the
 * diagonal and off-diagonal blocks, respectively.  This routine
 * can significantly improve the efficiency of matrix
 * construction, and should always be utilized if possible.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetDiagOffdSizes( /* in rarray[local_nrows] 
  */::sidl::array<int32_t> diag_sizes, /* in rarray[local_nrows] 
  */::sidl::array<int32_t> offdiag_sizes )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDiagOffdSizes))(loc_self, /* in 
    rarray[local_nrows] */ diag_sizes._get_ior(), /* in rarray[local_nrows] */ 
    offdiag_sizes._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the local range for a matrix object.  Each process owns
 * some unique consecutive range of rows, indicated by the
 * global row indices {\tt ilower} and {\tt iupper}.  The row
 * data is required to be such that the value of {\tt ilower} on
 * any process $p$ be exactly one more than the value of {\tt
 * iupper} on process $p-1$.  Note that the first row of the
 * global matrix may start with any integer value.  In
 * particular, one may use zero- or one-based indexing.
 * 
 * For square matrices, {\tt jlower} and {\tt jupper} typically
 * should match {\tt ilower} and {\tt iupper}, respectively.
 * For rectangular matrices, {\tt jlower} and {\tt jupper}
 * should define a partitioning of the columns.  This
 * partitioning must be used for any vector $v$ that will be
 * used in matrix-vector products with the rectangular matrix.
 * The matrix data structure may use {\tt jlower} and {\tt
 * jupper} to store the diagonal blocks (rectangular in general)
 * of the matrix separately from the rest of the matrix.
 * 
 * Collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetLocalRange( /* in */int32_t ilower, /* in */int32_t 
  iupper, /* in */int32_t jlower, /* in */int32_t jupper )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetLocalRange))(loc_self, /* in */ ilower, /* 
    in */ iupper, /* in */ jlower, /* in */ jupper, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetValues( /* in */int32_t nrows, /* in rarray[nrows] 
  */int32_t* ncols, /* in rarray[nrows] */int32_t* rows, /* in 
  rarray[nnonzeros] */int32_t* cols, /* in rarray[nnonzeros] */double* values, 
  /* in */int32_t nnonzeros )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper, 
    ncols_stride);
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_int__array cols_real;
  struct sidl_int__array *cols_tmp = &cols_real;
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper, 
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self, /* in rarray[nrows] */ 
    ncols_tmp, /* in rarray[nrows] */ rows_tmp, /* in rarray[nnonzeros] */ 
    cols_tmp, /* in rarray[nnonzeros] */ values_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
    sidl__array_deleteRef((struct sidl__array *)rows_tmp);
    sidl__array_deleteRef((struct sidl__array *)cols_tmp);
    sidl__array_deleteRef((struct sidl__array *)values_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
  sidl__array_deleteRef((struct sidl__array *)rows_tmp);
  sidl__array_deleteRef((struct sidl__array *)cols_tmp);
  sidl__array_deleteRef((struct sidl__array *)values_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Sets values for {\tt nrows} of the matrix.  The arrays {\tt
 * ncols} and {\tt rows} are of dimension {\tt nrows} and
 * contain the number of columns in each row and the row
 * indices, respectively.  The array {\tt cols} contains the
 * column indices for each of the {\tt rows}, and is ordered by
 * rows.  The data in the {\tt values} array corresponds
 * directly to the column entries in {\tt cols}.  The last argument
 * is the size of the cols and values arrays, i.e. the total number
 * of nonzeros being provided, i.e. the sum of all values in ncols.
 * This functin erases any previous values at the specified locations and
 * replaces them with new ones, or, if there was no value there before,
 * inserts a new one.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetValues( /* in rarray[nrows] */::sidl::array<int32_t> 
  ncols, /* in rarray[nrows] */::sidl::array<int32_t> rows, /* in 
  rarray[nnonzeros] */::sidl::array<int32_t> cols, /* in rarray[nnonzeros] 
  */::sidl::array<double> values )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetValues))(loc_self, /* in rarray[nrows] */ 
    ncols._get_ior(), /* in rarray[nrows] */ rows._get_ior(), /* in 
    rarray[nnonzeros] */ cols._get_ior(), /* in rarray[nnonzeros] */ 
    values._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::AddToValues( /* in */int32_t nrows, /* in rarray[nrows] 
  */int32_t* ncols, /* in rarray[nrows] */int32_t* rows, /* in 
  rarray[nnonzeros] */int32_t* cols, /* in rarray[nnonzeros] */double* values, 
  /* in */int32_t nnonzeros )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper, 
    ncols_stride);
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_int__array cols_real;
  struct sidl_int__array *cols_tmp = &cols_real;
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper, 
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self, /* in rarray[nrows] 
    */ ncols_tmp, /* in rarray[nrows] */ rows_tmp, /* in rarray[nnonzeros] */ 
    cols_tmp, /* in rarray[nnonzeros] */ values_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
    sidl__array_deleteRef((struct sidl__array *)rows_tmp);
    sidl__array_deleteRef((struct sidl__array *)cols_tmp);
    sidl__array_deleteRef((struct sidl__array *)values_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
  sidl__array_deleteRef((struct sidl__array *)rows_tmp);
  sidl__array_deleteRef((struct sidl__array *)cols_tmp);
  sidl__array_deleteRef((struct sidl__array *)values_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::AddToValues( /* in rarray[nrows] 
  */::sidl::array<int32_t> ncols, /* in rarray[nrows] */::sidl::array<int32_t> 
  rows, /* in rarray[nnonzeros] */::sidl::array<int32_t> cols, /* in 
  rarray[nnonzeros] */::sidl::array<double> values )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_AddToValues))(loc_self, /* in rarray[nrows] 
    */ ncols._get_ior(), /* in rarray[nrows] */ rows._get_ior(), /* in 
    rarray[nnonzeros] */ cols._get_ior(), /* in rarray[nnonzeros] */ 
    values._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetLocalRange( /* out */int32_t& ilower, /* out 
  */int32_t& iupper, /* out */int32_t& jlower, /* out */int32_t& jupper )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetLocalRange))(loc_self, /* out */ &ilower, 
    /* out */ &iupper, /* out */ &jlower, /* out */ &jupper, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetRowCounts( /* in */int32_t nrows, /* in 
  rarray[nrows] */int32_t* rows, /* inout rarray[nrows] */int32_t* ncols )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper, 
    ncols_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRowCounts))(loc_self, /* in rarray[nrows] 
    */ rows_tmp, /* inout rarray[nrows] */ &ncols_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)rows_tmp);
    sidl__array_deleteRef((struct sidl__array *)ncols_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)rows_tmp);
  sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetRowCounts( /* in rarray[nrows] 
  */::sidl::array<int32_t> rows, /* inout rarray[nrows] 
  */::sidl::array<int32_t>& ncols )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  if (ncols) {
    ncols.addRef();
  }
  struct sidl_int__array* _local_ncols = ncols._get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRowCounts))(loc_self, /* in rarray[nrows] 
    */ rows._get_ior(), /* inout rarray[nrows] */ &_local_ncols, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  ncols._set_ior(_local_ncols);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetValues( /* in */int32_t nrows, /* in rarray[nrows] 
  */int32_t* ncols, /* in rarray[nrows] */int32_t* rows, /* in 
  rarray[nnonzeros] */int32_t* cols, /* inout rarray[nnonzeros] */double* 
  values, /* in */int32_t nnonzeros )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t ncols_lower[1], ncols_upper[1], ncols_stride[1];
  struct sidl_int__array ncols_real;
  struct sidl_int__array *ncols_tmp = &ncols_real;
  ncols_upper[0] = nrows-1;
  sidl_int__array_init(ncols, ncols_tmp, 1, ncols_lower, ncols_upper, 
    ncols_stride);
  int32_t rows_lower[1], rows_upper[1], rows_stride[1];
  struct sidl_int__array rows_real;
  struct sidl_int__array *rows_tmp = &rows_real;
  rows_upper[0] = nrows-1;
  sidl_int__array_init(rows, rows_tmp, 1, rows_lower, rows_upper, rows_stride);
  int32_t cols_lower[1], cols_upper[1], cols_stride[1];
  struct sidl_int__array cols_real;
  struct sidl_int__array *cols_tmp = &cols_real;
  cols_upper[0] = nnonzeros-1;
  sidl_int__array_init(cols, cols_tmp, 1, cols_lower, cols_upper, cols_stride);
  int32_t values_lower[1], values_upper[1], values_stride[1];
  struct sidl_double__array values_real;
  struct sidl_double__array *values_tmp = &values_real;
  values_upper[0] = nnonzeros-1;
  sidl_double__array_init(values, values_tmp, 1, values_lower, values_upper, 
    values_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self, /* in rarray[nrows] */ 
    ncols_tmp, /* in rarray[nrows] */ rows_tmp, /* in rarray[nnonzeros] */ 
    cols_tmp, /* inout rarray[nnonzeros] */ &values_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
    sidl__array_deleteRef((struct sidl__array *)rows_tmp);
    sidl__array_deleteRef((struct sidl__array *)cols_tmp);
    sidl__array_deleteRef((struct sidl__array *)values_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)ncols_tmp);
  sidl__array_deleteRef((struct sidl__array *)rows_tmp);
  sidl__array_deleteRef((struct sidl__array *)cols_tmp);
  sidl__array_deleteRef((struct sidl__array *)values_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetValues( /* in rarray[nrows] */::sidl::array<int32_t> 
  ncols, /* in rarray[nrows] */::sidl::array<int32_t> rows, /* in 
  rarray[nnonzeros] */::sidl::array<int32_t> cols, /* inout rarray[nnonzeros] 
  */::sidl::array<double>& values )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  if (values) {
    values.addRef();
  }
  struct sidl_double__array* _local_values = values._get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetValues))(loc_self, /* in rarray[nrows] */ 
    ncols._get_ior(), /* in rarray[nrows] */ rows._get_ior(), /* in 
    rarray[nnonzeros] */ cols._get_ior(), /* inout rarray[nnonzeros] */ 
    &_local_values, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  values._set_ior(_local_values);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetRowSizes( /* in rarray[nrows] */int32_t* sizes, /* 
  in */int32_t nrows )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t sizes_lower[1], sizes_upper[1], sizes_stride[1];
  struct sidl_int__array sizes_real;
  struct sidl_int__array *sizes_tmp = &sizes_real;
  sizes_upper[0] = nrows-1;
  sidl_int__array_init(sizes, sizes_tmp, 1, sizes_lower, sizes_upper, 
    sizes_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetRowSizes))(loc_self, /* in rarray[nrows] 
    */ sizes_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)sizes_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)sizes_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetRowSizes( /* in rarray[nrows] 
  */::sidl::array<int32_t> sizes )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetRowSizes))(loc_self, /* in rarray[nrows] 
    */ sizes._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 */
int32_t
bHYPRE::IJParCSRMatrix::Print( /* in */const ::std::string& filename )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Print))(loc_self, /* in */ filename.c_str(), 
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 */
int32_t
bHYPRE::IJParCSRMatrix::Read( /* in */const ::std::string& filename, /* in 
  */::bHYPRE::MPICommunicator comm )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  struct bHYPRE_MPICommunicator__object* _local_comm = (struct 
    bHYPRE_MPICommunicator__object*) comm.::bHYPRE::MPICommunicator::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Read))(loc_self, /* in */ filename.c_str(), 
    /* in */ _local_comm, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */
int32_t
bHYPRE::IJParCSRMatrix::SetCommunicator( /* in */::bHYPRE::MPICommunicator 
  mpi_comm )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  struct bHYPRE_MPICommunicator__object* _local_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) 
    mpi_comm.::bHYPRE::MPICommunicator::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetCommunicator))(loc_self, /* in */ 
    _local_mpi_comm, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */
void
bHYPRE::IJParCSRMatrix::Destroy(  )

{

  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f_Destroy))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}



/**
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */
int32_t
bHYPRE::IJParCSRMatrix::Initialize(  )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Initialize))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */
int32_t
bHYPRE::IJParCSRMatrix::Assemble(  )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Assemble))(loc_self, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetIntParameter( /* in */const ::std::string& name, /* 
  in */int32_t value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntParameter))(loc_self, /* in */ 
    name.c_str(), /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetDoubleParameter( /* in */const ::std::string& name, 
  /* in */double value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleParameter))(loc_self, /* in */ 
    name.c_str(), /* in */ value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the string parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetStringParameter( /* in */const ::std::string& name, 
  /* in */const ::std::string& value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetStringParameter))(loc_self, /* in */ 
    name.c_str(), /* in */ value.c_str(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int 1-D array parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetIntArray1Parameter( /* in */const ::std::string& 
  name, /* in rarray[nvalues] */int32_t* value, /* in */int32_t nvalues )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t value_lower[1], value_upper[1], value_stride[1];
  struct sidl_int__array value_real;
  struct sidl_int__array *value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_int__array_init(value, value_tmp, 1, value_lower, value_upper, 
    value_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntArray1Parameter))(loc_self, /* in */ 
    name.c_str(), /* in rarray[nvalues] */ value_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)value_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)value_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int 1-D array parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetIntArray1Parameter( /* in */const ::std::string& 
  name, /* in rarray[nvalues] */::sidl::array<int32_t> value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntArray1Parameter))(loc_self, /* in */ 
    name.c_str(), /* in rarray[nvalues] */ value._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int 2-D array parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetIntArray2Parameter( /* in */const ::std::string& 
  name, /* in array<int,2,column-major> */::sidl::array<int32_t> value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetIntArray2Parameter))(loc_self, /* in */ 
    name.c_str(), /* in array<int,2,column-major> */ value._get_ior(), 
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double 1-D array parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetDoubleArray1Parameter( /* in */const ::std::string& 
  name, /* in rarray[nvalues] */double* value, /* in */int32_t nvalues )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  int32_t value_lower[1], value_upper[1], value_stride[1];
  struct sidl_double__array value_real;
  struct sidl_double__array *value_tmp = &value_real;
  value_upper[0] = nvalues-1;
  sidl_double__array_init(value, value_tmp, 1, value_lower, value_upper, 
    value_stride);
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleArray1Parameter))(loc_self, /* in */ 
    name.c_str(), /* in rarray[nvalues] */ value_tmp, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {
    sidl__array_deleteRef((struct sidl__array *)value_tmp);

    throwException0(_exception);
  }
  sidl__array_deleteRef((struct sidl__array *)value_tmp);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double 1-D array parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetDoubleArray1Parameter( /* in */const ::std::string& 
  name, /* in rarray[nvalues] */::sidl::array<double> value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleArray1Parameter))(loc_self, /* in */ 
    name.c_str(), /* in rarray[nvalues] */ value._get_ior(), &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the double 2-D array parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::SetDoubleArray2Parameter( /* in */const ::std::string& 
  name, /* in array<double,2,column-major> */::sidl::array<double> value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_SetDoubleArray2Parameter))(loc_self, /* in */ 
    name.c_str(), /* in array<double,2,column-major> */ value._get_ior(), 
    &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Set the int parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetIntValue( /* in */const ::std::string& name, /* out 
  */int32_t& value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetIntValue))(loc_self, /* in */ name.c_str(),
    /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Get the double parameter associated with {\tt name}.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetDoubleValue( /* in */const ::std::string& name, /* 
  out */double& value )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetDoubleValue))(loc_self, /* in */ 
    name.c_str(), /* out */ &value, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */
int32_t
bHYPRE::IJParCSRMatrix::Setup( /* in */::bHYPRE::Vector b, /* in 
  */::bHYPRE::Vector x )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  struct bHYPRE_Vector__object* _local_b = (struct bHYPRE_Vector__object*) 
    b.::bHYPRE::Vector::_get_ior();
  struct bHYPRE_Vector__object* _local_x = (struct bHYPRE_Vector__object*) 
    x.::bHYPRE::Vector::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Setup))(loc_self, /* in */ _local_b, /* in */ 
    _local_x, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Apply the operator to {\tt b}, returning {\tt x}.
 */
int32_t
bHYPRE::IJParCSRMatrix::Apply( /* in */::bHYPRE::Vector b, /* inout 
  */::bHYPRE::Vector& x )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  struct bHYPRE_Vector__object* _local_b = (struct bHYPRE_Vector__object*) 
    b.::bHYPRE::Vector::_get_ior();
  struct bHYPRE_Vector__object* _local_x = (struct bHYPRE_Vector__object*) 
    x.::bHYPRE::Vector::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_Apply))(loc_self, /* in */ _local_b, /* inout 
    */ &_local_x, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  x._set_ior( _local_x);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */
int32_t
bHYPRE::IJParCSRMatrix::ApplyAdjoint( /* in */::bHYPRE::Vector b, /* inout 
  */::bHYPRE::Vector& x )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  struct bHYPRE_Vector__object* _local_b = (struct bHYPRE_Vector__object*) 
    b.::bHYPRE::Vector::_get_ior();
  struct bHYPRE_Vector__object* _local_x = (struct bHYPRE_Vector__object*) 
    x.::bHYPRE::Vector::_get_ior();
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_ApplyAdjoint))(loc_self, /* in */ _local_b, 
    /* inout */ &_local_x, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  x._set_ior( _local_x);
  /*unpack results and cleanup*/
  return _result;
}



/**
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 */
int32_t
bHYPRE::IJParCSRMatrix::GetRow( /* in */int32_t row, /* out */int32_t& size, /* 
  out array<int,column-major> */::sidl::array<int32_t>& col_ind, /* out 
  array<double,column-major> */::sidl::array<double>& values )

{
  int32_t _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  struct sidl_int__array* _local_col_ind;
  struct sidl_double__array* _local_values;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  _result = (*(loc_self->d_epv->f_GetRow))(loc_self, /* in */ row, /* out */ 
    &size, /* out array<int,column-major> */ &_local_col_ind, /* out 
    array<double,column-major> */ &_local_values, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  col_ind._set_ior(_local_col_ind);
  values._set_ior(_local_values);
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
::bHYPRE::IJParCSRMatrix
bHYPRE::IJParCSRMatrix::_create() {
  struct sidl_BaseInterface__object * _exception;
  ::bHYPRE::IJParCSRMatrix self( (*_get_ext()->createObject)(NULL,&_exception), 
    false );
  if (_exception) {
    throwException0(_exception);
  }
  return self;
}

// Internal data wrapping method
::bHYPRE::IJParCSRMatrix::ior_t*
bHYPRE::IJParCSRMatrix::_wrapObj(void* private_data) {
  struct sidl_BaseInterface__object *_exception;
  ::bHYPRE::IJParCSRMatrix::ior_t* returnValue = (*_get_ext()->createObject)(
    private_data,&_exception);
  if (_exception) {
    throwException0(_exception);
  }
  return returnValue;
}

// remote constructor
::bHYPRE::IJParCSRMatrix
bHYPRE::IJParCSRMatrix::_create(const std::string& url) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception;
  ior_self = bHYPRE_IJParCSRMatrix__remoteCreate( url.c_str(), &_exception );
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  return ::bHYPRE::IJParCSRMatrix( ior_self, false );
}

// remote connector
::bHYPRE::IJParCSRMatrix
bHYPRE::IJParCSRMatrix::_connect(const std::string& url, const bool ar ) {
  ior_t* ior_self;
  struct sidl_BaseInterface__object *_exception;
  ior_self = bHYPRE_IJParCSRMatrix__remoteConnect( url.c_str(), ar?TRUE:FALSE, 
    &_exception );
  if (_exception != 0 ) {
    throwException0(_exception);
  }
  return ::bHYPRE::IJParCSRMatrix( ior_self, false );
}

// copy constructor
bHYPRE::IJParCSRMatrix::IJParCSRMatrix ( const ::bHYPRE::IJParCSRMatrix& 
  original ) {
  d_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    original.::bHYPRE::IJParCSRMatrix::_get_ior();
  if(d_self) {

    bHYPRE_CoefficientAccess_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_coefficientaccess);
    bHYPRE_Operator_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_operator);
    bHYPRE_IJMatrixView_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_ijmatrixview);
    bHYPRE_ProblemDefinition_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_problemdefinition);
    bHYPRE_MatrixVectorView_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_matrixvectorview);

    addRef();
  }
  d_weak_reference = false;
}

// assignment operator
::bHYPRE::IJParCSRMatrix&
bHYPRE::IJParCSRMatrix::operator=( const ::bHYPRE::IJParCSRMatrix& rhs ) {
  if ( d_self != rhs.d_self ) {
    if ( d_self != 0 ) {
      deleteRef();
    }
    d_self = (struct bHYPRE_IJParCSRMatrix__object*) 
      rhs.::bHYPRE::IJParCSRMatrix::_get_ior();
    if(d_self) {

      bHYPRE_CoefficientAccess_IORCache = &((*reinterpret_cast< ior_t*>(
        d_self)).d_bhypre_coefficientaccess);
      bHYPRE_Operator_IORCache = &((*reinterpret_cast< ior_t*>(
        d_self)).d_bhypre_operator);
      bHYPRE_IJMatrixView_IORCache = &((*reinterpret_cast< ior_t*>(
        d_self)).d_bhypre_ijmatrixview);
      bHYPRE_ProblemDefinition_IORCache = &((*reinterpret_cast< ior_t*>(
        d_self)).d_bhypre_problemdefinition);
      bHYPRE_MatrixVectorView_IORCache = &((*reinterpret_cast< ior_t*>(
        d_self)).d_bhypre_matrixvectorview);

      addRef();
    }
    d_weak_reference = false;
  }
  return *this;
}

// conversion from ior to C++ class
bHYPRE::IJParCSRMatrix::IJParCSRMatrix ( ::bHYPRE::IJParCSRMatrix::ior_t* ior ) 
   : StubBase(reinterpret_cast< void*>(ior)) { 
  if(d_self) {

    bHYPRE_CoefficientAccess_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_coefficientaccess);
    bHYPRE_Operator_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_operator);
    bHYPRE_IJMatrixView_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_ijmatrixview);
    bHYPRE_ProblemDefinition_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_problemdefinition);
    bHYPRE_MatrixVectorView_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_matrixvectorview);

  }
}

// Alternate constructor: does not call addRef()
// (sets d_weak_reference=isWeak)
// For internal use by Impls (fixes bug#275)
bHYPRE::IJParCSRMatrix::IJParCSRMatrix ( ::bHYPRE::IJParCSRMatrix::ior_t* ior, 
  bool isWeak ) : 
StubBase(reinterpret_cast< void*>(ior), isWeak){ 
  if(d_self) {

    bHYPRE_CoefficientAccess_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_coefficientaccess);
    bHYPRE_Operator_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_operator);
    bHYPRE_IJMatrixView_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_ijmatrixview);
    bHYPRE_ProblemDefinition_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_problemdefinition);
    bHYPRE_MatrixVectorView_IORCache = &((*reinterpret_cast< ior_t*>(
      d_self)).d_bhypre_matrixvectorview);

  }
}

// exec has special argument passing to avoid #include circularities
void ::bHYPRE::IJParCSRMatrix::_exec( const std::string& methodName, 
                        sidl::rmi::Call& inArgs,
                        sidl::rmi::Return& outArgs) { 
  ::bHYPRE::IJParCSRMatrix::ior_t* const loc_self = _get_ior();
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
bHYPRE::IJParCSRMatrix::_getURL(  )
// throws:
//       ::sidl::RuntimeException

{
  ::std::string _result;
  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
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
bHYPRE::IJParCSRMatrix::_set_hooks( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  ior_t* const loc_self = (struct bHYPRE_IJParCSRMatrix__object*) 
    ::bHYPRE::IJParCSRMatrix::_get_ior();
  sidl_bool _local_on = on;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  (*(loc_self->d_epv->f__set_hooks))(loc_self, /* in */ _local_on, &_exception 
    );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}


/**
 * Static Method to set whether or not method hooks should be invoked.
 */
void
bHYPRE::IJParCSRMatrix::_set_hooks_static( /* in */bool on )
// throws:
//       ::sidl::RuntimeException

{

  sidl_bool _local_on = on;
  sidl_BaseInterface__object * _exception;
  /*pack args to dispatch to ior*/
  ( _get_sepv()->f__set_hooks_static)( /* in */ _local_on, &_exception );
  /*dispatch to ior*/
  if (_exception != 0 ) {

    throwException0(_exception);
  }
  /*unpack results and cleanup*/
}

// protected method that implements casting
struct bHYPRE_IJParCSRMatrix__object* bHYPRE::IJParCSRMatrix::_cast(const void* 
  src)
{
  ior_t* cast = NULL;

  if(!connect_loaded) {
    struct sidl_BaseInterface__object *throwaway_exception;
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.IJParCSRMatrix", (
      void*)bHYPRE_IJParCSRMatrix__IHConnect, &throwaway_exception);
    connect_loaded = 1;
  }
  if ( src != 0 ) {
    // Actually, this thing is still const
    void* tmp = const_cast<void*>(src);
    struct sidl_BaseInterface__object *throwaway_exception;
    struct sidl_BaseInterface__object * base = reinterpret_cast< struct 
      sidl_BaseInterface__object *>(tmp);
    cast = reinterpret_cast< ior_t*>((*base->d_epv->f__cast)(base->d_object, 
      "bHYPRE.IJParCSRMatrix", &throwaway_exception));
  }
  return cast;
}

// Static data type
const ::bHYPRE::IJParCSRMatrix::ext_t * bHYPRE::IJParCSRMatrix::s_ext = 0;

// private static method to get static data type
const ::bHYPRE::IJParCSRMatrix::ext_t *
bHYPRE::IJParCSRMatrix::_get_ext()
  throw ( ::sidl::NullIORException)
{
  if (! s_ext ) {
#ifdef SIDL_STATIC_LIBRARY
    s_ext = bHYPRE_IJParCSRMatrix__externals();
#else
    s_ext = (struct bHYPRE_IJParCSRMatrix__external*)sidl_dynamicLoadIOR(
      "bHYPRE.IJParCSRMatrix","bHYPRE_IJParCSRMatrix__externals") ;
#endif
    sidl_checkIORVersion("bHYPRE.IJParCSRMatrix", s_ext->d_ior_major_version, 
      s_ext->d_ior_minor_version, 1, 0);
  }
  return s_ext;
}

