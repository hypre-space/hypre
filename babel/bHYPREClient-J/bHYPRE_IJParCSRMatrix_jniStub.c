/*
 * File:          bHYPRE_IJParCSRMatrix_jniStub.c
 * Symbol:        bHYPRE.IJParCSRMatrix-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side JNI glue code for bHYPRE.IJParCSRMatrix
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidl_Java.h"
#include "sidl_Loader.h"
#include "sidl_String.h"
#include "bHYPRE_IJParCSRMatrix_IOR.h"
#include "babel_config.h"
/*
 * Includes for all method dependencies.
 */

#ifndef included_bHYPRE_CoefficientAccess_jniStub_h
#include "bHYPRE_CoefficientAccess_jniStub.h"
#endif
#ifndef included_bHYPRE_IJMatrixView_jniStub_h
#include "bHYPRE_IJMatrixView_jniStub.h"
#endif
#ifndef included_bHYPRE_IJParCSRMatrix_jniStub_h
#include "bHYPRE_IJParCSRMatrix_jniStub.h"
#endif
#ifndef included_bHYPRE_MPICommunicator_jniStub_h
#include "bHYPRE_MPICommunicator_jniStub.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_jniStub_h
#include "bHYPRE_MatrixVectorView_jniStub.h"
#endif
#ifndef included_bHYPRE_Operator_jniStub_h
#include "bHYPRE_Operator_jniStub.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_jniStub_h
#include "bHYPRE_ProblemDefinition_jniStub.h"
#endif
#ifndef included_bHYPRE_Vector_jniStub_h
#include "bHYPRE_Vector_jniStub.h"
#endif
#ifndef included_sidl_BaseClass_jniStub_h
#include "sidl_BaseClass_jniStub.h"
#endif
#ifndef included_sidl_BaseInterface_jniStub_h
#include "sidl_BaseInterface_jniStub.h"
#endif
#ifndef included_sidl_ClassInfo_jniStub_h
#include "sidl_ClassInfo_jniStub.h"
#endif
#ifndef included_sidl_RuntimeException_jniStub_h
#include "sidl_RuntimeException_jniStub.h"
#endif

/*
 * Convert between jlong and void* pointers.
 */

#if (SIZEOF_VOID_P == 8)
#define JLONG_TO_POINTER(x) ((void*)(x))
#define POINTER_TO_JLONG(x) ((jlong)(x))
#else
#define JLONG_TO_POINTER(x) ((void*)(int32_t)(x))
#define POINTER_TO_JLONG(x) ((jlong)(int32_t)(x))
#endif

#ifndef NULL
#define NULL 0
#endif


#define LANG_SPECIFIC_INIT()

/*
 * connect_loaded is a boolean value showing if the IHConnect for this object has been loaded into the connectRegistry
 */

static int connect_loaded = 0;

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

/* Static variables to hold version of IOR */
static const int32_t s_IOR_MAJOR_VERSION = 1;
static const int32_t s_IOR_MINOR_VERSION = 0;

/* Static variables for managing EPV initialization. */
static int s_remote_initialized = 0;

static struct bHYPRE_IJParCSRMatrix__epv s_rem_epv__bhypre_ijparcsrmatrix;

static struct bHYPRE_CoefficientAccess__epv s_rem_epv__bhypre_coefficientaccess;

static struct bHYPRE_IJMatrixView__epv s_rem_epv__bhypre_ijmatrixview;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_Operator__epv s_rem_epv__bhypre_operator;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
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
    cast =  (*func)(((struct bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih,
      _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_IJParCSRMatrix__delete(
  struct bHYPRE_IJParCSRMatrix__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
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

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_IJParCSRMatrix__raddRef(
  struct bHYPRE_IJParCSRMatrix__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
  sidl_rmi_Response _rsvp = NULL;
  sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
    "addRef", _ex ); SIDL_CHECK(*_ex);
  /* send actual RMI request */
  _rsvp = sidl_rmi_Invocation_invokeMethod(_inv,_ex);SIDL_CHECK(*_ex);
  /* Check for exceptions */
  netex = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);
  if(netex != NULL) {
    sidl_BaseInterface throwaway_exception = NULL;
    *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(netex,
      &throwaway_exception);
    return;
  }

  /* cleanup and return */
  EXIT:
  if(_inv) { sidl_rmi_Invocation_deleteRef(_inv,&_throwaway); }
  if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp,&_throwaway); }
  return;
}

/* REMOTE ISREMOTE: returns true if this object is Remote (it is). */
static sidl_bool
remote_bHYPRE_IJParCSRMatrix__isRemote(
    struct bHYPRE_IJParCSRMatrix__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_IJParCSRMatrix__set_hooks(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* in */ sidl_bool on,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix._set_hooks.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE EXEC: call the exec function for the object. */
static void remote_bHYPRE_IJParCSRMatrix__exec(
  struct bHYPRE_IJParCSRMatrix__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:SetDiagOffdSizes */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetDiagOffdSizes", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "diag_sizes", diag_sizes,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "offdiag_sizes", offdiag_sizes,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDiagOffdSizes.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:addRef */
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

/* REMOTE METHOD STUB:deleteRef */
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

/* REMOTE METHOD STUB:isSame */
static sidl_bool
remote_bHYPRE_IJParCSRMatrix_isSame(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* in */ struct sidl_BaseInterface__object* iobj,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isSame", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(iobj){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)iobj, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "iobj", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "iobj", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.isSame.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:isType */
static sidl_bool
remote_bHYPRE_IJParCSRMatrix_isType(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* in */ const char* name,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    sidl_bool _retval = FALSE;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.isType.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackBool( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:getClassInfo */
static struct sidl_ClassInfo__object*
remote_bHYPRE_IJParCSRMatrix_getClassInfo(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char*_retval_str = NULL;
    struct sidl_ClassInfo__object* _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.getClassInfo.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackString( _rsvp, "_retval", &_retval_str, 
      _ex);SIDL_CHECK(*_ex);
    _retval = sidl_ClassInfo__connectI(_retval_str, FALSE, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetLocalRange */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetLocalRange", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "ilower", ilower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "iupper", iupper, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "jlower", jlower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "jupper", jupper, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetLocalRange.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetValues */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:AddToValues */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "AddToValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.AddToValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetLocalRange */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetLocalRange", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetLocalRange.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "ilower", ilower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackInt( _rsvp, "iupper", iupper, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackInt( _rsvp, "jlower", jlower, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackInt( _rsvp, "jupper", jupper, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetRowCounts */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetRowCounts", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", *ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetRowCounts.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackIntArray( _rsvp, "ncols", ncols,
      sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetValues */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "ncols", ncols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "rows", rows,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "cols", cols,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", *values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetValues.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDoubleArray( _rsvp, "values", values,
      sidl_column_major_order,1,TRUE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetRowSizes */
static int32_t
remote_bHYPRE_IJParCSRMatrix_SetRowSizes(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* in rarray[nrows] */ struct sidl_int__array* sizes,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetRowSizes", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packIntArray( _inv, "sizes", sizes,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetRowSizes.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Print */
static int32_t
remote_bHYPRE_IJParCSRMatrix_Print(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* in */ const char* filename,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Print", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex);SIDL_CHECK(
      *_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Print.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Read */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Read", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex);SIDL_CHECK(
      *_ex);
    if(comm){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)comm, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "comm", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "comm", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Read.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_IJParCSRMatrix_SetCommunicator(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetCommunicator", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
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

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetCommunicator.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Destroy */
static void
remote_bHYPRE_IJParCSRMatrix_Destroy(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Destroy.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return;
  }
}

/* REMOTE METHOD STUB:Initialize */
static int32_t
remote_bHYPRE_IJParCSRMatrix_Initialize(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Initialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Initialize.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Assemble */
static int32_t
remote_bHYPRE_IJParCSRMatrix_Assemble(
  /* in */ struct bHYPRE_IJParCSRMatrix__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Assemble.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetIntParameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetIntParameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetIntParameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetDoubleParameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetDoubleParameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDoubleParameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetStringParameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetStringParameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packString( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetStringParameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetIntArray1Parameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetIntArray1Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "value", value,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetIntArray1Parameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetIntArray2Parameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetIntArray2Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "value", value,
      sidl_column_major_order,2,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetIntArray2Parameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetDoubleArray1Parameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetDoubleArray1Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDoubleArray1Parameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetDoubleArray2Parameter */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetDoubleArray2Parameter", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "value", value,
      sidl_column_major_order,2,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.SetDoubleArray2Parameter.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetIntValue */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetIntValue", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetIntValue.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "value", value, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetDoubleValue */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetDoubleValue", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetDoubleValue.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "value", value, _ex);SIDL_CHECK(
      *_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Setup */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Setup", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
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

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Setup.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Apply */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* x_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Apply", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
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
    /* Transfer this reference */
    if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
      SIDL_CHECK(*_ex);
      (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
        sidl_BaseInterface)*x)->d_object, _ex);SIDL_CHECK(*_ex);
      sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x, _ex);SIDL_CHECK(
        *_ex); 
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.Apply.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "x", &x_str, _ex);SIDL_CHECK(*_ex);
    *x = bHYPRE_Vector__connectI(x_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:ApplyAdjoint */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* x_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "ApplyAdjoint", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
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
    /* Transfer this reference */
    if(*x && sidl_BaseInterface__isRemote((sidl_BaseInterface)*x, _ex)) {
      SIDL_CHECK(*_ex);
      (*((sidl_BaseInterface)*x)->d_epv->f__raddRef)(((
        sidl_BaseInterface)*x)->d_object, _ex);SIDL_CHECK(*_ex);
      sidl_BaseInterface_deleteRef((sidl_BaseInterface)*x, _ex);SIDL_CHECK(
        *_ex); 
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.ApplyAdjoint.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "x", &x_str, _ex);SIDL_CHECK(*_ex);
    *x = bHYPRE_Vector__connectI(x_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:GetRow */
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
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_IJParCSRMatrix__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetRow", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "row", row, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.IJParCSRMatrix.GetRow.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackInt( _rsvp, "size", size, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackIntArray( _rsvp, "col_ind", col_ind,
      sidl_column_major_order,1,FALSE, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Response_unpackDoubleArray( _rsvp, "values", values,
      sidl_column_major_order,1,FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_IJParCSRMatrix__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_IJParCSRMatrix__epv*    epv = &s_rem_epv__bhypre_ijparcsrmatrix;
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
  epv->f__raddRef                      = remote_bHYPRE_IJParCSRMatrix__raddRef;
  epv->f__isRemote                     = remote_bHYPRE_IJParCSRMatrix__isRemote;
  epv->f__set_hooks                    = 
    remote_bHYPRE_IJParCSRMatrix__set_hooks;
  epv->f__ctor                         = NULL;
  epv->f__ctor2                        = NULL;
  epv->f__dtor                         = NULL;
  epv->f_SetDiagOffdSizes              = 
    remote_bHYPRE_IJParCSRMatrix_SetDiagOffdSizes;
  epv->f_addRef                        = remote_bHYPRE_IJParCSRMatrix_addRef;
  epv->f_deleteRef                     = remote_bHYPRE_IJParCSRMatrix_deleteRef;
  epv->f_isSame                        = remote_bHYPRE_IJParCSRMatrix_isSame;
  epv->f_isType                        = remote_bHYPRE_IJParCSRMatrix_isType;
  epv->f_getClassInfo                  = 
    remote_bHYPRE_IJParCSRMatrix_getClassInfo;
  epv->f_SetLocalRange                 = 
    remote_bHYPRE_IJParCSRMatrix_SetLocalRange;
  epv->f_SetValues                     = remote_bHYPRE_IJParCSRMatrix_SetValues;
  epv->f_AddToValues                   = 
    remote_bHYPRE_IJParCSRMatrix_AddToValues;
  epv->f_GetLocalRange                 = 
    remote_bHYPRE_IJParCSRMatrix_GetLocalRange;
  epv->f_GetRowCounts                  = 
    remote_bHYPRE_IJParCSRMatrix_GetRowCounts;
  epv->f_GetValues                     = remote_bHYPRE_IJParCSRMatrix_GetValues;
  epv->f_SetRowSizes                   = 
    remote_bHYPRE_IJParCSRMatrix_SetRowSizes;
  epv->f_Print                         = remote_bHYPRE_IJParCSRMatrix_Print;
  epv->f_Read                          = remote_bHYPRE_IJParCSRMatrix_Read;
  epv->f_SetCommunicator               = 
    remote_bHYPRE_IJParCSRMatrix_SetCommunicator;
  epv->f_Destroy                       = remote_bHYPRE_IJParCSRMatrix_Destroy;
  epv->f_Initialize                    = 
    remote_bHYPRE_IJParCSRMatrix_Initialize;
  epv->f_Assemble                      = remote_bHYPRE_IJParCSRMatrix_Assemble;
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
  e0->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e1->f__cast           = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
    epv->f__cast;
  e1->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
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
    sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,struct 
    sidl_BaseInterface__object **)) epv->f_SetValues;
  e1->f_AddToValues     = (int32_t (*)(void*,struct sidl_int__array*,struct 
    sidl_int__array*,struct sidl_int__array*,struct sidl_double__array*,struct 
    sidl_BaseInterface__object **)) epv->f_AddToValues;
  e1->f_GetLocalRange   = (int32_t (*)(void*,int32_t*,int32_t*,int32_t*,
    int32_t*,struct sidl_BaseInterface__object **)) epv->f_GetLocalRange;
  e1->f_GetRowCounts    = (int32_t (*)(void*,struct sidl_int__array*,struct 
    sidl_int__array**,struct sidl_BaseInterface__object **)) 
    epv->f_GetRowCounts;
  e1->f_GetValues       = (int32_t (*)(void*,struct sidl_int__array*,struct 
    sidl_int__array*,struct sidl_int__array*,struct sidl_double__array**,struct 
    sidl_BaseInterface__object **)) epv->f_GetValues;
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
  e1->f_Initialize      = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Initialize;
  e1->f_Assemble        = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Assemble;
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
  e2->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e2->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e2->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
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
  e2->f_Initialize      = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Initialize;
  e2->f_Assemble        = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Assemble;
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
  e3->f_SetDoubleParameter       = (int32_t (*)(void*,const char*,double,struct 
    sidl_BaseInterface__object **)) epv->f_SetDoubleParameter;
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
  e4->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e4->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e4->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
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
  e4->f_Initialize      = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Initialize;
  e4->f_Assemble        = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Assemble;
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
  e6->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e6->f_isType       = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e6->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
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
      sidl_BaseInterface)sidl_rmi_InstanceRegistry_getInstanceByString(objectID,
      _ex); SIDL_CHECK(*_ex);
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

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
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
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
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

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
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
/* REMOTE: generate remote instance given URL string. */
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

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
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
/*
 * Cast method for interface and class type conversions.
 */

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

/*
 * RMI connector function for the class.
 */

struct bHYPRE_IJParCSRMatrix__object*
bHYPRE_IJParCSRMatrix__connectI(const char* url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex)
{
  return bHYPRE_IJParCSRMatrix__remoteConnect(url, ar, _ex);
}

/*
 * Function to extract IOR reference from the Java object.
 */

static struct bHYPRE_IJParCSRMatrix__object* _get_ior(
  JNIEnv* env,
  jobject obj)
{
  void* ptr = NULL;
  static jmethodID mid = (jmethodID) NULL;

  if (mid == (jmethodID) NULL) {
    jclass cls = (*env)->GetObjectClass(env, obj);
    mid = (*env)->GetMethodID(env, cls, "_get_ior", "()J");
    (*env)->DeleteLocalRef(env, cls);
  }

  ptr = JLONG_TO_POINTER((*env)->CallLongMethod(env, obj, mid));
  return (struct bHYPRE_IJParCSRMatrix__object*) ptr;
}

/*
 * External reference to IOR methods.
 */

static const struct bHYPRE_IJParCSRMatrix__external* s_external = NULL;

/*
 * External reference to static EPV.
 */

static const struct bHYPRE_IJParCSRMatrix__sepv* s_sepv = NULL;

/*
 * Create object instance and return reference.
 */

static jlong jni__create_ior(
  JNIEnv* env,
  jclass  cls)
{
  (void) env;
  (void) cls;
  struct sidl_BaseInterface__object *_ex = NULL;
  jlong _res = POINTER_TO_JLONG((*s_external->createObject)(NULL, &_ex));
  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  return _res;

}

/*
 * Create a remote object instance and return reference.
 */

static jlong jni__create_remote_ior(
  JNIEnv* env,
  jclass  cls,
  jstring url)
{
  (void) env;
  (void) cls;
  struct sidl_BaseInterface__object *_ex = NULL;
  jlong _res = 0;
  char* _tmp_url = sidl_Java_J2I_string(env, url);
  _res = POINTER_TO_JLONG(bHYPRE_IJParCSRMatrix__remoteCreate(_tmp_url, &_ex));
  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  return _res;

}

/*
 * Create object instance and seet ddata to be the passed in object.
 */

static jlong jni__wrap(
  JNIEnv* env,
  jclass  cls,
  jobject obj)
{
  (void) env;
  (void) cls;
  struct sidl_BaseInterface__object *_ex = NULL;
  jlong _res = 0;
  void* g_data = (void*)(*env)->NewGlobalRef(env, obj);
  struct bHYPRE_IJParCSRMatrix__object *_ptr = (*s_external->createObject)(
    g_data, &_ex);
  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = POINTER_TO_JLONG(_ptr);
  return _res;

}

/*
 * Connect to a remote object instance and return reference.
 */

static jlong jni__connect_remote_ior(
  JNIEnv* env,
  jclass  cls,
  jstring url)
{
  (void) env;
  (void) cls;
  struct sidl_BaseInterface__object *_ex = NULL;
  jlong _res = 0;
  char* _tmp_url = sidl_Java_J2I_string(env, url);
  _res = POINTER_TO_JLONG(bHYPRE_IJParCSRMatrix__remoteConnect(_tmp_url, 1, 
    &_ex));
  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  return _res;

}

/*
 *  This function is the preferred way to create an IJParCSR Matrix. 
 */

static jobject
jni_Create(
  JNIEnv* env,
  jclass  cls,
  jobject _arg_mpi_comm,
  jint _arg_ilower,
  jint _arg_iupper,
  jint _arg_jlower,
  jint _arg_jupper)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_MPICommunicator__object* _tmp_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  int32_t _tmp_ilower = 0;
  int32_t _tmp_iupper = 0;
  int32_t _tmp_jlower = 0;
  int32_t _tmp_jupper = 0;
  struct bHYPRE_IJParCSRMatrix__object* _ior_res = (struct 
    bHYPRE_IJParCSRMatrix__object*) NULL;
  jobject _res = (jobject) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _tmp_mpi_comm = (struct bHYPRE_MPICommunicator__object*) sidl_Java_J2I_cls(
    env, _arg_mpi_comm, FALSE);JAVA_CHECK(env);
  _tmp_ilower = (int32_t) _arg_ilower;
  _tmp_iupper = (int32_t) _arg_iupper;
  _tmp_jlower = (int32_t) _arg_jlower;
  _tmp_jupper = (int32_t) _arg_jupper;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(s_sepv->f_Create))(
    _tmp_mpi_comm,
    _tmp_ilower,
    _tmp_iupper,
    _tmp_jlower,
    _tmp_jupper,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = sidl_Java_I2J_cls(env, _ior_res, "bHYPRE.IJParCSRMatrix", 
    FALSE);JAVA_CHECK(env);

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * Method:  GenerateLaplacian[]
 */

static jobject
jni_GenerateLaplacian(
  JNIEnv* env,
  jclass  cls,
  jobject _arg_mpi_comm,
  jint _arg_nx,
  jint _arg_ny,
  jint _arg_nz,
  jint _arg_Px,
  jint _arg_Py,
  jint _arg_Pz,
  jint _arg_p,
  jint _arg_q,
  jint _arg_r,
  jobject _arg_values,
  jint _arg_discretization)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_MPICommunicator__object* _tmp_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  int32_t _tmp_nx = 0;
  int32_t _tmp_ny = 0;
  int32_t _tmp_nz = 0;
  int32_t _tmp_Px = 0;
  int32_t _tmp_Py = 0;
  int32_t _tmp_Pz = 0;
  int32_t _tmp_p = 0;
  int32_t _tmp_q = 0;
  int32_t _tmp_r = 0;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _tmp_discretization = 0;
  struct bHYPRE_IJParCSRMatrix__object* _ior_res = (struct 
    bHYPRE_IJParCSRMatrix__object*) NULL;
  jobject _res = (jobject) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _tmp_mpi_comm = (struct bHYPRE_MPICommunicator__object*) sidl_Java_J2I_cls(
    env, _arg_mpi_comm, FALSE);JAVA_CHECK(env);
  _tmp_nx = (int32_t) _arg_nx;
  _tmp_ny = (int32_t) _arg_ny;
  _tmp_nz = (int32_t) _arg_nz;
  _tmp_Px = (int32_t) _arg_Px;
  _tmp_Py = (int32_t) _arg_Py;
  _tmp_Pz = (int32_t) _arg_Pz;
  _tmp_p = (int32_t) _arg_p;
  _tmp_q = (int32_t) _arg_q;
  _tmp_r = (int32_t) _arg_r;
  _tmp_values = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_values);
  _tmp_discretization = (int32_t) _arg_discretization;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(s_sepv->f_GenerateLaplacian))(
    _tmp_mpi_comm,
    _tmp_nx,
    _tmp_ny,
    _tmp_nz,
    _tmp_Px,
    _tmp_Py,
    _tmp_Pz,
    _tmp_p,
    _tmp_q,
    _tmp_r,
    _tmp_values,
    _tmp_discretization,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = sidl_Java_I2J_cls(env, _ior_res, "bHYPRE.IJParCSRMatrix", 
    FALSE);JAVA_CHECK(env);

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
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

static jint
jni_SetDiagOffdSizes(
  JNIEnv* env,
  jobject obj,
  jobject _arg_diag_sizes,
  jobject _arg_offdiag_sizes)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_int__array* _tmp_diag_sizes = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_offdiag_sizes = (struct sidl_int__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_diag_sizes = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_diag_sizes);
  _tmp_offdiag_sizes = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env,
    _arg_offdiag_sizes);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetDiagOffdSizes))(
    _ior,
    _tmp_diag_sizes,
    _tmp_offdiag_sizes,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
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

static jint
jni_SetLocalRange(
  JNIEnv* env,
  jobject obj,
  jint _arg_ilower,
  jint _arg_iupper,
  jint _arg_jlower,
  jint _arg_jupper)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  int32_t _tmp_ilower = 0;
  int32_t _tmp_iupper = 0;
  int32_t _tmp_jlower = 0;
  int32_t _tmp_jupper = 0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_ilower = (int32_t) _arg_ilower;
  _tmp_iupper = (int32_t) _arg_iupper;
  _tmp_jlower = (int32_t) _arg_jlower;
  _tmp_jupper = (int32_t) _arg_jupper;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetLocalRange))(
    _ior,
    _tmp_ilower,
    _tmp_iupper,
    _tmp_jlower,
    _tmp_jupper,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
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

static jint
jni_SetValues(
  JNIEnv* env,
  jobject obj,
  jobject _arg_ncols,
  jobject _arg_rows,
  jobject _arg_cols,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_int__array* _tmp_ncols = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_rows = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_cols = (struct sidl_int__array*) NULL;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_ncols = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_ncols);
  _tmp_rows = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_rows);
  _tmp_cols = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_cols);
  _tmp_values = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_values);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetValues))(
    _ior,
    _tmp_ncols,
    _tmp_rows,
    _tmp_cols,
    _tmp_values,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Adds to values for {\tt nrows} of the matrix.  Usage details
 * are analogous to {\tt SetValues}.  Adds to any previous
 * values at the specified locations, or, if there was no value
 * there before, inserts a new one.
 * 
 * Not collective.
 */

static jint
jni_AddToValues(
  JNIEnv* env,
  jobject obj,
  jobject _arg_ncols,
  jobject _arg_rows,
  jobject _arg_cols,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_int__array* _tmp_ncols = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_rows = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_cols = (struct sidl_int__array*) NULL;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_ncols = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_ncols);
  _tmp_rows = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_rows);
  _tmp_cols = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_cols);
  _tmp_values = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_values);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_AddToValues))(
    _ior,
    _tmp_ncols,
    _tmp_rows,
    _tmp_cols,
    _tmp_values,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Gets range of rows owned by this processor and range of
 * column partitioning for this processor.
 */

static jint
jni_GetLocalRange(
  JNIEnv* env,
  jobject obj,
  jobject _arg_ilower,
  jobject _arg_iupper,
  jobject _arg_jlower,
  jobject _arg_jupper)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  int32_t _tmp_ilower = 0;
  int32_t _tmp_iupper = 0;
  int32_t _tmp_jlower = 0;
  int32_t _tmp_jupper = 0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  if(_arg_ilower== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }
  if(_arg_iupper== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }
  if(_arg_jlower== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }
  if(_arg_jupper== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetLocalRange))(
    _ior,
    &_tmp_ilower,
    &_tmp_iupper,
    &_tmp_jlower,
    &_tmp_jupper,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_Java_I2J_int_holder(env, _arg_ilower, _tmp_ilower);
  sidl_Java_I2J_int_holder(env, _arg_iupper, _tmp_iupper);
  sidl_Java_I2J_int_holder(env, _arg_jlower, _tmp_jlower);
  sidl_Java_I2J_int_holder(env, _arg_jupper, _tmp_jupper);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Gets number of nonzeros elements for {\tt nrows} rows
 * specified in {\tt rows} and returns them in {\tt ncols},
 * which needs to be allocated by the user.
 */

static jint
jni_GetRowCounts(
  JNIEnv* env,
  jobject obj,
  jobject _arg_rows,
  jobject _arg_ncols)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_int__array* _tmp_rows = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_ncols = (struct sidl_int__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_rows = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_rows);
  if(_arg_ncols== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as INOUT Argument");
    return 0;
    }
  _tmp_ncols = (struct sidl_int__array*) sidl_Java_J2I_array_holder(env, 
    _arg_ncols, "sidl.Integer$Array1");

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetRowCounts))(
    _ior,
    _tmp_rows,
    &_tmp_ncols,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_Java_I2J_array_holder(env, _arg_ncols, _tmp_ncols, 
    "sidl.Integer$Array1");
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Gets values for {\tt nrows} rows or partial rows of the
 * matrix.  Usage details are analogous to {\tt SetValues}.
 */

static jint
jni_GetValues(
  JNIEnv* env,
  jobject obj,
  jobject _arg_ncols,
  jobject _arg_rows,
  jobject _arg_cols,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_int__array* _tmp_ncols = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_rows = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_cols = (struct sidl_int__array*) NULL;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_ncols = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_ncols);
  _tmp_rows = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_rows);
  _tmp_cols = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_cols);
  if(_arg_values== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as INOUT Argument");
    return 0;
    }
  _tmp_values = (struct sidl_double__array*) sidl_Java_J2I_array_holder(env, 
    _arg_values, "sidl.Double$Array1");

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetValues))(
    _ior,
    _tmp_ncols,
    _tmp_rows,
    _tmp_cols,
    &_tmp_values,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_Java_I2J_array_holder(env, _arg_values, _tmp_values, 
    "sidl.Double$Array1");
  _res = (jint) _ior_res;

  return _res;
}

/*
 * (Optional) Set the max number of nonzeros to expect in each
 * row.  The array {\tt sizes} contains estimated sizes for each
 * row on this process.  The integer nrows is the number of rows in
 * the local matrix.  This call can significantly improve the
 * efficiency of matrix construction, and should always be
 * utilized if possible.
 * 
 * Not collective.
 */

static jint
jni_SetRowSizes(
  JNIEnv* env,
  jobject obj,
  jobject _arg_sizes)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_int__array* _tmp_sizes = (struct sidl_int__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_sizes = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_sizes);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetRowSizes))(
    _ior,
    _tmp_sizes,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Print the matrix to file.  This is mainly for debugging
 * purposes.
 */

static jint
jni_Print(
  JNIEnv* env,
  jobject obj,
  jstring _arg_filename)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_filename = (char*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_filename = sidl_Java_J2I_string(env, _arg_filename);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Print))(
    _ior,
    _tmp_filename,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_filename);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Read the matrix from file.  This is mainly for debugging
 * purposes.
 */

static jint
jni_Read(
  JNIEnv* env,
  jobject obj,
  jstring _arg_filename,
  jobject _arg_comm)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_filename = (char*) NULL;
  struct bHYPRE_MPICommunicator__object* _tmp_comm = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_filename = sidl_Java_J2I_string(env, _arg_filename);
  _tmp_comm = (struct bHYPRE_MPICommunicator__object*) sidl_Java_J2I_cls(env, 
    _arg_comm, FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Read))(
    _ior,
    _tmp_filename,
    _tmp_comm,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_filename);
  _res = (jint) _ior_res;

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * Set the MPI Communicator.  DEPRECATED, Use Create()
 */

static jint
jni_SetCommunicator(
  JNIEnv* env,
  jobject obj,
  jobject _arg_mpi_comm)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct bHYPRE_MPICommunicator__object* _tmp_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_mpi_comm = (struct bHYPRE_MPICommunicator__object*) sidl_Java_J2I_cls(
    env, _arg_mpi_comm, FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetCommunicator))(
    _ior,
    _tmp_mpi_comm,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * The Destroy function doesn't necessarily destroy anything.
 * It is just another name for deleteRef.  Thus it decrements the
 * object's reference count.  The Babel memory management system will
 * destroy the object if the reference count goes to zero.
 */

static void
jni_Destroy(
  JNIEnv* env,
  jobject obj)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);

  /*
   * Call the IOR method through the EPV.
   */

  (*(_ior->d_epv->f_Destroy))(
    _ior,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);
    return;
  }
}

/*
 * Prepare an object for setting coefficient values, whether for
 * the first time or subsequently.
 */

static jint
jni_Initialize(
  JNIEnv* env,
  jobject obj)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Initialize))(
    _ior,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Finalize the construction of an object before using, either
 * for the first time or on subsequent uses. {\tt Initialize}
 * and {\tt Assemble} always appear in a matched set, with
 * Initialize preceding Assemble. Values can only be set in
 * between a call to Initialize and Assemble.
 */

static jint
jni_Assemble(
  JNIEnv* env,
  jobject obj)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Assemble))(
    _ior,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the int parameter associated with {\tt name}.
 */

static jint
jni_SetIntParameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jint _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  int32_t _tmp_value = 0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = (int32_t) _arg_value;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetIntParameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the double parameter associated with {\tt name}.
 */

static jint
jni_SetDoubleParameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jdouble _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  double _tmp_value = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = (double) _arg_value;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetDoubleParameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the string parameter associated with {\tt name}.
 */

static jint
jni_SetStringParameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jstring _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  char* _tmp_value = (char*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = sidl_Java_J2I_string(env, _arg_value);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetStringParameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  sidl_String_free(_tmp_value);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the int 1-D array parameter associated with {\tt name}.
 */

static jint
jni_SetIntArray1Parameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  struct sidl_int__array* _tmp_value = (struct sidl_int__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_value);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetIntArray1Parameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the int 2-D array parameter associated with {\tt name}.
 */

static jint
jni_SetIntArray2Parameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  struct sidl_int__array* _tmp_value = (struct sidl_int__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_value);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetIntArray2Parameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the double 1-D array parameter associated with {\tt name}.
 */

static jint
jni_SetDoubleArray1Parameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  struct sidl_double__array* _tmp_value = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_value);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetDoubleArray1Parameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the double 2-D array parameter associated with {\tt name}.
 */

static jint
jni_SetDoubleArray2Parameter(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  struct sidl_double__array* _tmp_value = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  _tmp_value = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_value);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetDoubleArray2Parameter))(
    _ior,
    _tmp_name,
    _tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set the int parameter associated with {\tt name}.
 */

static jint
jni_GetIntValue(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  int32_t _tmp_value = 0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  if(_arg_value== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetIntValue))(
    _ior,
    _tmp_name,
    &_tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  sidl_Java_I2J_int_holder(env, _arg_value, _tmp_value);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Get the double parameter associated with {\tt name}.
 */

static jint
jni_GetDoubleValue(
  JNIEnv* env,
  jobject obj,
  jstring _arg_name,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_name = (char*) NULL;
  double _tmp_value = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_name = sidl_Java_J2I_string(env, _arg_name);
  if(_arg_value== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetDoubleValue))(
    _ior,
    _tmp_name,
    &_tmp_value,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_String_free(_tmp_name);
  sidl_Java_I2J_double_holder(env, _arg_value, _tmp_value);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * (Optional) Do any preprocessing that may be necessary in
 * order to execute {\tt Apply}.
 */

static jint
jni_Setup(
  JNIEnv* env,
  jobject obj,
  jobject _arg_b,
  jobject _arg_x)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct bHYPRE_Vector__object* _tmp_b = (struct bHYPRE_Vector__object*) NULL;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_b = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_b, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);
  _tmp_x = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_x, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Setup))(
    _ior,
    _tmp_b,
    _tmp_x,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  _res = (jint) _ior_res;

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * Apply the operator to {\tt b}, returning {\tt x}.
 */

static jint
jni_Apply(
  JNIEnv* env,
  jobject obj,
  jobject _arg_b,
  jobject _arg_x)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct bHYPRE_Vector__object* _tmp_b = (struct bHYPRE_Vector__object*) NULL;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_b = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_b, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);
  if(_arg_x== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as INOUT Argument");
    return 0;
    }
  _tmp_x = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc_holder(env, _arg_x,
    "bHYPRE.Vector", TRUE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Apply))(
    _ior,
    _tmp_b,
    &_tmp_x,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_Java_I2J_ifc_holder(env, _arg_x, _tmp_x, "bHYPRE.Vector", 
    FALSE);JAVA_CHECK(env);
  _res = (jint) _ior_res;

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * Apply the adjoint of the operator to {\tt b}, returning {\tt x}.
 */

static jint
jni_ApplyAdjoint(
  JNIEnv* env,
  jobject obj,
  jobject _arg_b,
  jobject _arg_x)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  struct bHYPRE_Vector__object* _tmp_b = (struct bHYPRE_Vector__object*) NULL;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_b = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_b, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);
  if(_arg_x== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as INOUT Argument");
    return 0;
    }
  _tmp_x = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc_holder(env, _arg_x,
    "bHYPRE.Vector", TRUE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_ApplyAdjoint))(
    _ior,
    _tmp_b,
    &_tmp_x,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_Java_I2J_ifc_holder(env, _arg_x, _tmp_x, "bHYPRE.Vector", 
    FALSE);JAVA_CHECK(env);
  _res = (jint) _ior_res;

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * The GetRow method will allocate space for its two output
 * arrays on the first call.  The space will be reused on
 * subsequent calls.  Thus the user must not delete them, yet
 * must not depend on the data from GetRow to persist beyond the
 * next GetRow call.
 */

static jint
jni_GetRow(
  JNIEnv* env,
  jobject obj,
  jint _arg_row,
  jobject _arg_size,
  jobject _arg_col_ind,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  int32_t _tmp_row = 0;
  int32_t _tmp_size = 0;
  struct sidl_int__array* _tmp_col_ind = (struct sidl_int__array*) NULL;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_row = (int32_t) _arg_row;
  if(_arg_size== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }
  if(_arg_col_ind== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }
  if(_arg_values== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetRow))(
    _ior,
    _tmp_row,
    &_tmp_size,
    &_tmp_col_ind,
    &_tmp_values,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);

    return _res;
  }
  sidl_Java_I2J_int_holder(env, _arg_size, _tmp_size);
  sidl_Java_I2J_array_holder(env, _arg_col_ind, _tmp_col_ind, 
    "sidl.Integer$Array1");
  sidl_Java_I2J_array_holder(env, _arg_values, _tmp_values, 
    "sidl.Double$Array1");
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Select and execute a method by name
 */

static void
jni__exec(
  JNIEnv* env,
  jobject obj,
  jstring _arg_methodName,
  jobject _arg_inArgs,
  jobject _arg_outArgs)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  char* _tmp_methodName = (char*) NULL;
  struct sidl_rmi_Call__object* _tmp_inArgs = (struct sidl_rmi_Call__object*) 
    NULL;
  struct sidl_rmi_Return__object* _tmp_outArgs = (struct 
    sidl_rmi_Return__object*) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_methodName = sidl_Java_J2I_string(env, _arg_methodName);
  _tmp_inArgs = (struct sidl_rmi_Call__object*) sidl_Java_J2I_ifc(env, 
    _arg_inArgs, "sidl.rmi.Call", FALSE);JAVA_CHECK(env);
  _tmp_outArgs = (struct sidl_rmi_Return__object*) sidl_Java_J2I_ifc(env, 
    _arg_outArgs, "sidl.rmi.Return", FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  (*(_ior->d_epv->f__exec))(
    _ior,
    _tmp_methodName,
    _tmp_inArgs,
    _tmp_outArgs,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);
    return;
  }
  sidl_String_free(_tmp_methodName);
  JAVA_EXIT:
  return;
}
/*
 * Method to set whether or not method hooks should be invoked.
 */

static void
jni__set_hooks(
  JNIEnv* env,
  jobject obj,
  jboolean _arg_on)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_IJParCSRMatrix__object* _ior = NULL;
  sidl_bool _tmp_on = FALSE;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_on = (sidl_bool) _arg_on;

  /*
   * Call the IOR method through the EPV.
   */

  (*(_ior->d_epv->f__set_hooks))(
    _ior,
    _tmp_on,
    &_ex);

  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);
    return;
  }
}

/*
 * Static Method to set whether or not method hooks should be invoked.
 */

static void
jni__set_hooks_static(
  JNIEnv* env,
  jclass  cls,
  jboolean _arg_on)
{
  /*
   * Declare return and temporary variables.
   */

  sidl_bool _tmp_on = FALSE;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _tmp_on = (sidl_bool) _arg_on;

  /*
   * Call the IOR method through the EPV.
   */

  (*(s_sepv->f__set_hooks_static))(
    _tmp_on,
    &_ex);

  s_sepv = (*(s_external->getStaticEPV))();
  /*
   * Postprocess OUT, INOUT, returns, and exceptions.
   */

  if(_ex) {
    sidl_Java_CheckException(
      env,
      _ex,
      "sidl.RuntimeException",
      NULL);
    return;
  }
}

/*
 * Register JNI methods with the Java JVM.
 */

void bHYPRE_IJParCSRMatrix__register(JNIEnv* env)
{
  JNINativeMethod methods[36];
  jclass cls;

#ifdef SIDL_STATIC_LIBRARY
  s_external = bHYPRE_IJParCSRMatrix__externals();
#else
  s_external = (struct bHYPRE_IJParCSRMatrix__external*)sidl_dynamicLoadIOR(
    "bHYPRE.IJParCSRMatrix","bHYPRE_IJParCSRMatrix__externals") ;
  sidl_checkIORVersion("bHYPRE.IJParCSRMatrix", s_external->d_ior_major_version,
    s_external->d_ior_minor_version, 1, 0);
#endif

  s_sepv = (*(s_external->getStaticEPV))();

  methods[0].name      = "_connect_remote_ior";
  methods[0].signature = "(Ljava/lang/String;)J";
  methods[0].fnPtr     = (void *)jni__connect_remote_ior;
  methods[1].name      = "_create_remote_ior";
  methods[1].signature = "(Ljava/lang/String;)J";
  methods[1].fnPtr     = (void *)jni__create_remote_ior;
  methods[2].name      = "_wrap";
  methods[2].signature = "(LbHYPRE/IJParCSRMatrix;)J";
  methods[2].fnPtr     = (void *)jni__wrap;
  methods[3].name      = "_create_ior";
  methods[3].signature = "()J";
  methods[3].fnPtr     = (void *)jni__create_ior;
  methods[4].name      = "_exec";
  methods[4].signature = "(Ljava/lang/String;Lsidl/rmi/Call;Lsidl/rmi/Return;)V";
  methods[4].fnPtr     = (void *)jni__exec;
  methods[5].name      = "_set_hooks";
  methods[5].signature = "(Z)V";
  methods[5].fnPtr     = (void *)jni__set_hooks;
  methods[6].name      = "_set_hooks_static";
  methods[6].signature = "(Z)V";
  methods[6].fnPtr     = (void *)jni__set_hooks_static;
  methods[7].name      = "Create";
  methods[7].signature = "(LbHYPRE/MPICommunicator;IIII)LbHYPRE/IJParCSRMatrix;";
  methods[7].fnPtr     = (void *)jni_Create;
  methods[8].name      = "GenerateLaplacian";
  methods[8].signature = "(LbHYPRE/MPICommunicator;IIIIIIIIILsidl/Double$Array1;I)LbHYPRE/IJParCSRMatrix;";
  methods[8].fnPtr     = (void *)jni_GenerateLaplacian;
  methods[9].name      = "SetDiagOffdSizes";
  methods[9].signature = "(Lsidl/Integer$Array1;Lsidl/Integer$Array1;)I";
  methods[9].fnPtr     = (void *)jni_SetDiagOffdSizes;
  methods[10].name      = "SetLocalRange";
  methods[10].signature = "(IIII)I";
  methods[10].fnPtr     = (void *)jni_SetLocalRange;
  methods[11].name      = "SetValues";
  methods[11].signature = "(Lsidl/Integer$Array1;Lsidl/Integer$Array1;Lsidl/Integer$Array1;Lsidl/Double$Array1;)I";
  methods[11].fnPtr     = (void *)jni_SetValues;
  methods[12].name      = "AddToValues";
  methods[12].signature = "(Lsidl/Integer$Array1;Lsidl/Integer$Array1;Lsidl/Integer$Array1;Lsidl/Double$Array1;)I";
  methods[12].fnPtr     = (void *)jni_AddToValues;
  methods[13].name      = "GetLocalRange";
  methods[13].signature = "(Lsidl/Integer$Holder;Lsidl/Integer$Holder;Lsidl/Integer$Holder;Lsidl/Integer$Holder;)I";
  methods[13].fnPtr     = (void *)jni_GetLocalRange;
  methods[14].name      = "GetRowCounts";
  methods[14].signature = "(Lsidl/Integer$Array1;Lsidl/Integer$Array1$Holder;)I";
  methods[14].fnPtr     = (void *)jni_GetRowCounts;
  methods[15].name      = "GetValues";
  methods[15].signature = "(Lsidl/Integer$Array1;Lsidl/Integer$Array1;Lsidl/Integer$Array1;Lsidl/Double$Array1$Holder;)I";
  methods[15].fnPtr     = (void *)jni_GetValues;
  methods[16].name      = "SetRowSizes";
  methods[16].signature = "(Lsidl/Integer$Array1;)I";
  methods[16].fnPtr     = (void *)jni_SetRowSizes;
  methods[17].name      = "Print";
  methods[17].signature = "(Ljava/lang/String;)I";
  methods[17].fnPtr     = (void *)jni_Print;
  methods[18].name      = "Read";
  methods[18].signature = "(Ljava/lang/String;LbHYPRE/MPICommunicator;)I";
  methods[18].fnPtr     = (void *)jni_Read;
  methods[19].name      = "SetCommunicator";
  methods[19].signature = "(LbHYPRE/MPICommunicator;)I";
  methods[19].fnPtr     = (void *)jni_SetCommunicator;
  methods[20].name      = "Destroy";
  methods[20].signature = "()V";
  methods[20].fnPtr     = (void *)jni_Destroy;
  methods[21].name      = "Initialize";
  methods[21].signature = "()I";
  methods[21].fnPtr     = (void *)jni_Initialize;
  methods[22].name      = "Assemble";
  methods[22].signature = "()I";
  methods[22].fnPtr     = (void *)jni_Assemble;
  methods[23].name      = "SetIntParameter";
  methods[23].signature = "(Ljava/lang/String;I)I";
  methods[23].fnPtr     = (void *)jni_SetIntParameter;
  methods[24].name      = "SetDoubleParameter";
  methods[24].signature = "(Ljava/lang/String;D)I";
  methods[24].fnPtr     = (void *)jni_SetDoubleParameter;
  methods[25].name      = "SetStringParameter";
  methods[25].signature = "(Ljava/lang/String;Ljava/lang/String;)I";
  methods[25].fnPtr     = (void *)jni_SetStringParameter;
  methods[26].name      = "SetIntArray1Parameter";
  methods[26].signature = "(Ljava/lang/String;Lsidl/Integer$Array1;)I";
  methods[26].fnPtr     = (void *)jni_SetIntArray1Parameter;
  methods[27].name      = "SetIntArray2Parameter";
  methods[27].signature = "(Ljava/lang/String;Lsidl/Integer$Array2;)I";
  methods[27].fnPtr     = (void *)jni_SetIntArray2Parameter;
  methods[28].name      = "SetDoubleArray1Parameter";
  methods[28].signature = "(Ljava/lang/String;Lsidl/Double$Array1;)I";
  methods[28].fnPtr     = (void *)jni_SetDoubleArray1Parameter;
  methods[29].name      = "SetDoubleArray2Parameter";
  methods[29].signature = "(Ljava/lang/String;Lsidl/Double$Array2;)I";
  methods[29].fnPtr     = (void *)jni_SetDoubleArray2Parameter;
  methods[30].name      = "GetIntValue";
  methods[30].signature = "(Ljava/lang/String;Lsidl/Integer$Holder;)I";
  methods[30].fnPtr     = (void *)jni_GetIntValue;
  methods[31].name      = "GetDoubleValue";
  methods[31].signature = "(Ljava/lang/String;Lsidl/Double$Holder;)I";
  methods[31].fnPtr     = (void *)jni_GetDoubleValue;
  methods[32].name      = "Setup";
  methods[32].signature = "(LbHYPRE/Vector;LbHYPRE/Vector;)I";
  methods[32].fnPtr     = (void *)jni_Setup;
  methods[33].name      = "Apply";
  methods[33].signature = "(LbHYPRE/Vector;LbHYPRE/Vector$Holder;)I";
  methods[33].fnPtr     = (void *)jni_Apply;
  methods[34].name      = "ApplyAdjoint";
  methods[34].signature = "(LbHYPRE/Vector;LbHYPRE/Vector$Holder;)I";
  methods[34].fnPtr     = (void *)jni_ApplyAdjoint;
  methods[35].name      = "GetRow";
  methods[35].signature = "(ILsidl/Integer$Holder;Lsidl/Integer$Array1$Holder;Lsidl/Double$Array1$Holder;)I";
  methods[35].fnPtr     = (void *)jni_GetRow;


  cls = (*env)->FindClass(env, "bHYPRE/IJParCSRMatrix");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 36);
    (*env)->DeleteLocalRef(env, cls);
  }
}
