/*
 * File:          bHYPRE_SStructParCSRVector_jniStub.c
 * Symbol:        bHYPRE.SStructParCSRVector-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side JNI glue code for bHYPRE.SStructParCSRVector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidl_Java.h"
#include "sidl_Loader.h"
#include "sidl_String.h"
#include "bHYPRE_SStructParCSRVector_IOR.h"
#include "babel_config.h"
/*
 * Includes for all method dependencies.
 */

#ifndef included_bHYPRE_MPICommunicator_jniStub_h
#include "bHYPRE_MPICommunicator_jniStub.h"
#endif
#ifndef included_bHYPRE_MatrixVectorView_jniStub_h
#include "bHYPRE_MatrixVectorView_jniStub.h"
#endif
#ifndef included_bHYPRE_ProblemDefinition_jniStub_h
#include "bHYPRE_ProblemDefinition_jniStub.h"
#endif
#ifndef included_bHYPRE_SStructGrid_jniStub_h
#include "bHYPRE_SStructGrid_jniStub.h"
#endif
#ifndef included_bHYPRE_SStructMatrixVectorView_jniStub_h
#include "bHYPRE_SStructMatrixVectorView_jniStub.h"
#endif
#ifndef included_bHYPRE_SStructParCSRVector_jniStub_h
#include "bHYPRE_SStructParCSRVector_jniStub.h"
#endif
#ifndef included_bHYPRE_SStructVectorView_jniStub_h
#include "bHYPRE_SStructVectorView_jniStub.h"
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
static struct sidl_recursive_mutex_t bHYPRE_SStructParCSRVector__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_SStructParCSRVector__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_SStructParCSRVector__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_SStructParCSRVector__mutex )==EDEADLOCK) */
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

static struct bHYPRE_SStructParCSRVector__epv 
  s_rem_epv__bhypre_sstructparcsrvector;

static struct bHYPRE_MatrixVectorView__epv s_rem_epv__bhypre_matrixvectorview;

static struct bHYPRE_ProblemDefinition__epv s_rem_epv__bhypre_problemdefinition;

static struct bHYPRE_SStructMatrixVectorView__epv 
  s_rem_epv__bhypre_sstructmatrixvectorview;

static struct bHYPRE_SStructVectorView__epv s_rem_epv__bhypre_sstructvectorview;

static struct bHYPRE_Vector__epv s_rem_epv__bhypre_vector;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_SStructParCSRVector__cast(
  struct bHYPRE_SStructParCSRVector__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1,
    cmp2,
    cmp3;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "bHYPRE.SStructVectorView");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = &((*self).d_bhypre_sstructvectorview);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.SStructMatrixVectorView");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_bhypre_sstructmatrixvectorview);
      return cast;
    }
    else if (cmp1 < 0) {
      cmp2 = strcmp(name, "bHYPRE.ProblemDefinition");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_problemdefinition);
        return cast;
      }
      else if (cmp2 < 0) {
        cmp3 = strcmp(name, "bHYPRE.MatrixVectorView");
        if (!cmp3) {
          (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
          cast = &((*self).d_bhypre_matrixvectorview);
          return cast;
        }
      }
    }
    else if (cmp1 > 0) {
      cmp2 = strcmp(name, "bHYPRE.SStructParCSRVector");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = ((struct bHYPRE_SStructParCSRVector__object*)self);
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
      cmp2 = strcmp(name, "bHYPRE.Vector");
      if (!cmp2) {
        (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
        cast = &((*self).d_bhypre_vector);
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_SStructParCSRVector__delete(
  struct bHYPRE_SStructParCSRVector__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_SStructParCSRVector__getURL(
  struct bHYPRE_SStructParCSRVector__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_SStructParCSRVector__raddRef(
  struct bHYPRE_SStructParCSRVector__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
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
remote_bHYPRE_SStructParCSRVector__isRemote(
    struct bHYPRE_SStructParCSRVector__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_SStructParCSRVector__set_hooks(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector._set_hooks.", &throwaway_exception);
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
static void remote_bHYPRE_SStructParCSRVector__exec(
  struct bHYPRE_SStructParCSRVector__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_SStructParCSRVector_addRef(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructParCSRVector__remote* r_obj = (struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_SStructParCSRVector_deleteRef(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_SStructParCSRVector__remote* r_obj = (struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data;
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
remote_bHYPRE_SStructParCSRVector_isSame(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.isSame.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_isType(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.isType.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_getClassInfo(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.getClassInfo.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetGrid */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetGrid(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ struct bHYPRE_SStructGrid__object* grid,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetGrid", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    if(grid){
      char* _url = sidl_BaseInterface__getURL((sidl_BaseInterface)grid, 
        _ex);SIDL_CHECK(*_ex);
      sidl_rmi_Invocation_packString( _inv, "grid", _url, _ex);SIDL_CHECK(*_ex);
      free((void*)_url);
    } else {
      sidl_rmi_Invocation_packString( _inv, "grid", NULL, _ex);SIDL_CHECK(*_ex);
    }

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetGrid.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_SetValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetBoxValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetBoxValues.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_AddToValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "AddToValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDouble( _inv, "value", value, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.AddToValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:AddToBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_AddToBoxValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* in rarray[nvalues] */ struct sidl_double__array* values,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "AddToBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.AddToBoxValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Gather */
static int32_t
remote_bHYPRE_SStructParCSRVector_Gather(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Gather", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Gather.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_GetValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* index,
  /* in */ int32_t var,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "index", index,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.GetValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetBoxValues */
static int32_t
remote_bHYPRE_SStructParCSRVector_GetBoxValues(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ int32_t part,
  /* in rarray[dim] */ struct sidl_int__array* ilower,
  /* in rarray[dim] */ struct sidl_int__array* iupper,
  /* in */ int32_t var,
  /* inout rarray[nvalues] */ struct sidl_double__array** values,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetBoxValues", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packInt( _inv, "part", part, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "ilower", ilower,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packIntArray( _inv, "iupper", iupper,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packInt( _inv, "var", var, _ex);SIDL_CHECK(*_ex);
    sidl_rmi_Invocation_packDoubleArray( _inv, "values", *values,
      sidl_column_major_order,1,0, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.GetBoxValues.", &throwaway_exception);
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

/* REMOTE METHOD STUB:SetComplex */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetComplex(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "SetComplex", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetComplex.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_Print(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ const char* filename,
  /* in */ int32_t all,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Print", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "filename", filename, _ex);SIDL_CHECK(
      *_ex);
    sidl_rmi_Invocation_packInt( _inv, "all", all, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Print.", &throwaway_exception);
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

/* REMOTE METHOD STUB:GetObject */
static int32_t
remote_bHYPRE_SStructParCSRVector_GetObject(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct sidl_BaseInterface__object** A,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    /* initialize a new invocation */
    sidl_BaseInterface _throwaway = NULL;
    sidl_BaseException _be = NULL;
    sidl_rmi_Response _rsvp = NULL;
    char* A_str= NULL;
    int32_t _retval = 0;
    struct sidl_rmi_InstanceHandle__object * _conn = ((struct 
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "GetObject", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.GetObject.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackString( _rsvp, "A", &A_str, _ex);SIDL_CHECK(*_ex);
    *A = sidl_BaseInterface__connectI(A_str, FALSE, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:SetCommunicator */
static int32_t
remote_bHYPRE_SStructParCSRVector_SetCommunicator(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.SetCommunicator.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_Destroy(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Destroy.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_Initialize(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Initialize", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Initialize.", &throwaway_exception);
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
remote_bHYPRE_SStructParCSRVector_Assemble(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Assemble", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Assemble.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Clear */
static int32_t
remote_bHYPRE_SStructParCSRVector_Clear(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Clear", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Clear.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Copy */
static int32_t
remote_bHYPRE_SStructParCSRVector_Copy(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Copy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Copy.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Clone */
static int32_t
remote_bHYPRE_SStructParCSRVector_Clone(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* out */ struct bHYPRE_Vector__object** x,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Clone", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Clone.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Scale */
static int32_t
remote_bHYPRE_SStructParCSRVector_Scale(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ double a,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Scale", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "a", a, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Scale.", &throwaway_exception);
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

/* REMOTE METHOD STUB:Dot */
static int32_t
remote_bHYPRE_SStructParCSRVector_Dot(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ struct bHYPRE_Vector__object* x,
  /* out */ double* d,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Dot", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Dot.", &throwaway_exception);
      *_ex = (sidl_BaseInterface) sidl_BaseInterface__rmicast(_be,
        &throwaway_exception);
      goto EXIT;
    }

    /* extract return value */
    sidl_rmi_Response_unpackInt( _rsvp, "_retval", &_retval, _ex);SIDL_CHECK(
      *_ex);

    /* unpack out and inout arguments */
    sidl_rmi_Response_unpackDouble( _rsvp, "d", d, _ex);SIDL_CHECK(*_ex);

    /* cleanup and return */
    EXIT:
    if(_inv) { sidl_rmi_Invocation_deleteRef(_inv, &_throwaway); }
    if(_rsvp) { sidl_rmi_Response_deleteRef(_rsvp, &_throwaway); }
    return _retval;
  }
}

/* REMOTE METHOD STUB:Axpy */
static int32_t
remote_bHYPRE_SStructParCSRVector_Axpy(
  /* in */ struct bHYPRE_SStructParCSRVector__object* self ,
  /* in */ double a,
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
      bHYPRE_SStructParCSRVector__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Axpy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packDouble( _inv, "a", a, _ex);SIDL_CHECK(*_ex);
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.SStructParCSRVector.Axpy.", &throwaway_exception);
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

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_SStructParCSRVector__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_SStructParCSRVector__epv*     epv = 
    &s_rem_epv__bhypre_sstructparcsrvector;
  struct bHYPRE_MatrixVectorView__epv*        e0  = 
    &s_rem_epv__bhypre_matrixvectorview;
  struct bHYPRE_ProblemDefinition__epv*       e1  = 
    &s_rem_epv__bhypre_problemdefinition;
  struct bHYPRE_SStructMatrixVectorView__epv* e2  = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  struct bHYPRE_SStructVectorView__epv*       e3  = 
    &s_rem_epv__bhypre_sstructvectorview;
  struct bHYPRE_Vector__epv*                  e4  = &s_rem_epv__bhypre_vector;
  struct sidl_BaseClass__epv*                 e5  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*             e6  = 
    &s_rem_epv__sidl_baseinterface;

  epv->f__cast                = remote_bHYPRE_SStructParCSRVector__cast;
  epv->f__delete              = remote_bHYPRE_SStructParCSRVector__delete;
  epv->f__exec                = remote_bHYPRE_SStructParCSRVector__exec;
  epv->f__getURL              = remote_bHYPRE_SStructParCSRVector__getURL;
  epv->f__raddRef             = remote_bHYPRE_SStructParCSRVector__raddRef;
  epv->f__isRemote            = remote_bHYPRE_SStructParCSRVector__isRemote;
  epv->f__set_hooks           = remote_bHYPRE_SStructParCSRVector__set_hooks;
  epv->f__ctor                = NULL;
  epv->f__ctor2               = NULL;
  epv->f__dtor                = NULL;
  epv->f_addRef               = remote_bHYPRE_SStructParCSRVector_addRef;
  epv->f_deleteRef            = remote_bHYPRE_SStructParCSRVector_deleteRef;
  epv->f_isSame               = remote_bHYPRE_SStructParCSRVector_isSame;
  epv->f_isType               = remote_bHYPRE_SStructParCSRVector_isType;
  epv->f_getClassInfo         = remote_bHYPRE_SStructParCSRVector_getClassInfo;
  epv->f_SetGrid              = remote_bHYPRE_SStructParCSRVector_SetGrid;
  epv->f_SetValues            = remote_bHYPRE_SStructParCSRVector_SetValues;
  epv->f_SetBoxValues         = remote_bHYPRE_SStructParCSRVector_SetBoxValues;
  epv->f_AddToValues          = remote_bHYPRE_SStructParCSRVector_AddToValues;
  epv->f_AddToBoxValues       = 
    remote_bHYPRE_SStructParCSRVector_AddToBoxValues;
  epv->f_Gather               = remote_bHYPRE_SStructParCSRVector_Gather;
  epv->f_GetValues            = remote_bHYPRE_SStructParCSRVector_GetValues;
  epv->f_GetBoxValues         = remote_bHYPRE_SStructParCSRVector_GetBoxValues;
  epv->f_SetComplex           = remote_bHYPRE_SStructParCSRVector_SetComplex;
  epv->f_Print                = remote_bHYPRE_SStructParCSRVector_Print;
  epv->f_GetObject            = remote_bHYPRE_SStructParCSRVector_GetObject;
  epv->f_SetCommunicator      = 
    remote_bHYPRE_SStructParCSRVector_SetCommunicator;
  epv->f_Destroy              = remote_bHYPRE_SStructParCSRVector_Destroy;
  epv->f_Initialize           = remote_bHYPRE_SStructParCSRVector_Initialize;
  epv->f_Assemble             = remote_bHYPRE_SStructParCSRVector_Assemble;
  epv->f_Clear                = remote_bHYPRE_SStructParCSRVector_Clear;
  epv->f_Copy                 = remote_bHYPRE_SStructParCSRVector_Copy;
  epv->f_Clone                = remote_bHYPRE_SStructParCSRVector_Clone;
  epv->f_Scale                = remote_bHYPRE_SStructParCSRVector_Scale;
  epv->f_Dot                  = remote_bHYPRE_SStructParCSRVector_Dot;
  epv->f_Axpy                 = remote_bHYPRE_SStructParCSRVector_Axpy;

  e0->f__cast           = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
    epv->f__cast;
  e0->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e0->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e0->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e0->f__isRemote       = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
    epv->f__isRemote;
  e0->f__set_hooks      = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
    epv->f__set_hooks;
  e0->f__exec           = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_SetCommunicator = (int32_t (*)(void*,struct 
    bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
    epv->f_SetCommunicator;
  e0->f_Destroy         = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Destroy;
  e0->f_Initialize      = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Initialize;
  e0->f_Assemble        = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Assemble;
  e0->f_addRef          = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_addRef;
  e0->f_deleteRef       = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_deleteRef;
  e0->f_isSame          = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e0->f_isType          = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,struct 
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
  e2->f_GetObject       = (int32_t (*)(void*,struct 
    sidl_BaseInterface__object**,struct sidl_BaseInterface__object **)) 
    epv->f_GetObject;
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

  e3->f__cast           = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
    epv->f__cast;
  e3->f__delete         = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e3->f__getURL         = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e3->f__raddRef        = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e3->f__isRemote       = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
    epv->f__isRemote;
  e3->f__set_hooks      = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
    epv->f__set_hooks;
  e3->f__exec           = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e3->f_SetGrid         = (int32_t (*)(void*,struct bHYPRE_SStructGrid__object*,
    struct sidl_BaseInterface__object **)) epv->f_SetGrid;
  e3->f_SetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,double,struct sidl_BaseInterface__object **)) epv->f_SetValues;
  e3->f_SetBoxValues    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_double__array*,struct 
    sidl_BaseInterface__object **)) epv->f_SetBoxValues;
  e3->f_AddToValues     = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,double,struct sidl_BaseInterface__object **)) epv->f_AddToValues;
  e3->f_AddToBoxValues  = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_double__array*,struct 
    sidl_BaseInterface__object **)) epv->f_AddToBoxValues;
  e3->f_Gather          = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Gather;
  e3->f_GetValues       = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    int32_t,double*,struct sidl_BaseInterface__object **)) epv->f_GetValues;
  e3->f_GetBoxValues    = (int32_t (*)(void*,int32_t,struct sidl_int__array*,
    struct sidl_int__array*,int32_t,struct sidl_double__array**,struct 
    sidl_BaseInterface__object **)) epv->f_GetBoxValues;
  e3->f_SetComplex      = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_SetComplex;
  e3->f_Print           = (int32_t (*)(void*,const char*,int32_t,struct 
    sidl_BaseInterface__object **)) epv->f_Print;
  e3->f_GetObject       = (int32_t (*)(void*,struct 
    sidl_BaseInterface__object**,struct sidl_BaseInterface__object **)) 
    epv->f_GetObject;
  e3->f_SetCommunicator = (int32_t (*)(void*,struct 
    bHYPRE_MPICommunicator__object*,struct sidl_BaseInterface__object **)) 
    epv->f_SetCommunicator;
  e3->f_Destroy         = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Destroy;
  e3->f_Initialize      = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Initialize;
  e3->f_Assemble        = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Assemble;
  e3->f_addRef          = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_addRef;
  e3->f_deleteRef       = (void (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_deleteRef;
  e3->f_isSame          = (sidl_bool (*)(void*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e3->f_isType          = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e3->f_getClassInfo    = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  e4->f__cast        = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
    epv->f__cast;
  e4->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e4->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e4->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e4->f__isRemote    = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
    epv->f__isRemote;
  e4->f__set_hooks   = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
    epv->f__set_hooks;
  e4->f__exec        = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e4->f_Clear        = (int32_t (*)(void*,struct sidl_BaseInterface__object 
    **)) epv->f_Clear;
  e4->f_Copy         = (int32_t (*)(void*,struct bHYPRE_Vector__object*,struct 
    sidl_BaseInterface__object **)) epv->f_Copy;
  e4->f_Clone        = (int32_t (*)(void*,struct bHYPRE_Vector__object**,struct 
    sidl_BaseInterface__object **)) epv->f_Clone;
  e4->f_Scale        = (int32_t (*)(void*,double,struct 
    sidl_BaseInterface__object **)) epv->f_Scale;
  e4->f_Dot          = (int32_t (*)(void*,struct bHYPRE_Vector__object*,double*,
    struct sidl_BaseInterface__object **)) epv->f_Dot;
  e4->f_Axpy         = (int32_t (*)(void*,double,struct bHYPRE_Vector__object*,
    struct sidl_BaseInterface__object **)) epv->f_Axpy;
  e4->f_addRef       = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_addRef;
  e4->f_deleteRef    = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_deleteRef;
  e4->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e4->f_isType       = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e4->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,struct 
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
static struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__remoteConnect(const char *url, sidl_bool ar, 
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRVector__object* self;

  struct bHYPRE_SStructParCSRVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructParCSRVector__remote* r_obj;
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
    return bHYPRE_SStructParCSRVector__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex ); 
    SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  r_obj =
    (struct bHYPRE_SStructParCSRVector__remote*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

  s0->d_bhypre_sstructvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructvectorview;
  s0->d_bhypre_sstructvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrvector;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__IHConnect(sidl_rmi_InstanceHandle instance, 
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_SStructParCSRVector__object* self;

  struct bHYPRE_SStructParCSRVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructParCSRVector__remote* r_obj;
  self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  r_obj =
    (struct bHYPRE_SStructParCSRVector__remote*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

  s0->d_bhypre_sstructvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructvectorview;
  s0->d_bhypre_sstructvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrvector;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__remoteCreate(const char *url, sidl_BaseInterface 
  *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_SStructParCSRVector__object* self;

  struct bHYPRE_SStructParCSRVector__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_SStructParCSRVector__remote* r_obj;
  sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
    url, "bHYPRE.SStructParCSRVector", _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_SStructParCSRVector__object*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__object));

  r_obj =
    (struct bHYPRE_SStructParCSRVector__remote*) malloc(
      sizeof(struct bHYPRE_SStructParCSRVector__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                      self;
  s1 =                                      &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_SStructParCSRVector__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_bhypre_matrixvectorview.d_epv    = &s_rem_epv__bhypre_matrixvectorview;
  s0->d_bhypre_matrixvectorview.d_object = (void*) self;

  s0->d_bhypre_problemdefinition.d_epv    = 
    &s_rem_epv__bhypre_problemdefinition;
  s0->d_bhypre_problemdefinition.d_object = (void*) self;

  s0->d_bhypre_sstructmatrixvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructmatrixvectorview;
  s0->d_bhypre_sstructmatrixvectorview.d_object = (void*) self;

  s0->d_bhypre_sstructvectorview.d_epv    = 
    &s_rem_epv__bhypre_sstructvectorview;
  s0->d_bhypre_sstructvectorview.d_object = (void*) self;

  s0->d_bhypre_vector.d_epv    = &s_rem_epv__bhypre_vector;
  s0->d_bhypre_vector.d_object = (void*) self;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_sstructparcsrvector;

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

struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_SStructParCSRVector__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.SStructParCSRVector", (
      void*)bHYPRE_SStructParCSRVector__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_SStructParCSRVector__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.SStructParCSRVector", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_SStructParCSRVector__object*
bHYPRE_SStructParCSRVector__connectI(const char* url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex)
{
  return bHYPRE_SStructParCSRVector__remoteConnect(url, ar, _ex);
}

/*
 * Function to extract IOR reference from the Java object.
 */

static struct bHYPRE_SStructParCSRVector__object* _get_ior(
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
  return (struct bHYPRE_SStructParCSRVector__object*) ptr;
}

/*
 * External reference to IOR methods.
 */

static const struct bHYPRE_SStructParCSRVector__external* s_external = NULL;

/*
 * External reference to static EPV.
 */

static const struct bHYPRE_SStructParCSRVector__sepv* s_sepv = NULL;

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
  _res = POINTER_TO_JLONG(bHYPRE_SStructParCSRVector__remoteCreate(_tmp_url, 
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
  struct bHYPRE_SStructParCSRVector__object *_ptr = (*s_external->createObject)(
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
  _res = POINTER_TO_JLONG(bHYPRE_SStructParCSRVector__remoteConnect(_tmp_url, 1,
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
 *  This function is the preferred way to create a SStruct ParCSR Vector. 
 */

static jobject
jni_Create(
  JNIEnv* env,
  jclass  cls,
  jobject _arg_mpi_comm,
  jobject _arg_grid)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_MPICommunicator__object* _tmp_mpi_comm = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  struct bHYPRE_SStructGrid__object* _tmp_grid = (struct 
    bHYPRE_SStructGrid__object*) NULL;
  struct bHYPRE_SStructParCSRVector__object* _ior_res = (struct 
    bHYPRE_SStructParCSRVector__object*) NULL;
  jobject _res = (jobject) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _tmp_mpi_comm = (struct bHYPRE_MPICommunicator__object*) sidl_Java_J2I_cls(
    env, _arg_mpi_comm, FALSE);JAVA_CHECK(env);
  _tmp_grid = (struct bHYPRE_SStructGrid__object*) sidl_Java_J2I_cls(env, 
    _arg_grid, FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(s_sepv->f_Create))(
    _tmp_mpi_comm,
    _tmp_grid,
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
  _res = sidl_Java_I2J_cls(env, _ior_res, "bHYPRE.SStructParCSRVector", 
    FALSE);JAVA_CHECK(env);

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * Set the vector grid.
 */

static jint
jni_SetGrid(
  JNIEnv* env,
  jobject obj,
  jobject _arg_grid)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  struct bHYPRE_SStructGrid__object* _tmp_grid = (struct 
    bHYPRE_SStructGrid__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_grid = (struct bHYPRE_SStructGrid__object*) sidl_Java_J2I_cls(env, 
    _arg_grid, FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetGrid))(
    _ior,
    _tmp_grid,
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
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

static jint
jni_SetValues(
  JNIEnv* env,
  jobject obj,
  jint _arg_part,
  jobject _arg_index,
  jint _arg_var,
  jdouble _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  int32_t _tmp_part = 0;
  struct sidl_int__array* _tmp_index = (struct sidl_int__array*) NULL;
  int32_t _tmp_var = 0;
  double _tmp_value = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_part = (int32_t) _arg_part;
  _tmp_index = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_index);
  _tmp_var = (int32_t) _arg_var;
  _tmp_value = (double) _arg_value;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetValues))(
    _ior,
    _tmp_part,
    _tmp_index,
    _tmp_var,
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
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

static jint
jni_SetBoxValues(
  JNIEnv* env,
  jobject obj,
  jint _arg_part,
  jobject _arg_ilower,
  jobject _arg_iupper,
  jint _arg_var,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  int32_t _tmp_part = 0;
  struct sidl_int__array* _tmp_ilower = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_iupper = (struct sidl_int__array*) NULL;
  int32_t _tmp_var = 0;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_part = (int32_t) _arg_part;
  _tmp_ilower = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_ilower);
  _tmp_iupper = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_iupper);
  _tmp_var = (int32_t) _arg_var;
  _tmp_values = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_values);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_SetBoxValues))(
    _ior,
    _tmp_part,
    _tmp_ilower,
    _tmp_iupper,
    _tmp_var,
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
 * Set vector coefficients index by index.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

static jint
jni_AddToValues(
  JNIEnv* env,
  jobject obj,
  jint _arg_part,
  jobject _arg_index,
  jint _arg_var,
  jdouble _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  int32_t _tmp_part = 0;
  struct sidl_int__array* _tmp_index = (struct sidl_int__array*) NULL;
  int32_t _tmp_var = 0;
  double _tmp_value = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_part = (int32_t) _arg_part;
  _tmp_index = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_index);
  _tmp_var = (int32_t) _arg_var;
  _tmp_value = (double) _arg_value;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_AddToValues))(
    _ior,
    _tmp_part,
    _tmp_index,
    _tmp_var,
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
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Set vector coefficients a box at a time.
 * 
 * NOTE: Users are required to set values on all processes that
 * own the associated variables.  This means that some data will
 * be multiply defined.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

static jint
jni_AddToBoxValues(
  JNIEnv* env,
  jobject obj,
  jint _arg_part,
  jobject _arg_ilower,
  jobject _arg_iupper,
  jint _arg_var,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  int32_t _tmp_part = 0;
  struct sidl_int__array* _tmp_ilower = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_iupper = (struct sidl_int__array*) NULL;
  int32_t _tmp_var = 0;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_part = (int32_t) _arg_part;
  _tmp_ilower = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_ilower);
  _tmp_iupper = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_iupper);
  _tmp_var = (int32_t) _arg_var;
  _tmp_values = (struct sidl_double__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_values);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_AddToBoxValues))(
    _ior,
    _tmp_part,
    _tmp_ilower,
    _tmp_iupper,
    _tmp_var,
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
 * Gather vector data before calling {\tt GetValues}.
 */

static jint
jni_Gather(
  JNIEnv* env,
  jobject obj)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  _ior_res = (*(_ior->d_epv->f_Gather))(
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
 * Get vector coefficients index by index.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt value} consists of a pair
 * of doubles representing the real and imaginary parts of the
 * complex value.
 */

static jint
jni_GetValues(
  JNIEnv* env,
  jobject obj,
  jint _arg_part,
  jobject _arg_index,
  jint _arg_var,
  jobject _arg_value)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  int32_t _tmp_part = 0;
  struct sidl_int__array* _tmp_index = (struct sidl_int__array*) NULL;
  int32_t _tmp_var = 0;
  double _tmp_value = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_part = (int32_t) _arg_part;
  _tmp_index = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_index);
  _tmp_var = (int32_t) _arg_var;
  if(_arg_value== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetValues))(
    _ior,
    _tmp_part,
    _tmp_index,
    _tmp_var,
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
  sidl_Java_I2J_double_holder(env, _arg_value, _tmp_value);
  _res = (jint) _ior_res;

  return _res;
}

/*
 * Get vector coefficients a box at a time.
 * 
 * NOTE: Users may only get values on processes that own the
 * associated variables.
 * 
 * If the vector is complex, then {\tt values} consists of pairs
 * of doubles representing the real and imaginary parts of each
 * complex value.
 */

static jint
jni_GetBoxValues(
  JNIEnv* env,
  jobject obj,
  jint _arg_part,
  jobject _arg_ilower,
  jobject _arg_iupper,
  jint _arg_var,
  jobject _arg_values)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  int32_t _tmp_part = 0;
  struct sidl_int__array* _tmp_ilower = (struct sidl_int__array*) NULL;
  struct sidl_int__array* _tmp_iupper = (struct sidl_int__array*) NULL;
  int32_t _tmp_var = 0;
  struct sidl_double__array* _tmp_values = (struct sidl_double__array*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_part = (int32_t) _arg_part;
  _tmp_ilower = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_ilower);
  _tmp_iupper = (struct sidl_int__array*) sidl_Java_J2I_borrow_array(env, 
    _arg_iupper);
  _tmp_var = (int32_t) _arg_var;
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

  _ior_res = (*(_ior->d_epv->f_GetBoxValues))(
    _ior,
    _tmp_part,
    _tmp_ilower,
    _tmp_iupper,
    _tmp_var,
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
 * Set the vector to be complex.
 */

static jint
jni_SetComplex(
  JNIEnv* env,
  jobject obj)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  _ior_res = (*(_ior->d_epv->f_SetComplex))(
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
 * Print the vector to file.  This is mainly for debugging
 * purposes.
 */

static jint
jni_Print(
  JNIEnv* env,
  jobject obj,
  jstring _arg_filename,
  jint _arg_all)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  char* _tmp_filename = (char*) NULL;
  int32_t _tmp_all = 0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_filename = sidl_Java_J2I_string(env, _arg_filename);
  _tmp_all = (int32_t) _arg_all;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Print))(
    _ior,
    _tmp_filename,
    _tmp_all,
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
 * A semi-structured matrix or vector contains a Struct or IJ matrix
 * or vector.  GetObject returns it.
 * The returned type is a sidl.BaseInterface.
 * A cast must be used on the returned object to convert it into a known type.
 */

static jint
jni_GetObject(
  JNIEnv* env,
  jobject obj,
  jobject _arg_A)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  struct sidl_BaseInterface__object* _tmp_A = (struct 
    sidl_BaseInterface__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  if(_arg_A== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_GetObject))(
    _ior,
    &_tmp_A,
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
  sidl_Java_I2J_ifc_holder(env, _arg_A, _tmp_A, "sidl.BaseInterface", 
    FALSE);JAVA_CHECK(env);
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

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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
 * Set {\tt self} to 0.
 */

static jint
jni_Clear(
  JNIEnv* env,
  jobject obj)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  _ior_res = (*(_ior->d_epv->f_Clear))(
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
 * Copy data from x into {\tt self}.
 */

static jint
jni_Copy(
  JNIEnv* env,
  jobject obj,
  jobject _arg_x)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_x = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_x, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Copy))(
    _ior,
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
 * Create an {\tt x} compatible with {\tt self}.
 * The new vector's data is not specified.
 * 
 * NOTE: When this method is used in an inherited class, the
 * cloned {\tt Vector} object can be cast to an object with the
 * inherited class type.
 */

static jint
jni_Clone(
  JNIEnv* env,
  jobject obj,
  jobject _arg_x)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  if(_arg_x== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Clone))(
    _ior,
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
 * Scale {\tt self} by {\tt a}.
 */

static jint
jni_Scale(
  JNIEnv* env,
  jobject obj,
  jdouble _arg_a)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  double _tmp_a = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_a = (double) _arg_a;

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Scale))(
    _ior,
    _tmp_a,
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
 * Compute {\tt d}, the inner-product of {\tt self} and {\tt x}.
 */

static jint
jni_Dot(
  JNIEnv* env,
  jobject obj,
  jobject _arg_x,
  jobject _arg_d)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  double _tmp_d = 0.0;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_x = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_x, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);
  if(_arg_d== NULL) {
    jclass newExcCls = (*env)->FindClass(env, "java/lang/RuntimeException");
    (*env)->ThrowNew(env, newExcCls, "Null Holder Sent as OUT Argument");
    return 0;
    }

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Dot))(
    _ior,
    _tmp_x,
    &_tmp_d,
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
  sidl_Java_I2J_double_holder(env, _arg_d, _tmp_d);
  _res = (jint) _ior_res;

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 * Add {\tt a}{\tt x} to {\tt self}.
 */

static jint
jni_Axpy(
  JNIEnv* env,
  jobject obj,
  jdouble _arg_a,
  jobject _arg_x)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
  double _tmp_a = 0.0;
  struct bHYPRE_Vector__object* _tmp_x = (struct bHYPRE_Vector__object*) NULL;
  int32_t _ior_res = 0;
  jint _res = 0;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _ior = _get_ior(env, obj);
  _tmp_a = (double) _arg_a;
  _tmp_x = (struct bHYPRE_Vector__object*) sidl_Java_J2I_ifc(env, _arg_x, 
    "bHYPRE.Vector", FALSE);JAVA_CHECK(env);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(_ior->d_epv->f_Axpy))(
    _ior,
    _tmp_a,
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

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

  struct bHYPRE_SStructParCSRVector__object* _ior = NULL;
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

void bHYPRE_SStructParCSRVector__register(JNIEnv* env)
{
  JNINativeMethod methods[29];
  jclass cls;

#ifdef SIDL_STATIC_LIBRARY
  s_external = bHYPRE_SStructParCSRVector__externals();
#else
  s_external = (struct 
    bHYPRE_SStructParCSRVector__external*)sidl_dynamicLoadIOR(
    "bHYPRE.SStructParCSRVector","bHYPRE_SStructParCSRVector__externals") ;
  sidl_checkIORVersion("bHYPRE.SStructParCSRVector", 
    s_external->d_ior_major_version, s_external->d_ior_minor_version, 1, 0);
#endif

  s_sepv = (*(s_external->getStaticEPV))();

  methods[0].name      = "_connect_remote_ior";
  methods[0].signature = "(Ljava/lang/String;)J";
  methods[0].fnPtr     = (void *)jni__connect_remote_ior;
  methods[1].name      = "_create_remote_ior";
  methods[1].signature = "(Ljava/lang/String;)J";
  methods[1].fnPtr     = (void *)jni__create_remote_ior;
  methods[2].name      = "_wrap";
  methods[2].signature = "(LbHYPRE/SStructParCSRVector;)J";
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
  methods[7].signature = "(LbHYPRE/MPICommunicator;LbHYPRE/SStructGrid;)LbHYPRE/SStructParCSRVector;";
  methods[7].fnPtr     = (void *)jni_Create;
  methods[8].name      = "SetGrid";
  methods[8].signature = "(LbHYPRE/SStructGrid;)I";
  methods[8].fnPtr     = (void *)jni_SetGrid;
  methods[9].name      = "SetValues";
  methods[9].signature = "(ILsidl/Integer$Array1;ID)I";
  methods[9].fnPtr     = (void *)jni_SetValues;
  methods[10].name      = "SetBoxValues";
  methods[10].signature = "(ILsidl/Integer$Array1;Lsidl/Integer$Array1;ILsidl/Double$Array1;)I";
  methods[10].fnPtr     = (void *)jni_SetBoxValues;
  methods[11].name      = "AddToValues";
  methods[11].signature = "(ILsidl/Integer$Array1;ID)I";
  methods[11].fnPtr     = (void *)jni_AddToValues;
  methods[12].name      = "AddToBoxValues";
  methods[12].signature = "(ILsidl/Integer$Array1;Lsidl/Integer$Array1;ILsidl/Double$Array1;)I";
  methods[12].fnPtr     = (void *)jni_AddToBoxValues;
  methods[13].name      = "Gather";
  methods[13].signature = "()I";
  methods[13].fnPtr     = (void *)jni_Gather;
  methods[14].name      = "GetValues";
  methods[14].signature = "(ILsidl/Integer$Array1;ILsidl/Double$Holder;)I";
  methods[14].fnPtr     = (void *)jni_GetValues;
  methods[15].name      = "GetBoxValues";
  methods[15].signature = "(ILsidl/Integer$Array1;Lsidl/Integer$Array1;ILsidl/Double$Array1$Holder;)I";
  methods[15].fnPtr     = (void *)jni_GetBoxValues;
  methods[16].name      = "SetComplex";
  methods[16].signature = "()I";
  methods[16].fnPtr     = (void *)jni_SetComplex;
  methods[17].name      = "Print";
  methods[17].signature = "(Ljava/lang/String;I)I";
  methods[17].fnPtr     = (void *)jni_Print;
  methods[18].name      = "GetObject";
  methods[18].signature = "(Lsidl/BaseInterface$Holder;)I";
  methods[18].fnPtr     = (void *)jni_GetObject;
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
  methods[23].name      = "Clear";
  methods[23].signature = "()I";
  methods[23].fnPtr     = (void *)jni_Clear;
  methods[24].name      = "Copy";
  methods[24].signature = "(LbHYPRE/Vector;)I";
  methods[24].fnPtr     = (void *)jni_Copy;
  methods[25].name      = "Clone";
  methods[25].signature = "(LbHYPRE/Vector$Holder;)I";
  methods[25].fnPtr     = (void *)jni_Clone;
  methods[26].name      = "Scale";
  methods[26].signature = "(D)I";
  methods[26].fnPtr     = (void *)jni_Scale;
  methods[27].name      = "Dot";
  methods[27].signature = "(LbHYPRE/Vector;Lsidl/Double$Holder;)I";
  methods[27].fnPtr     = (void *)jni_Dot;
  methods[28].name      = "Axpy";
  methods[28].signature = "(DLbHYPRE/Vector;)I";
  methods[28].fnPtr     = (void *)jni_Axpy;


  cls = (*env)->FindClass(env, "bHYPRE/SStructParCSRVector");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 29);
    (*env)->DeleteLocalRef(env, cls);
  }
}
