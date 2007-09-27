/*
 * File:          bHYPRE_MPICommunicator_jniStub.c
 * Symbol:        bHYPRE.MPICommunicator-v1.0.0
 * Symbol Type:   class
 * Babel Version: 1.0.4
 * Description:   Client-side JNI glue code for bHYPRE.MPICommunicator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 */

#include "sidl_Java.h"
#include "sidl_Loader.h"
#include "sidl_String.h"
#include "bHYPRE_MPICommunicator_IOR.h"
#include "babel_config.h"
/*
 * Includes for all method dependencies.
 */

#ifndef included_bHYPRE_MPICommunicator_jniStub_h
#include "bHYPRE_MPICommunicator_jniStub.h"
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
static struct sidl_recursive_mutex_t bHYPRE_MPICommunicator__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &bHYPRE_MPICommunicator__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &bHYPRE_MPICommunicator__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &bHYPRE_MPICommunicator__mutex )==EDEADLOCK) */
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

static struct bHYPRE_MPICommunicator__epv s_rem_epv__bhypre_mpicommunicator;

static struct sidl_BaseClass__epv  s_rem_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_rem_epv__sidl_baseinterface;


/* REMOTE CAST: dynamic type casting for remote objects. */
static void* remote_bHYPRE_MPICommunicator__cast(
  struct bHYPRE_MPICommunicator__object* self,
  const char* name, sidl_BaseInterface* _ex)
{
  int
    cmp0,
    cmp1;
  void* cast = NULL;
  *_ex = NULL; /* default to no exception */
  cmp0 = strcmp(name, "sidl.BaseClass");
  if (!cmp0) {
    (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
    cast = ((struct sidl_BaseClass__object*)self);
    return cast;
  }
  else if (cmp0 < 0) {
    cmp1 = strcmp(name, "bHYPRE.MPICommunicator");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = ((struct bHYPRE_MPICommunicator__object*)self);
      return cast;
    }
  }
  else if (cmp0 > 0) {
    cmp1 = strcmp(name, "sidl.BaseInterface");
    if (!cmp1) {
      (*self->d_epv->f_addRef)(self, _ex); SIDL_CHECK(*_ex);
      cast = &((*self).d_sidl_baseclass.d_sidl_baseinterface);
      return cast;
    }
  }
  if ((*self->d_epv->f_isType)(self,name, _ex)) {
    void* (*func)(struct sidl_rmi_InstanceHandle__object*, struct 
      sidl_BaseInterface__object**) = 
      (void* (*)(struct sidl_rmi_InstanceHandle__object*, struct 
        sidl_BaseInterface__object**)) 
      sidl_rmi_ConnectRegistry_getConnect(name, _ex);SIDL_CHECK(*_ex);
    cast =  (*func)(((struct 
      bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih, _ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/* REMOTE DELETE: call the remote destructor for the object. */
static void remote_bHYPRE_MPICommunicator__delete(
  struct bHYPRE_MPICommunicator__object* self,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
  free((void*) self);
}

/* REMOTE GETURL: call the getURL function for the object. */
static char* remote_bHYPRE_MPICommunicator__getURL(
  struct bHYPRE_MPICommunicator__object* self, sidl_BaseInterface* _ex)
{
  struct sidl_rmi_InstanceHandle__object *conn = ((struct 
    bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
  *_ex = NULL;
  if(conn != NULL) {
    return sidl_rmi_InstanceHandle_getObjectURL(conn, _ex);
  }
  return NULL;
}

/* REMOTE ADDREF: For internal babel use only! Remote addRef. */
static void remote_bHYPRE_MPICommunicator__raddRef(
  struct bHYPRE_MPICommunicator__object* self,sidl_BaseInterface* _ex)
{
  sidl_BaseException netex = NULL;
  /* initialize a new invocation */
  sidl_BaseInterface _throwaway = NULL;
  struct sidl_rmi_InstanceHandle__object *_conn = ((struct 
    bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
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
remote_bHYPRE_MPICommunicator__isRemote(
    struct bHYPRE_MPICommunicator__object* self, 
    sidl_BaseInterface *_ex) {
  *_ex = NULL;
  return TRUE;
}

/* REMOTE METHOD STUB:_set_hooks */
static void
remote_bHYPRE_MPICommunicator__set_hooks(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
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
      bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "_set_hooks", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packBool( _inv, "on", on, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.MPICommunicator._set_hooks.", &throwaway_exception);
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
static void remote_bHYPRE_MPICommunicator__exec(
  struct bHYPRE_MPICommunicator__object* self,const char* methodName,
  sidl_rmi_Call inArgs,
  sidl_rmi_Return outArgs,
  sidl_BaseInterface* _ex)
{
  *_ex = NULL;
}

/* REMOTE METHOD STUB:Destroy */
static void
remote_bHYPRE_MPICommunicator_Destroy(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
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
      bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "Destroy", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.MPICommunicator.Destroy.", &throwaway_exception);
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

/* REMOTE METHOD STUB:addRef */
static void
remote_bHYPRE_MPICommunicator_addRef(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_MPICommunicator__remote* r_obj = (struct 
      bHYPRE_MPICommunicator__remote*)self->d_data;
    LOCK_STATIC_GLOBALS;
    r_obj->d_refcount++;
    UNLOCK_STATIC_GLOBALS;
  }
}

/* REMOTE METHOD STUB:deleteRef */
static void
remote_bHYPRE_MPICommunicator_deleteRef(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
  /* out */ struct sidl_BaseInterface__object* *_ex)
{
  LANG_SPECIFIC_INIT();
  *_ex = NULL;
  {
    struct bHYPRE_MPICommunicator__remote* r_obj = (struct 
      bHYPRE_MPICommunicator__remote*)self->d_data;
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
remote_bHYPRE_MPICommunicator_isSame(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
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
      bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
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
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.MPICommunicator.isSame.", &throwaway_exception);
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
remote_bHYPRE_MPICommunicator_isType(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
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
      bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "isType", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */
    sidl_rmi_Invocation_packString( _inv, "name", name, _ex);SIDL_CHECK(*_ex);

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.MPICommunicator.isType.", &throwaway_exception);
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
remote_bHYPRE_MPICommunicator_getClassInfo(
  /* in */ struct bHYPRE_MPICommunicator__object* self ,
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
      bHYPRE_MPICommunicator__remote*)self->d_data)->d_ih;
    sidl_rmi_Invocation _inv = sidl_rmi_InstanceHandle_createInvocation( _conn, 
      "getClassInfo", _ex ); SIDL_CHECK(*_ex);

    /* pack in and inout arguments */

    /* send actual RMI request */
    _rsvp = sidl_rmi_Invocation_invokeMethod(_inv, _ex);SIDL_CHECK(*_ex);

    _be = sidl_rmi_Response_getExceptionThrown(_rsvp, _ex);SIDL_CHECK(*_ex);
    if(_be != NULL) {
      sidl_BaseInterface throwaway_exception = NULL;
sidl_BaseException_addLine(_be, "Exception unserialized from bHYPRE.MPICommunicator.getClassInfo.", &throwaway_exception);
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

/* REMOTE EPV: create remote entry point vectors (EPVs). */
static void bHYPRE_MPICommunicator__init_remote_epv(void)
{
  /* assert( HAVE_LOCKED_STATIC_GLOBALS ); */
  struct bHYPRE_MPICommunicator__epv* epv = &s_rem_epv__bhypre_mpicommunicator;
  struct sidl_BaseClass__epv*         e0  = &s_rem_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*     e1  = &s_rem_epv__sidl_baseinterface;

  epv->f__cast             = remote_bHYPRE_MPICommunicator__cast;
  epv->f__delete           = remote_bHYPRE_MPICommunicator__delete;
  epv->f__exec             = remote_bHYPRE_MPICommunicator__exec;
  epv->f__getURL           = remote_bHYPRE_MPICommunicator__getURL;
  epv->f__raddRef          = remote_bHYPRE_MPICommunicator__raddRef;
  epv->f__isRemote         = remote_bHYPRE_MPICommunicator__isRemote;
  epv->f__set_hooks        = remote_bHYPRE_MPICommunicator__set_hooks;
  epv->f__ctor             = NULL;
  epv->f__ctor2            = NULL;
  epv->f__dtor             = NULL;
  epv->f_Destroy           = remote_bHYPRE_MPICommunicator_Destroy;
  epv->f_addRef            = remote_bHYPRE_MPICommunicator_addRef;
  epv->f_deleteRef         = remote_bHYPRE_MPICommunicator_deleteRef;
  epv->f_isSame            = remote_bHYPRE_MPICommunicator_isSame;
  epv->f_isType            = remote_bHYPRE_MPICommunicator_isType;
  epv->f_getClassInfo      = remote_bHYPRE_MPICommunicator_getClassInfo;

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
    struct sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e0->f_addRef       = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_addRef;
  e0->f_deleteRef    = (void (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object **)) epv->f_deleteRef;
  e0->f_isSame       = (sidl_bool (*)(struct sidl_BaseClass__object*,struct 
    sidl_BaseInterface__object*,struct sidl_BaseInterface__object **)) 
    epv->f_isSame;
  e0->f_isType       = (sidl_bool (*)(struct sidl_BaseClass__object*,const 
    char*,struct sidl_BaseInterface__object **)) epv->f_isType;
  e0->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*,struct sidl_BaseInterface__object **)) 
    epv->f_getClassInfo;

  e1->f__cast        = (void* (*)(void*,const char*,sidl_BaseInterface*)) 
    epv->f__cast;
  e1->f__delete      = (void (*)(void*,sidl_BaseInterface*)) epv->f__delete;
  e1->f__getURL      = (char* (*)(void*,sidl_BaseInterface*)) epv->f__getURL;
  e1->f__raddRef     = (void (*)(void*,sidl_BaseInterface*)) epv->f__raddRef;
  e1->f__isRemote    = (sidl_bool (*)(void*,sidl_BaseInterface*)) 
    epv->f__isRemote;
  e1->f__set_hooks   = (void (*)(void*,int32_t, sidl_BaseInterface*)) 
    epv->f__set_hooks;
  e1->f__exec        = (void (*)(void*,const char*,struct 
    sidl_rmi_Call__object*,struct sidl_rmi_Return__object*,struct 
    sidl_BaseInterface__object **)) epv->f__exec;
  e1->f_addRef       = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_addRef;
  e1->f_deleteRef    = (void (*)(void*,struct sidl_BaseInterface__object **)) 
    epv->f_deleteRef;
  e1->f_isSame       = (sidl_bool (*)(void*,struct sidl_BaseInterface__object*,
    struct sidl_BaseInterface__object **)) epv->f_isSame;
  e1->f_isType       = (sidl_bool (*)(void*,const char*,struct 
    sidl_BaseInterface__object **)) epv->f_isType;
  e1->f_getClassInfo = (struct sidl_ClassInfo__object* (*)(void*,struct 
    sidl_BaseInterface__object **)) epv->f_getClassInfo;

  s_remote_initialized = 1;
}

/* Create an instance that connects to an existing remote object. */
static struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__remoteConnect(const char *url, sidl_bool ar, 
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_MPICommunicator__object* self;

  struct bHYPRE_MPICommunicator__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_MPICommunicator__remote* r_obj;
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
    return bHYPRE_MPICommunicator__rmicast(bi,_ex);SIDL_CHECK(*_ex);
  }
  instance = sidl_rmi_ProtocolFactory_connectInstance(url, ar, _ex ); 
    SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_MPICommunicator__object*) malloc(
      sizeof(struct bHYPRE_MPICommunicator__object));

  r_obj =
    (struct bHYPRE_MPICommunicator__remote*) malloc(
      sizeof(struct bHYPRE_MPICommunicator__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_MPICommunicator__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_mpicommunicator;

  self->d_data = (void*) r_obj;

  return self;
  EXIT:
  return NULL;
}
/* Create an instance that uses an already existing  */
/* InstanceHandle to connect to an existing remote object. */
static struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__IHConnect(sidl_rmi_InstanceHandle instance, 
  sidl_BaseInterface *_ex)
{
  struct bHYPRE_MPICommunicator__object* self;

  struct bHYPRE_MPICommunicator__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_MPICommunicator__remote* r_obj;
  self =
    (struct bHYPRE_MPICommunicator__object*) malloc(
      sizeof(struct bHYPRE_MPICommunicator__object));

  r_obj =
    (struct bHYPRE_MPICommunicator__remote*) malloc(
      sizeof(struct bHYPRE_MPICommunicator__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_MPICommunicator__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_mpicommunicator;

  self->d_data = (void*) r_obj;

  sidl_rmi_InstanceHandle_addRef(instance,_ex);SIDL_CHECK(*_ex);
  return self;
  EXIT:
  return NULL;
}
/* REMOTE: generate remote instance given URL string. */
static struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__remoteCreate(const char *url, sidl_BaseInterface *_ex)
{
  sidl_BaseInterface _throwaway_exception = NULL;
  struct bHYPRE_MPICommunicator__object* self;

  struct bHYPRE_MPICommunicator__object* s0;
  struct sidl_BaseClass__object* s1;

  struct bHYPRE_MPICommunicator__remote* r_obj;
  sidl_rmi_InstanceHandle instance = sidl_rmi_ProtocolFactory_createInstance(
    url, "bHYPRE.MPICommunicator", _ex ); SIDL_CHECK(*_ex);
  if ( instance == NULL) { return NULL; }
  self =
    (struct bHYPRE_MPICommunicator__object*) malloc(
      sizeof(struct bHYPRE_MPICommunicator__object));

  r_obj =
    (struct bHYPRE_MPICommunicator__remote*) malloc(
      sizeof(struct bHYPRE_MPICommunicator__remote));

  r_obj->d_refcount = 1;
  r_obj->d_ih = instance;
  s0 =                                  self;
  s1 =                                  &s0->d_sidl_baseclass;

  LOCK_STATIC_GLOBALS;
  if (!s_remote_initialized) {
    bHYPRE_MPICommunicator__init_remote_epv();
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv    = &s_rem_epv__sidl_baseinterface;
  s1->d_sidl_baseinterface.d_object = (void*) self;

  s1->d_data = (void*) r_obj;
  s1->d_epv  = &s_rem_epv__sidl_baseclass;

  s0->d_data = (void*) r_obj;
  s0->d_epv  = &s_rem_epv__bhypre_mpicommunicator;

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

struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__rmicast(
  void* obj,
  sidl_BaseInterface* _ex)
{
  struct bHYPRE_MPICommunicator__object* cast = NULL;

  *_ex = NULL;
  if(!connect_loaded) {
    sidl_rmi_ConnectRegistry_registerConnect("bHYPRE.MPICommunicator", (
      void*)bHYPRE_MPICommunicator__IHConnect, _ex);
    connect_loaded = 1;
  }
  if (obj != NULL) {
    struct sidl_BaseInterface__object* base = (struct 
      sidl_BaseInterface__object*) obj;
    cast = (struct bHYPRE_MPICommunicator__object*) (*base->d_epv->f__cast)(
      base->d_object,
      "bHYPRE.MPICommunicator", _ex); SIDL_CHECK(*_ex);
  }

  return cast;
  EXIT:
  return NULL;
}

/*
 * RMI connector function for the class.
 */

struct bHYPRE_MPICommunicator__object*
bHYPRE_MPICommunicator__connectI(const char* url, sidl_bool ar, struct 
  sidl_BaseInterface__object **_ex)
{
  return bHYPRE_MPICommunicator__remoteConnect(url, ar, _ex);
}

/*
 * Function to extract IOR reference from the Java object.
 */

static struct bHYPRE_MPICommunicator__object* _get_ior(
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
  return (struct bHYPRE_MPICommunicator__object*) ptr;
}

/*
 * External reference to IOR methods.
 */

static const struct bHYPRE_MPICommunicator__external* s_external = NULL;

/*
 * External reference to static EPV.
 */

static const struct bHYPRE_MPICommunicator__sepv* s_sepv = NULL;

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
  _res = POINTER_TO_JLONG(bHYPRE_MPICommunicator__remoteCreate(_tmp_url, &_ex));
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
  struct bHYPRE_MPICommunicator__object *_ptr = (*s_external->createObject)(
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
  _res = POINTER_TO_JLONG(bHYPRE_MPICommunicator__remoteConnect(_tmp_url, 1, 
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
 *  Create an MPICommunicator object from C code. 
 */

static jobject
jni_CreateC(
  JNIEnv* env,
  jclass  cls,
  jlong _arg_mpi_comm)
{
  /*
   * Declare return and temporary variables.
   */

  void* _tmp_mpi_comm = (void*) NULL;
  struct bHYPRE_MPICommunicator__object* _ior_res = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  jobject _res = (jobject) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _tmp_mpi_comm = (void*) JLONG_TO_POINTER(_arg_mpi_comm);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(s_sepv->f_CreateC))(
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
  _res = sidl_Java_I2J_cls(env, _ior_res, "bHYPRE.MPICommunicator", 
    FALSE);JAVA_CHECK(env);

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 *  Create an MPICommunicator object from Fortran code. 
 */

static jobject
jni_CreateF(
  JNIEnv* env,
  jclass  cls,
  jlong _arg_mpi_comm)
{
  /*
   * Declare return and temporary variables.
   */

  void* _tmp_mpi_comm = (void*) NULL;
  struct bHYPRE_MPICommunicator__object* _ior_res = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  jobject _res = (jobject) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */

  _tmp_mpi_comm = (void*) JLONG_TO_POINTER(_arg_mpi_comm);

  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(s_sepv->f_CreateF))(
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
  _res = sidl_Java_I2J_cls(env, _ior_res, "bHYPRE.MPICommunicator", 
    FALSE);JAVA_CHECK(env);

  return _res;
  JAVA_EXIT:

  return _res;
}

/*
 *  Create an MPICommunicator object which represents MPI_Comm_World. 
 */

static jobject
jni_Create_MPICommWorld(
  JNIEnv* env,
  jclass  cls)
{
  /*
   * Declare return and temporary variables.
   */

  struct bHYPRE_MPICommunicator__object* _ior_res = (struct 
    bHYPRE_MPICommunicator__object*) NULL;
  jobject _res = (jobject) NULL;
  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */


  /*
   * Call the IOR method through the EPV.
   */

  _ior_res = (*(s_sepv->f_Create_MPICommWorld))(
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
  _res = sidl_Java_I2J_cls(env, _ior_res, "bHYPRE.MPICommunicator", 
    FALSE);JAVA_CHECK(env);

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

  struct bHYPRE_MPICommunicator__object* _ior = NULL;
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
 *  Init and Finalize are to help debug MPI interfaces;
 * you should normally use the MPI library more directly:
 */

static void
jni_Init(
  JNIEnv* env,
  jclass  cls)
{
  /*
   * Declare return and temporary variables.
   */

  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */


  /*
   * Call the IOR method through the EPV.
   */

  (*(s_sepv->f_Init))(
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
 * Method:  Finalize[]
 */

static void
jni_Finalize(
  JNIEnv* env,
  jclass  cls)
{
  /*
   * Declare return and temporary variables.
   */

  struct sidl_BaseInterface__object *_ex = NULL;

  /*
   * Preprocess Java types and convert into IOR.
   */


  /*
   * Call the IOR method through the EPV.
   */

  (*(s_sepv->f_Finalize))(
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

  struct bHYPRE_MPICommunicator__object* _ior = NULL;
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

  struct bHYPRE_MPICommunicator__object* _ior = NULL;
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

void bHYPRE_MPICommunicator__register(JNIEnv* env)
{
  JNINativeMethod methods[13];
  jclass cls;

#ifdef SIDL_STATIC_LIBRARY
  s_external = bHYPRE_MPICommunicator__externals();
#else
  s_external = (struct bHYPRE_MPICommunicator__external*)sidl_dynamicLoadIOR(
    "bHYPRE.MPICommunicator","bHYPRE_MPICommunicator__externals") ;
  sidl_checkIORVersion("bHYPRE.MPICommunicator", 
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
  methods[2].signature = "(LbHYPRE/MPICommunicator;)J";
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
  methods[7].name      = "CreateC";
  methods[7].signature = "(J)LbHYPRE/MPICommunicator;";
  methods[7].fnPtr     = (void *)jni_CreateC;
  methods[8].name      = "CreateF";
  methods[8].signature = "(J)LbHYPRE/MPICommunicator;";
  methods[8].fnPtr     = (void *)jni_CreateF;
  methods[9].name      = "Create_MPICommWorld";
  methods[9].signature = "()LbHYPRE/MPICommunicator;";
  methods[9].fnPtr     = (void *)jni_Create_MPICommWorld;
  methods[10].name      = "Destroy";
  methods[10].signature = "()V";
  methods[10].fnPtr     = (void *)jni_Destroy;
  methods[11].name      = "Init";
  methods[11].signature = "()V";
  methods[11].fnPtr     = (void *)jni_Init;
  methods[12].name      = "Finalize";
  methods[12].signature = "()V";
  methods[12].fnPtr     = (void *)jni_Finalize;


  cls = (*env)->FindClass(env, "bHYPRE/MPICommunicator");
  if (cls) {
    (*env)->RegisterNatives(env, cls, methods, 13);
    (*env)->DeleteLocalRef(env, cls);
  }
}
