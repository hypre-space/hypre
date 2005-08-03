/*
 * File:          sidlx_rmi_Simsponse_IOR.c
 * Symbol:        sidlx.rmi.Simsponse-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for sidlx.rmi.Simsponse
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidlx_rmi_Simsponse_IOR.h"
#ifndef included_sidl_BaseClass_Impl_h
#include "sidl_BaseClass_Impl.h"
#endif
#ifndef included_sidl_BaseClass_h
#include "sidl_BaseClass.h"
#endif
#ifndef included_sidl_ClassInfo_h
#include "sidl_ClassInfo.h"
#endif
#ifndef included_sidl_ClassInfoI_h
#include "sidl_ClassInfoI.h"
#endif

#ifndef NULL
#define NULL 0
#endif

#include "sidl_thread.h"
#ifdef HAVE_PTHREAD
static struct sidl_recursive_mutex_t sidlx_rmi_Simsponse__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_Simsponse__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_Simsponse__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_Simsponse__mutex )==EDEADLOCK) */
#else
#define LOCK_STATIC_GLOBALS
#define UNLOCK_STATIC_GLOBALS
/* #define HAVE_LOCKED_STATIC_GLOBALS (1) */
#endif

/*
 * Static variables to hold version of IOR
 */

static const int32_t s_IOR_MAJOR_VERSION = 0;
static const int32_t s_IOR_MINOR_VERSION = 9;

/*
 * Static variable to hold shared ClassInfo interface.
 */

static sidl_ClassInfo s_classInfo = NULL;
static int s_classInfo_init = 1;

/*
 * Static variable to make sure _load called no more than once
 */

static int s_load_called = 0;
/*
 * Static variables for managing EPV initialization.
 */

static int s_method_initialized = 0;

static struct sidlx_rmi_Simsponse__epv s_new_epv__sidlx_rmi_simsponse;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

static struct sidl_io_Deserializer__epv s_new_epv__sidl_io_deserializer;

static struct sidl_rmi_Response__epv s_new_epv__sidl_rmi_response;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_rmi_Simsponse__set_epv(
  struct sidlx_rmi_Simsponse__epv* epv);
extern void sidlx_rmi_Simsponse__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_rmi_Simsponse_addRef__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_addRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_deleteRef__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_deleteRef)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_isSame__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* iobj;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_isSame)(
    self,
    iobj);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_queryInt__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_BaseInterface__object* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_queryInt)(
    self,
    name);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_isType__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "name", &name, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_isType)(
    self,
    name);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_getClassInfo__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_init__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* methodName= NULL;
  char* className= NULL;
  char* objectid= NULL;
  struct sidlx_rmi_Socket__object* sock;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "methodName", &methodName, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "className", &className, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "objectid", &objectid, _ex2);

  /* make the call */
  (self->d_epv->f_init)(
    self,
    methodName,
    className,
    objectid,
    sock,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_getMethodName__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getMethodName)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_getClassName__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassName)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_getObjectID__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getObjectID)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_unpackBool__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  sidl_bool value_tmp;
  sidl_bool* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackBool)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packBool( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackChar__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  char value_tmp;
  char* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackChar)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packChar( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackInt__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  int32_t value_tmp;
  int32_t* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackInt)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackLong__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  int64_t value_tmp;
  int64_t* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackLong)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packLong( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackFloat__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  float value_tmp;
  float* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackFloat)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packFloat( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackDouble__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  double value_tmp;
  double* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackDouble)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packDouble( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackFcomplex__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_fcomplex value_tmp;
  struct sidl_fcomplex* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackFcomplex)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packFcomplex( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackDcomplex__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_dcomplex value_tmp;
  struct sidl_dcomplex* value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackDcomplex)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packDcomplex( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_unpackString__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  char* value_tmp;
  char** value= &value_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);

  /* make the call */
  (self->d_epv->f_unpackString)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packString( outArgs, "value", *value, _ex2);

}

static void
sidlx_rmi_Simsponse_getExceptionThrown__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseException__object* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getExceptionThrown)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simsponse_done__exec(
        struct sidlx_rmi_Simsponse__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_done)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void ior_sidlx_rmi_Simsponse__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_rmi_Simsponse__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidlx_rmi_Simsponse__cast(
  struct sidlx_rmi_Simsponse__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidlx_rmi_Simsponse__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.rmi.Simsponse")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.io.Deserializer")) {
    cast = (void*) &s0->d_sidl_io_deserializer;
  } else if (!strcmp(name, "sidl.rmi.Response")) {
    cast = (void*) &s0->d_sidl_rmi_response;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s1->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidlx_rmi_Simsponse__delete(
  struct sidlx_rmi_Simsponse__object* self)
{
  sidlx_rmi_Simsponse__fini(self);
  memset((void*)self, 0, sizeof(struct sidlx_rmi_Simsponse__object));
  free((void*) self);
}

static char*
ior_sidlx_rmi_Simsponse__getURL(
    struct sidlx_rmi_Simsponse__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidlx_rmi_Simsponse__method {
  const char *d_name;
  void (*d_func)(struct sidlx_rmi_Simsponse__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidlx_rmi_Simsponse__exec(
    struct sidlx_rmi_Simsponse__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidlx_rmi_Simsponse__method  s_methods[] = {
    { "addRef", sidlx_rmi_Simsponse_addRef__exec },
    { "deleteRef", sidlx_rmi_Simsponse_deleteRef__exec },
    { "done", sidlx_rmi_Simsponse_done__exec },
    { "getClassInfo", sidlx_rmi_Simsponse_getClassInfo__exec },
    { "getClassName", sidlx_rmi_Simsponse_getClassName__exec },
    { "getExceptionThrown", sidlx_rmi_Simsponse_getExceptionThrown__exec },
    { "getMethodName", sidlx_rmi_Simsponse_getMethodName__exec },
    { "getObjectID", sidlx_rmi_Simsponse_getObjectID__exec },
    { "init", sidlx_rmi_Simsponse_init__exec },
    { "isSame", sidlx_rmi_Simsponse_isSame__exec },
    { "isType", sidlx_rmi_Simsponse_isType__exec },
    { "queryInt", sidlx_rmi_Simsponse_queryInt__exec },
    { "unpackBool", sidlx_rmi_Simsponse_unpackBool__exec },
    { "unpackChar", sidlx_rmi_Simsponse_unpackChar__exec },
    { "unpackDcomplex", sidlx_rmi_Simsponse_unpackDcomplex__exec },
    { "unpackDouble", sidlx_rmi_Simsponse_unpackDouble__exec },
    { "unpackFcomplex", sidlx_rmi_Simsponse_unpackFcomplex__exec },
    { "unpackFloat", sidlx_rmi_Simsponse_unpackFloat__exec },
    { "unpackInt", sidlx_rmi_Simsponse_unpackInt__exec },
    { "unpackLong", sidlx_rmi_Simsponse_unpackLong__exec },
    { "unpackString", sidlx_rmi_Simsponse_unpackString__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_rmi_Simsponse__method);
  if (methodName) {
    /* Use binary search to locate method */
    while (l < u) {
      i = (l + u) >> 1;
      if (!(cmp=strcmp(methodName, s_methods[i].d_name))) {
        (s_methods[i].d_func)(self, inArgs, outArgs);
        return;
      }
      else if (cmp < 0) u = i;
      else l = i + 1;
    }
  }
  /* TODO: add code for method not found */
}
/*
 * EPV: create method entry point vector (EPV) structure.
 */

static void sidlx_rmi_Simsponse__init_epv(
  struct sidlx_rmi_Simsponse__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_rmi_Simsponse__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  struct sidlx_rmi_Simsponse__epv*  epv  = &s_new_epv__sidlx_rmi_simsponse;
  struct sidl_BaseClass__epv*       e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*   e1   = &s_new_epv__sidl_baseinterface;
  struct sidl_io_Deserializer__epv* e2   = &s_new_epv__sidl_io_deserializer;
  struct sidl_rmi_Response__epv*    e3   = &s_new_epv__sidl_rmi_response;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_sidlx_rmi_Simsponse__cast;
  epv->f__delete                  = ior_sidlx_rmi_Simsponse__delete;
  epv->f__exec                    = ior_sidlx_rmi_Simsponse__exec;
  epv->f__getURL                  = ior_sidlx_rmi_Simsponse__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_rmi_Simsponse__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_rmi_Simsponse__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_rmi_Simsponse__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidlx_rmi_Simsponse__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_rmi_Simsponse__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_Simsponse__object*)) s1->d_epv->f_getClassInfo;
  epv->f_init                     = NULL;
  epv->f_getMethodName            = NULL;
  epv->f_getClassName             = NULL;
  epv->f_getObjectID              = NULL;
  epv->f_unpackBool               = NULL;
  epv->f_unpackChar               = NULL;
  epv->f_unpackInt                = NULL;
  epv->f_unpackLong               = NULL;
  epv->f_unpackFloat              = NULL;
  epv->f_unpackDouble             = NULL;
  epv->f_unpackFcomplex           = NULL;
  epv->f_unpackDcomplex           = NULL;
  epv->f_unpackString             = NULL;
  epv->f_getExceptionThrown       = NULL;
  epv->f_done                     = NULL;

  sidlx_rmi_Simsponse__set_epv(epv);

  e0->f__cast               = (void* (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f__cast;
  e0->f__delete             = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f__delete;
  e0->f__exec               = (void (*)(struct sidl_BaseClass__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e0->f_addRef              = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_addRef;
  e0->f_deleteRef           = (void (*)(struct sidl_BaseClass__object*)) 
    epv->f_deleteRef;
  e0->f_isSame              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e0->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_BaseClass__object*,const char*)) epv->f_queryInt;
  e0->f_isType              = (sidl_bool (*)(struct sidl_BaseClass__object*,
    const char*)) epv->f_isType;
  e0->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_BaseClass__object*)) epv->f_getClassInfo;

  e1->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e1->f__delete             = (void (*)(void*)) epv->f__delete;
  e1->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e1->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e1->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e1->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e1->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e1->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e1->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;

  e2->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e2->f__delete             = (void (*)(void*)) epv->f__delete;
  e2->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e2->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e2->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e2->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e2->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e2->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e2->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e2->f_unpackBool          = (void (*)(void*,const char*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_unpackBool;
  e2->f_unpackChar          = (void (*)(void*,const char*,char*,
    struct sidl_BaseInterface__object **)) epv->f_unpackChar;
  e2->f_unpackInt           = (void (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackInt;
  e2->f_unpackLong          = (void (*)(void*,const char*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackLong;
  e2->f_unpackFloat         = (void (*)(void*,const char*,float*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e2->f_unpackDouble        = (void (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e2->f_unpackFcomplex      = (void (*)(void*,const char*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e2->f_unpackDcomplex      = (void (*)(void*,const char*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e2->f_unpackString        = (void (*)(void*,const char*,char**,
    struct sidl_BaseInterface__object **)) epv->f_unpackString;

  e3->f__cast               = (void* (*)(void*,const char*)) epv->f__cast;
  e3->f__delete             = (void (*)(void*)) epv->f__delete;
  e3->f__exec               = (void (*)(void*,const char*,
    struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef              = (void (*)(void*)) epv->f_addRef;
  e3->f_deleteRef           = (void (*)(void*)) epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt            = (struct sidl_BaseInterface__object* (*)(void*,
    const char*)) epv->f_queryInt;
  e3->f_isType              = (sidl_bool (*)(void*,const char*)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(void*)) 
    epv->f_getClassInfo;
  e3->f_unpackBool          = (void (*)(void*,const char*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_unpackBool;
  e3->f_unpackChar          = (void (*)(void*,const char*,char*,
    struct sidl_BaseInterface__object **)) epv->f_unpackChar;
  e3->f_unpackInt           = (void (*)(void*,const char*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackInt;
  e3->f_unpackLong          = (void (*)(void*,const char*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_unpackLong;
  e3->f_unpackFloat         = (void (*)(void*,const char*,float*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFloat;
  e3->f_unpackDouble        = (void (*)(void*,const char*,double*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDouble;
  e3->f_unpackFcomplex      = (void (*)(void*,const char*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackFcomplex;
  e3->f_unpackDcomplex      = (void (*)(void*,const char*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_unpackDcomplex;
  e3->f_unpackString        = (void (*)(void*,const char*,char**,
    struct sidl_BaseInterface__object **)) epv->f_unpackString;
  e3->f_getExceptionThrown  = (struct sidl_BaseException__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getExceptionThrown;
  e3->f_done                = (sidl_bool (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_done;

  s_method_initialized = 1;
  ior_sidlx_rmi_Simsponse__ensure_load_called();
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidlx_rmi_Simsponse__super(void) {
  return s_old_epv__sidl_baseclass;
}

/*
 * initClassInfo: create a ClassInfo interface if necessary.
 */

static void
initClassInfo(sidl_ClassInfo *info)
{
  LOCK_STATIC_GLOBALS;
  if (s_classInfo_init) {
    sidl_ClassInfoI impl;
    s_classInfo_init = 0;
    impl = sidl_ClassInfoI__create();
    s_classInfo = sidl_ClassInfo__cast(impl);
    if (impl) {
      sidl_ClassInfoI_setName(impl, "sidlx.rmi.Simsponse");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
    }
  }
  if (s_classInfo) {
    if (*info) {
      sidl_ClassInfo_deleteRef(*info);
    }
    *info = s_classInfo;
    sidl_ClassInfo_addRef(*info);
  }
UNLOCK_STATIC_GLOBALS;
}

/*
 * initMetadata: store IOR version & class in sidl.BaseClass's data
 */

static void
initMetadata(struct sidlx_rmi_Simsponse__object* self)
{
  if (self) {
    struct sidl_BaseClass__data *data = 
      sidl_BaseClass__get_data(sidl_BaseClass__cast(self));
    if (data) {
      data->d_IOR_major_version = s_IOR_MAJOR_VERSION;
      data->d_IOR_minor_version = s_IOR_MINOR_VERSION;
      initClassInfo(&(data->d_classinfo));
    }
  }
}

/*
 * NEW: allocate object and initialize it.
 */

struct sidlx_rmi_Simsponse__object*
sidlx_rmi_Simsponse__new(void)
{
  struct sidlx_rmi_Simsponse__object* self =
    (struct sidlx_rmi_Simsponse__object*) malloc(
      sizeof(struct sidlx_rmi_Simsponse__object));
  sidlx_rmi_Simsponse__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_rmi_Simsponse__init(
  struct sidlx_rmi_Simsponse__object* self)
{
  struct sidlx_rmi_Simsponse__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_Simsponse__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_sidl_io_deserializer.d_epv = &s_new_epv__sidl_io_deserializer;
  s0->d_sidl_rmi_response.d_epv    = &s_new_epv__sidl_rmi_response;
  s0->d_epv                        = &s_new_epv__sidlx_rmi_simsponse;

  s0->d_sidl_io_deserializer.d_object = self;

  s0->d_sidl_rmi_response.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidlx_rmi_Simsponse__fini(
  struct sidlx_rmi_Simsponse__object* self)
{
  struct sidlx_rmi_Simsponse__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_rmi_Simsponse__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_rmi_Simsponse__external
s_externalEntryPoints = {
  sidlx_rmi_Simsponse__new,
  sidlx_rmi_Simsponse__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_Simsponse__external*
sidlx_rmi_Simsponse__externals(void)
{
  return &s_externalEntryPoints;
}

