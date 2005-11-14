/*
 * File:          sidlx_rmi_Simvocation_IOR.c
 * Symbol:        sidlx.rmi.Simvocation-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.rmi.Simvocation
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidlx_rmi_Simvocation_IOR.h"
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
static struct sidl_recursive_mutex_t sidlx_rmi_Simvocation__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_Simvocation__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_Simvocation__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_Simvocation__mutex )==EDEADLOCK) */
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

static struct sidlx_rmi_Simvocation__epv s_new_epv__sidlx_rmi_simvocation;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

static struct sidl_io_Serializer__epv s_new_epv__sidl_io_serializer;

static struct sidl_rmi_Invocation__epv s_new_epv__sidl_rmi_invocation;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_rmi_Simvocation__set_epv(
  struct sidlx_rmi_Simvocation__epv* epv);
extern void sidlx_rmi_Simvocation__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_rmi_Simvocation_addRef__exec(
        struct sidlx_rmi_Simvocation__object* self,
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
sidlx_rmi_Simvocation_deleteRef__exec(
        struct sidlx_rmi_Simvocation__object* self,
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
sidlx_rmi_Simvocation_isSame__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_BaseInterface__object* iobj = 0;
  sidl_bool _retval = FALSE;
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
sidlx_rmi_Simvocation_queryInt__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  struct sidl_BaseInterface__object* _retval = 0;
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
sidlx_rmi_Simvocation_isType__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* name= NULL;
  sidl_bool _retval = FALSE;
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
sidlx_rmi_Simvocation_getClassInfo__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_ClassInfo__object* _retval = 0;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getClassInfo)(
    self);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_init__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* methodName= NULL;
  char* className= NULL;
  char* objectid= NULL;
  struct sidlx_rmi_Socket__object* sock = 0;
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
sidlx_rmi_Simvocation_getMethodName__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval = 0;
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
sidlx_rmi_Simvocation_packBool__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  sidl_bool value = FALSE;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackBool( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packBool)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packChar__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  char value = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackChar( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packChar)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packInt__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  int32_t value = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packInt)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packLong__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  int64_t value = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackLong( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packLong)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packFloat__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  float value = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackFloat( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packFloat)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packDouble__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  double value = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackDouble( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packDouble)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packFcomplex__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_fcomplex value = { 0, 0 };
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackFcomplex( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packFcomplex)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packDcomplex__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  struct sidl_dcomplex value = { 0, 0 };
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackDcomplex( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packDcomplex)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_packString__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* key= NULL;
  char* value= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "key", &key, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "value", &value, _ex2);

  /* make the call */
  (self->d_epv->f_packString)(
    self,
    key,
    value,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_Simvocation_invokeMethod__exec(
        struct sidlx_rmi_Simvocation__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_rmi_Response__object* _retval = 0;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_invokeMethod)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void ior_sidlx_rmi_Simvocation__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_rmi_Simvocation__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidlx_rmi_Simvocation__cast(
  struct sidlx_rmi_Simvocation__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidlx_rmi_Simvocation__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.rmi.Simvocation")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.io.Serializer")) {
    cast = (void*) &s0->d_sidl_io_serializer;
  } else if (!strcmp(name, "sidl.rmi.Invocation")) {
    cast = (void*) &s0->d_sidl_rmi_invocation;
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

static void ior_sidlx_rmi_Simvocation__delete(
  struct sidlx_rmi_Simvocation__object* self)
{
  sidlx_rmi_Simvocation__fini(self);
  memset((void*)self, 0, sizeof(struct sidlx_rmi_Simvocation__object));
  free((void*) self);
}

static char*
ior_sidlx_rmi_Simvocation__getURL(
    struct sidlx_rmi_Simvocation__object* self) {
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidlx_rmi_Simvocation__method {
  const char *d_name;
  void (*d_func)(struct sidlx_rmi_Simvocation__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidlx_rmi_Simvocation__exec(
    struct sidlx_rmi_Simvocation__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidlx_rmi_Simvocation__method  s_methods[] = {
    { "addRef", sidlx_rmi_Simvocation_addRef__exec },
    { "deleteRef", sidlx_rmi_Simvocation_deleteRef__exec },
    { "getClassInfo", sidlx_rmi_Simvocation_getClassInfo__exec },
    { "getMethodName", sidlx_rmi_Simvocation_getMethodName__exec },
    { "init", sidlx_rmi_Simvocation_init__exec },
    { "invokeMethod", sidlx_rmi_Simvocation_invokeMethod__exec },
    { "isSame", sidlx_rmi_Simvocation_isSame__exec },
    { "isType", sidlx_rmi_Simvocation_isType__exec },
    { "packBool", sidlx_rmi_Simvocation_packBool__exec },
    { "packChar", sidlx_rmi_Simvocation_packChar__exec },
    { "packDcomplex", sidlx_rmi_Simvocation_packDcomplex__exec },
    { "packDouble", sidlx_rmi_Simvocation_packDouble__exec },
    { "packFcomplex", sidlx_rmi_Simvocation_packFcomplex__exec },
    { "packFloat", sidlx_rmi_Simvocation_packFloat__exec },
    { "packInt", sidlx_rmi_Simvocation_packInt__exec },
    { "packLong", sidlx_rmi_Simvocation_packLong__exec },
    { "packString", sidlx_rmi_Simvocation_packString__exec },
    { "queryInt", sidlx_rmi_Simvocation_queryInt__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_rmi_Simvocation__method);
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

static void sidlx_rmi_Simvocation__init_epv(
  struct sidlx_rmi_Simvocation__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_rmi_Simvocation__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  struct sidlx_rmi_Simvocation__epv*  epv  = &s_new_epv__sidlx_rmi_simvocation;
  struct sidl_BaseClass__epv*         e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*     e1   = &s_new_epv__sidl_baseinterface;
  struct sidl_io_Serializer__epv*     e2   = &s_new_epv__sidl_io_serializer;
  struct sidl_rmi_Invocation__epv*    e3   = &s_new_epv__sidl_rmi_invocation;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_sidlx_rmi_Simvocation__cast;
  epv->f__delete                  = ior_sidlx_rmi_Simvocation__delete;
  epv->f__exec                    = ior_sidlx_rmi_Simvocation__exec;
  epv->f__getURL                  = ior_sidlx_rmi_Simvocation__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_rmi_Simvocation__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_rmi_Simvocation__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_rmi_Simvocation__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidlx_rmi_Simvocation__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_rmi_Simvocation__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_Simvocation__object*)) s1->d_epv->f_getClassInfo;
  epv->f_init                     = NULL;
  epv->f_getMethodName            = NULL;
  epv->f_packBool                 = NULL;
  epv->f_packChar                 = NULL;
  epv->f_packInt                  = NULL;
  epv->f_packLong                 = NULL;
  epv->f_packFloat                = NULL;
  epv->f_packDouble               = NULL;
  epv->f_packFcomplex             = NULL;
  epv->f_packDcomplex             = NULL;
  epv->f_packString               = NULL;
  epv->f_invokeMethod             = NULL;

  sidlx_rmi_Simvocation__set_epv(epv);

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
  e2->f_packBool            = (void (*)(void*,const char*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packBool;
  e2->f_packChar            = (void (*)(void*,const char*,char,
    struct sidl_BaseInterface__object **)) epv->f_packChar;
  e2->f_packInt             = (void (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_packInt;
  e2->f_packLong            = (void (*)(void*,const char*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_packLong;
  e2->f_packFloat           = (void (*)(void*,const char*,float,
    struct sidl_BaseInterface__object **)) epv->f_packFloat;
  e2->f_packDouble          = (void (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_packDouble;
  e2->f_packFcomplex        = (void (*)(void*,const char*,struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packFcomplex;
  e2->f_packDcomplex        = (void (*)(void*,const char*,struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packDcomplex;
  e2->f_packString          = (void (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_packString;

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
  e3->f_packBool            = (void (*)(void*,const char*,sidl_bool,
    struct sidl_BaseInterface__object **)) epv->f_packBool;
  e3->f_packChar            = (void (*)(void*,const char*,char,
    struct sidl_BaseInterface__object **)) epv->f_packChar;
  e3->f_packInt             = (void (*)(void*,const char*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_packInt;
  e3->f_packLong            = (void (*)(void*,const char*,int64_t,
    struct sidl_BaseInterface__object **)) epv->f_packLong;
  e3->f_packFloat           = (void (*)(void*,const char*,float,
    struct sidl_BaseInterface__object **)) epv->f_packFloat;
  e3->f_packDouble          = (void (*)(void*,const char*,double,
    struct sidl_BaseInterface__object **)) epv->f_packDouble;
  e3->f_packFcomplex        = (void (*)(void*,const char*,struct sidl_fcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packFcomplex;
  e3->f_packDcomplex        = (void (*)(void*,const char*,struct sidl_dcomplex,
    struct sidl_BaseInterface__object **)) epv->f_packDcomplex;
  e3->f_packString          = (void (*)(void*,const char*,const char*,
    struct sidl_BaseInterface__object **)) epv->f_packString;
  e3->f_invokeMethod        = (struct sidl_rmi_Response__object* (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_invokeMethod;

  s_method_initialized = 1;
  ior_sidlx_rmi_Simvocation__ensure_load_called();
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidlx_rmi_Simvocation__super(void) {
  return s_old_epv__sidl_baseclass;
}

static void
cleanupClassInfo(void) {
  if (s_classInfo) {
    sidl_ClassInfo_deleteRef(s_classInfo);
  }
  s_classInfo_init = 1;
  s_classInfo = NULL;
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
      sidl_ClassInfoI_setName(impl, "sidlx.rmi.Simvocation");
      sidl_ClassInfoI_setIORVersion(impl, s_IOR_MAJOR_VERSION,
        s_IOR_MINOR_VERSION);
      atexit(cleanupClassInfo);
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
initMetadata(struct sidlx_rmi_Simvocation__object* self)
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

struct sidlx_rmi_Simvocation__object*
sidlx_rmi_Simvocation__new(void)
{
  struct sidlx_rmi_Simvocation__object* self =
    (struct sidlx_rmi_Simvocation__object*) malloc(
      sizeof(struct sidlx_rmi_Simvocation__object));
  sidlx_rmi_Simvocation__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_rmi_Simvocation__init(
  struct sidlx_rmi_Simvocation__object* self)
{
  struct sidlx_rmi_Simvocation__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_Simvocation__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_sidl_io_serializer.d_epv  = &s_new_epv__sidl_io_serializer;
  s0->d_sidl_rmi_invocation.d_epv = &s_new_epv__sidl_rmi_invocation;
  s0->d_epv                       = &s_new_epv__sidlx_rmi_simvocation;

  s0->d_sidl_io_serializer.d_object = self;

  s0->d_sidl_rmi_invocation.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidlx_rmi_Simvocation__fini(
  struct sidlx_rmi_Simvocation__object* self)
{
  struct sidlx_rmi_Simvocation__object* s0 = self;
  struct sidl_BaseClass__object*        s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_rmi_Simvocation__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_rmi_Simvocation__external
s_externalEntryPoints = {
  sidlx_rmi_Simvocation__new,
  sidlx_rmi_Simvocation__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_Simvocation__external*
sidlx_rmi_Simvocation__externals(void)
{
  return &s_externalEntryPoints;
}

