/*
 * File:          sidlx_io_IOException_IOR.c
 * Symbol:        sidlx.io.IOException-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for sidlx.io.IOException
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidlx_io_IOException_IOR.h"
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
static struct sidl_recursive_mutex_t sidlx_io_IOException__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_io_IOException__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_io_IOException__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_io_IOException__mutex )==EDEADLOCK) */
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

static struct sidlx_io_IOException__epv s_new_epv__sidlx_io_ioexception;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseException__epv  s_new_epv__sidl_baseexception;
static struct sidl_BaseException__epv* s_old_epv__sidl_baseexception;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

static struct sidl_SIDLException__epv  s_new_epv__sidl_sidlexception;
static struct sidl_SIDLException__epv* s_old_epv__sidl_sidlexception;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_io_IOException__set_epv(
  struct sidlx_io_IOException__epv* epv);
extern void sidlx_io_IOException__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_io_IOException_addRef__exec(
        struct sidlx_io_IOException__object* self,
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
sidlx_io_IOException_deleteRef__exec(
        struct sidlx_io_IOException__object* self,
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
sidlx_io_IOException_isSame__exec(
        struct sidlx_io_IOException__object* self,
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
sidlx_io_IOException_queryInt__exec(
        struct sidlx_io_IOException__object* self,
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
sidlx_io_IOException_isType__exec(
        struct sidlx_io_IOException__object* self,
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
sidlx_io_IOException_getClassInfo__exec(
        struct sidlx_io_IOException__object* self,
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
sidlx_io_IOException_getNote__exec(
        struct sidlx_io_IOException__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getNote)(
    self);

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_io_IOException_setNote__exec(
        struct sidlx_io_IOException__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* message= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "message", &message, _ex2);

  /* make the call */
  (self->d_epv->f_setNote)(
    self,
    message);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_io_IOException_getTrace__exec(
        struct sidlx_io_IOException__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getTrace)(
    self);

  /* pack return value */
  sidl_io_Serializer_packString( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_io_IOException_addLine__exec(
        struct sidlx_io_IOException__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* traceline= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "traceline", &traceline, _ex2);

  /* make the call */
  (self->d_epv->f_addLine)(
    self,
    traceline);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_io_IOException_add__exec(
        struct sidlx_io_IOException__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* filename= NULL;
  int32_t lineno;
  char* methodname= NULL;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackString( inArgs, "filename", &filename, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "lineno", &lineno, _ex2);
  sidl_io_Deserializer_unpackString( inArgs, "methodname", &methodname, _ex2);

  /* make the call */
  (self->d_epv->f_add)(
    self,
    filename,
    lineno,
    methodname);

  /* pack return value */
  /* pack out and inout argments */

}

static void ior_sidlx_io_IOException__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_io_IOException__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidlx_io_IOException__cast(
  struct sidlx_io_IOException__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidlx_io_IOException__object* s0 = self;
  struct sidl_SIDLException__object*   s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*       s2 = &s1->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.io.IOException")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidl.SIDLException")) {
    cast = (void*) s1;
  } else if (!strcmp(name, "sidl.BaseException")) {
    cast = (void*) &s1->d_sidl_baseexception;
  } else if (!strcmp(name, "sidl.BaseClass")) {
    cast = (void*) s2;
  } else if (!strcmp(name, "sidl.BaseInterface")) {
    cast = (void*) &s2->d_sidl_baseinterface;
  }

  return cast;
}

/*
 * DELETE: call destructor and free object memory.
 */

static void ior_sidlx_io_IOException__delete(
  struct sidlx_io_IOException__object* self)
{
  sidlx_io_IOException__fini(self);
  memset((void*)self, 0, sizeof(struct sidlx_io_IOException__object));
  free((void*) self);
}

static char*
ior_sidlx_io_IOException__getURL(
    struct sidlx_io_IOException__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidlx_io_IOException__method {
  const char *d_name;
  void (*d_func)(struct sidlx_io_IOException__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidlx_io_IOException__exec(
    struct sidlx_io_IOException__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidlx_io_IOException__method  s_methods[] = {
    { "add", sidlx_io_IOException_add__exec },
    { "addLine", sidlx_io_IOException_addLine__exec },
    { "addRef", sidlx_io_IOException_addRef__exec },
    { "deleteRef", sidlx_io_IOException_deleteRef__exec },
    { "getClassInfo", sidlx_io_IOException_getClassInfo__exec },
    { "getNote", sidlx_io_IOException_getNote__exec },
    { "getTrace", sidlx_io_IOException_getTrace__exec },
    { "isSame", sidlx_io_IOException_isSame__exec },
    { "isType", sidlx_io_IOException_isType__exec },
    { "queryInt", sidlx_io_IOException_queryInt__exec },
    { "setNote", sidlx_io_IOException_setNote__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_io_IOException__method);
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

static void sidlx_io_IOException__init_epv(
  struct sidlx_io_IOException__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_io_IOException__object* s0 = self;
  struct sidl_SIDLException__object*   s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*       s2 = &s1->d_sidl_baseclass;

  struct sidlx_io_IOException__epv*  epv  = &s_new_epv__sidlx_io_ioexception;
  struct sidl_BaseClass__epv*        e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseException__epv*    e1   = &s_new_epv__sidl_baseexception;
  struct sidl_BaseInterface__epv*    e2   = &s_new_epv__sidl_baseinterface;
  struct sidl_SIDLException__epv*    e3   = &s_new_epv__sidl_sidlexception;

  s_old_epv__sidl_baseinterface = s2->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s2->d_epv;

  s_old_epv__sidl_baseexception = s1->d_sidl_baseexception.d_epv;
  s_old_epv__sidl_sidlexception = s1->d_epv;

  epv->f__cast                    = ior_sidlx_io_IOException__cast;
  epv->f__delete                  = ior_sidlx_io_IOException__delete;
  epv->f__exec                    = ior_sidlx_io_IOException__exec;
  epv->f__getURL                  = ior_sidlx_io_IOException__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_io_IOException__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_io_IOException__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_io_IOException__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidlx_io_IOException__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_io_IOException__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_io_IOException__object*)) s1->d_epv->f_getClassInfo;
  epv->f_getNote                  = (char* (*)(struct 
    sidlx_io_IOException__object*)) s1->d_epv->f_getNote;
  epv->f_setNote                  = (void (*)(struct 
    sidlx_io_IOException__object*,const char*)) s1->d_epv->f_setNote;
  epv->f_getTrace                 = (char* (*)(struct 
    sidlx_io_IOException__object*)) s1->d_epv->f_getTrace;
  epv->f_addLine                  = (void (*)(struct 
    sidlx_io_IOException__object*,const char*)) s1->d_epv->f_addLine;
  epv->f_add                      = (void (*)(struct 
    sidlx_io_IOException__object*,const char*,int32_t,
    const char*)) s1->d_epv->f_add;

  sidlx_io_IOException__set_epv(epv);

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
  e1->f_getNote             = (char* (*)(void*)) epv->f_getNote;
  e1->f_setNote             = (void (*)(void*,const char*)) epv->f_setNote;
  e1->f_getTrace            = (char* (*)(void*)) epv->f_getTrace;
  e1->f_addLine             = (void (*)(void*,const char*)) epv->f_addLine;
  e1->f_add                 = (void (*)(void*,const char*,int32_t,
    const char*)) epv->f_add;

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

  e3->f__cast               = (void* (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f__cast;
  e3->f__delete             = (void (*)(struct sidl_SIDLException__object*)) 
    epv->f__delete;
  e3->f__exec               = (void (*)(struct sidl_SIDLException__object*,
    const char*,struct sidl_io_Deserializer__object*,
    struct sidl_io_Serializer__object*)) epv->f__exec;
  e3->f_addRef              = (void (*)(struct sidl_SIDLException__object*)) 
    epv->f_addRef;
  e3->f_deleteRef           = (void (*)(struct sidl_SIDLException__object*)) 
    epv->f_deleteRef;
  e3->f_isSame              = (sidl_bool (*)(struct sidl_SIDLException__object*,
    struct sidl_BaseInterface__object*)) epv->f_isSame;
  e3->f_queryInt            = (struct sidl_BaseInterface__object* (*)(struct 
    sidl_SIDLException__object*,const char*)) epv->f_queryInt;
  e3->f_isType              = (sidl_bool (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f_isType;
  e3->f_getClassInfo        = (struct sidl_ClassInfo__object* (*)(struct 
    sidl_SIDLException__object*)) epv->f_getClassInfo;
  e3->f_getNote             = (char* (*)(struct sidl_SIDLException__object*)) 
    epv->f_getNote;
  e3->f_setNote             = (void (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f_setNote;
  e3->f_getTrace            = (char* (*)(struct sidl_SIDLException__object*)) 
    epv->f_getTrace;
  e3->f_addLine             = (void (*)(struct sidl_SIDLException__object*,
    const char*)) epv->f_addLine;
  e3->f_add                 = (void (*)(struct sidl_SIDLException__object*,
    const char*,int32_t,const char*)) epv->f_add;

  s_method_initialized = 1;
  ior_sidlx_io_IOException__ensure_load_called();
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_SIDLException__epv* sidlx_io_IOException__super(void) {
  return s_old_epv__sidl_sidlexception;
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
      sidl_ClassInfoI_setName(impl, "sidlx.io.IOException");
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
initMetadata(struct sidlx_io_IOException__object* self)
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

struct sidlx_io_IOException__object*
sidlx_io_IOException__new(void)
{
  struct sidlx_io_IOException__object* self =
    (struct sidlx_io_IOException__object*) malloc(
      sizeof(struct sidlx_io_IOException__object));
  sidlx_io_IOException__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_io_IOException__init(
  struct sidlx_io_IOException__object* self)
{
  struct sidlx_io_IOException__object* s0 = self;
  struct sidl_SIDLException__object*   s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*       s2 = &s1->d_sidl_baseclass;

  sidl_SIDLException__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_io_IOException__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s2->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s2->d_epv                      = &s_new_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv = &s_new_epv__sidl_baseexception;
  s1->d_epv                      = &s_new_epv__sidl_sidlexception;

  s0->d_epv    = &s_new_epv__sidlx_io_ioexception;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidlx_io_IOException__fini(
  struct sidlx_io_IOException__object* self)
{
  struct sidlx_io_IOException__object* s0 = self;
  struct sidl_SIDLException__object*   s1 = &s0->d_sidl_sidlexception;
  struct sidl_BaseClass__object*       s2 = &s1->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s2->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s2->d_epv                      = s_old_epv__sidl_baseclass;

  s1->d_sidl_baseexception.d_epv = s_old_epv__sidl_baseexception;
  s1->d_epv                      = s_old_epv__sidl_sidlexception;

  sidl_SIDLException__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_io_IOException__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_io_IOException__external
s_externalEntryPoints = {
  sidlx_io_IOException__new,
  sidlx_io_IOException__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_io_IOException__external*
sidlx_io_IOException__externals(void)
{
  return &s_externalEntryPoints;
}

