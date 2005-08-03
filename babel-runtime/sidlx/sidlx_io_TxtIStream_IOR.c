/*
 * File:          sidlx_io_TxtIStream_IOR.c
 * Symbol:        sidlx.io.TxtIStream-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for sidlx.io.TxtIStream
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidlx_io_TxtIStream_IOR.h"
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
static struct sidl_recursive_mutex_t sidlx_io_TxtIStream__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_io_TxtIStream__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_io_TxtIStream__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_io_TxtIStream__mutex )==EDEADLOCK) */
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

static struct sidlx_io_TxtIStream__epv s_new_epv__sidlx_io_txtistream;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

static struct sidlx_io_IStream__epv s_new_epv__sidlx_io_istream;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_io_TxtIStream__set_epv(
  struct sidlx_io_TxtIStream__epv* epv);
extern void sidlx_io_TxtIStream__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_io_TxtIStream_addRef__exec(
        struct sidlx_io_TxtIStream__object* self,
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
sidlx_io_TxtIStream_deleteRef__exec(
        struct sidlx_io_TxtIStream__object* self,
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
sidlx_io_TxtIStream_isSame__exec(
        struct sidlx_io_TxtIStream__object* self,
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
sidlx_io_TxtIStream_queryInt__exec(
        struct sidlx_io_TxtIStream__object* self,
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
sidlx_io_TxtIStream_isType__exec(
        struct sidlx_io_TxtIStream__object* self,
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
sidlx_io_TxtIStream_getClassInfo__exec(
        struct sidlx_io_TxtIStream__object* self,
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
sidlx_io_TxtIStream_setFD__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t fd;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "fd", &fd, _ex2);

  /* make the call */
  (self->d_epv->f_setFD)(
    self,
    fd);

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_io_TxtIStream_atEnd__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  sidl_bool _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_atEnd)(
    self);

  /* pack return value */
  sidl_io_Serializer_packBool( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_io_TxtIStream_read__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t nbytes;
  struct sidl_char__array* data_tmp;
  struct sidl_char__array** data= &data_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "nbytes", &nbytes, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_read)(
    self,
    nbytes,
    data,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_io_TxtIStream_readline__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_char__array* data_tmp;
  struct sidl_char__array** data= &data_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_readline)(
    self,
    data,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_io_TxtIStream_getBool__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  sidl_bool item_tmp;
  sidl_bool* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getBool)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packBool( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getChar__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char item_tmp;
  char* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getChar)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packChar( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getInt__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t item_tmp;
  int32_t* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getInt)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getLong__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int64_t item_tmp;
  int64_t* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getLong)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packLong( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getFloat__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  float item_tmp;
  float* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getFloat)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packFloat( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getDouble__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  double item_tmp;
  double* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getDouble)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packDouble( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getFcomplex__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_fcomplex item_tmp;
  struct sidl_fcomplex* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getFcomplex)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packFcomplex( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getDcomplex__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  struct sidl_dcomplex item_tmp;
  struct sidl_dcomplex* item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getDcomplex)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packDcomplex( outArgs, "item", *item, _ex2);

}

static void
sidlx_io_TxtIStream_getString__exec(
        struct sidlx_io_TxtIStream__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  char* item_tmp;
  char** item= &item_tmp;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  (self->d_epv->f_getString)(
    self,
    item,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */
  sidl_io_Serializer_packString( outArgs, "item", *item, _ex2);

}

static void ior_sidlx_io_TxtIStream__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_io_TxtIStream__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidlx_io_TxtIStream__cast(
  struct sidlx_io_TxtIStream__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidlx_io_TxtIStream__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.io.TxtIStream")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidlx.io.IStream")) {
    cast = (void*) &s0->d_sidlx_io_istream;
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

static void ior_sidlx_io_TxtIStream__delete(
  struct sidlx_io_TxtIStream__object* self)
{
  sidlx_io_TxtIStream__fini(self);
  memset((void*)self, 0, sizeof(struct sidlx_io_TxtIStream__object));
  free((void*) self);
}

static char*
ior_sidlx_io_TxtIStream__getURL(
    struct sidlx_io_TxtIStream__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidlx_io_TxtIStream__method {
  const char *d_name;
  void (*d_func)(struct sidlx_io_TxtIStream__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidlx_io_TxtIStream__exec(
    struct sidlx_io_TxtIStream__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidlx_io_TxtIStream__method  s_methods[] = {
    { "addRef", sidlx_io_TxtIStream_addRef__exec },
    { "atEnd", sidlx_io_TxtIStream_atEnd__exec },
    { "deleteRef", sidlx_io_TxtIStream_deleteRef__exec },
    { "getBool", sidlx_io_TxtIStream_getBool__exec },
    { "getChar", sidlx_io_TxtIStream_getChar__exec },
    { "getClassInfo", sidlx_io_TxtIStream_getClassInfo__exec },
    { "getDcomplex", sidlx_io_TxtIStream_getDcomplex__exec },
    { "getDouble", sidlx_io_TxtIStream_getDouble__exec },
    { "getFcomplex", sidlx_io_TxtIStream_getFcomplex__exec },
    { "getFloat", sidlx_io_TxtIStream_getFloat__exec },
    { "getInt", sidlx_io_TxtIStream_getInt__exec },
    { "getLong", sidlx_io_TxtIStream_getLong__exec },
    { "getString", sidlx_io_TxtIStream_getString__exec },
    { "isSame", sidlx_io_TxtIStream_isSame__exec },
    { "isType", sidlx_io_TxtIStream_isType__exec },
    { "queryInt", sidlx_io_TxtIStream_queryInt__exec },
    { "read", sidlx_io_TxtIStream_read__exec },
    { "readline", sidlx_io_TxtIStream_readline__exec },
    { "setFD", sidlx_io_TxtIStream_setFD__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_io_TxtIStream__method);
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

static void sidlx_io_TxtIStream__init_epv(
  struct sidlx_io_TxtIStream__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_io_TxtIStream__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  struct sidlx_io_TxtIStream__epv*  epv  = &s_new_epv__sidlx_io_txtistream;
  struct sidl_BaseClass__epv*       e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*   e1   = &s_new_epv__sidl_baseinterface;
  struct sidlx_io_IStream__epv*     e2   = &s_new_epv__sidlx_io_istream;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_sidlx_io_TxtIStream__cast;
  epv->f__delete                  = ior_sidlx_io_TxtIStream__delete;
  epv->f__exec                    = ior_sidlx_io_TxtIStream__exec;
  epv->f__getURL                  = ior_sidlx_io_TxtIStream__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_io_TxtIStream__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_io_TxtIStream__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_io_TxtIStream__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidlx_io_TxtIStream__object*,const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_io_TxtIStream__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_io_TxtIStream__object*)) s1->d_epv->f_getClassInfo;
  epv->f_setFD                    = NULL;
  epv->f_atEnd                    = NULL;
  epv->f_read                     = NULL;
  epv->f_readline                 = NULL;
  epv->f_getBool                  = NULL;
  epv->f_getChar                  = NULL;
  epv->f_getInt                   = NULL;
  epv->f_getLong                  = NULL;
  epv->f_getFloat                 = NULL;
  epv->f_getDouble                = NULL;
  epv->f_getFcomplex              = NULL;
  epv->f_getDcomplex              = NULL;
  epv->f_getString                = NULL;

  sidlx_io_TxtIStream__set_epv(epv);

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
  e2->f_atEnd               = (sidl_bool (*)(void*)) epv->f_atEnd;
  e2->f_read                = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_read;
  e2->f_readline            = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e2->f_getBool             = (void (*)(void*,sidl_bool*,
    struct sidl_BaseInterface__object **)) epv->f_getBool;
  e2->f_getChar             = (void (*)(void*,char*,
    struct sidl_BaseInterface__object **)) epv->f_getChar;
  e2->f_getInt              = (void (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_getInt;
  e2->f_getLong             = (void (*)(void*,int64_t*,
    struct sidl_BaseInterface__object **)) epv->f_getLong;
  e2->f_getFloat            = (void (*)(void*,float*,
    struct sidl_BaseInterface__object **)) epv->f_getFloat;
  e2->f_getDouble           = (void (*)(void*,double*,
    struct sidl_BaseInterface__object **)) epv->f_getDouble;
  e2->f_getFcomplex         = (void (*)(void*,struct sidl_fcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getFcomplex;
  e2->f_getDcomplex         = (void (*)(void*,struct sidl_dcomplex*,
    struct sidl_BaseInterface__object **)) epv->f_getDcomplex;
  e2->f_getString           = (void (*)(void*,char**,
    struct sidl_BaseInterface__object **)) epv->f_getString;

  s_method_initialized = 1;
  ior_sidlx_io_TxtIStream__ensure_load_called();
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidlx_io_TxtIStream__super(void) {
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
      sidl_ClassInfoI_setName(impl, "sidlx.io.TxtIStream");
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
initMetadata(struct sidlx_io_TxtIStream__object* self)
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

struct sidlx_io_TxtIStream__object*
sidlx_io_TxtIStream__new(void)
{
  struct sidlx_io_TxtIStream__object* self =
    (struct sidlx_io_TxtIStream__object*) malloc(
      sizeof(struct sidlx_io_TxtIStream__object));
  sidlx_io_TxtIStream__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_io_TxtIStream__init(
  struct sidlx_io_TxtIStream__object* self)
{
  struct sidlx_io_TxtIStream__object* s0 = self;
  struct sidl_BaseClass__object*      s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_io_TxtIStream__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_sidlx_io_istream.d_epv = &s_new_epv__sidlx_io_istream;
  s0->d_epv                    = &s_new_epv__sidlx_io_txtistream;

  s0->d_sidlx_io_istream.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidlx_io_TxtIStream__fini(
  struct sidlx_io_TxtIStream__object* self)
{
  struct sidlx_io_TxtIStream__object* s0 = self;
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
sidlx_io_TxtIStream__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_io_TxtIStream__external
s_externalEntryPoints = {
  sidlx_io_TxtIStream__new,
  sidlx_io_TxtIStream__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_io_TxtIStream__external*
sidlx_io_TxtIStream__externals(void)
{
  return &s_externalEntryPoints;
}

