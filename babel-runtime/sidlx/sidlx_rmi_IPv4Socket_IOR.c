/*
 * File:          sidlx_rmi_IPv4Socket_IOR.c
 * Symbol:        sidlx.rmi.IPv4Socket-v0.1
 * Symbol Type:   class
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for sidlx.rmi.IPv4Socket
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#include "sidl_rmi_InstanceHandle.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include "sidlx_rmi_IPv4Socket_IOR.h"
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
static struct sidl_recursive_mutex_t sidlx_rmi_IPv4Socket__mutex= SIDL_RECURSIVE_MUTEX_INITIALIZER;
#define LOCK_STATIC_GLOBALS sidl_recursive_mutex_lock( &sidlx_rmi_IPv4Socket__mutex )
#define UNLOCK_STATIC_GLOBALS sidl_recursive_mutex_unlock( &sidlx_rmi_IPv4Socket__mutex )
/* #define HAVE_LOCKED_STATIC_GLOBALS (sidl_recursive_mutex_trylock( &sidlx_rmi_IPv4Socket__mutex )==EDEADLOCK) */
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

static struct sidlx_rmi_IPv4Socket__epv s_new_epv__sidlx_rmi_ipv4socket;

static struct sidl_BaseClass__epv  s_new_epv__sidl_baseclass;
static struct sidl_BaseClass__epv* s_old_epv__sidl_baseclass;

static struct sidl_BaseInterface__epv  s_new_epv__sidl_baseinterface;
static struct sidl_BaseInterface__epv* s_old_epv__sidl_baseinterface;

static struct sidlx_rmi_Socket__epv s_new_epv__sidlx_rmi_socket;

/*
 * Declare EPV routines defined in the skeleton file.
 */

#ifdef __cplusplus
extern "C" {
#endif

extern void sidlx_rmi_IPv4Socket__set_epv(
  struct sidlx_rmi_IPv4Socket__epv* epv);
extern void sidlx_rmi_IPv4Socket__call_load(void);
#ifdef __cplusplus
}
#endif

static void
sidlx_rmi_IPv4Socket_addRef__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
sidlx_rmi_IPv4Socket_deleteRef__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
sidlx_rmi_IPv4Socket_isSame__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
sidlx_rmi_IPv4Socket_queryInt__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
sidlx_rmi_IPv4Socket_isType__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
sidlx_rmi_IPv4Socket_getClassInfo__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
sidlx_rmi_IPv4Socket_getsockname__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t address_tmp;
  int32_t* address= &address_tmp;
  int32_t port_tmp;
  int32_t* port= &port_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "address", address, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "port", port, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_getsockname)(
    self,
    address,
    port,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "address", *address, _ex2);
  sidl_io_Serializer_packInt( outArgs, "port", *port, _ex2);

}

static void
sidlx_rmi_IPv4Socket_getpeername__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t address_tmp;
  int32_t* address= &address_tmp;
  int32_t port_tmp;
  int32_t* port= &port_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "address", address, _ex2);
  sidl_io_Deserializer_unpackInt( inArgs, "port", port, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_getpeername)(
    self,
    address,
    port,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "address", *address, _ex2);
  sidl_io_Serializer_packInt( outArgs, "port", *port, _ex2);

}

static void
sidlx_rmi_IPv4Socket_close__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_close)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void
sidlx_rmi_IPv4Socket_readn__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
  _retval = (self->d_epv->f_readn)(
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
sidlx_rmi_IPv4Socket_readline__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
  _retval = (self->d_epv->f_readline)(
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
sidlx_rmi_IPv4Socket_readstring__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
  _retval = (self->d_epv->f_readstring)(
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
sidlx_rmi_IPv4Socket_readstring_alloc__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
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
  _retval = (self->d_epv->f_readstring_alloc)(
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
sidlx_rmi_IPv4Socket_readint__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t data_tmp;
  int32_t* data= &data_tmp;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "data", data, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_readint)(
    self,
    data,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */
  sidl_io_Serializer_packInt( outArgs, "data", *data, _ex2);

}

static void
sidlx_rmi_IPv4Socket_writen__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t nbytes;
  struct sidl_char__array* data;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "nbytes", &nbytes, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_writen)(
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
sidlx_rmi_IPv4Socket_writestring__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t nbytes;
  struct sidl_char__array* data;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "nbytes", &nbytes, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_writestring)(
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
sidlx_rmi_IPv4Socket_writeint__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t data;
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "data", &data, _ex2);

  /* make the call */
  _retval = (self->d_epv->f_writeint)(
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
sidlx_rmi_IPv4Socket_setFileDescriptor__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t fd;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */
  sidl_io_Deserializer_unpackInt( inArgs, "fd", &fd, _ex2);

  /* make the call */
  (self->d_epv->f_setFileDescriptor)(
    self,
    fd,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  /* pack out and inout argments */

}

static void
sidlx_rmi_IPv4Socket_getFileDescriptor__exec(
        struct sidlx_rmi_IPv4Socket__object* self,
        struct sidl_io_Deserializer__object* inArgs,
        struct sidl_io_Serializer__object* outArgs) {
  /* stack space for arguments */
  int32_t _retval;
  sidl_BaseInterface _ex   = NULL;
  sidl_BaseInterface *_ex2 = &_ex;
  /* unpack in and inout argments */

  /* make the call */
  _retval = (self->d_epv->f_getFileDescriptor)(
    self,
    _ex2);

  /* check if exception thrown */
  /* FIXME */

  /* pack return value */
  sidl_io_Serializer_packInt( outArgs, "_retval", _retval, _ex2);

  /* pack out and inout argments */

}

static void ior_sidlx_rmi_IPv4Socket__ensure_load_called(void) {
  /*
   * assert( HAVE_LOCKED_STATIC_GLOBALS );
   */

  if (! s_load_called ) {
    sidlx_rmi_IPv4Socket__call_load();
    s_load_called=1;
  }
}
/*
 * CAST: dynamic type casting support.
 */

static void* ior_sidlx_rmi_IPv4Socket__cast(
  struct sidlx_rmi_IPv4Socket__object* self,
  const char* name)
{
  void* cast = NULL;

  struct sidlx_rmi_IPv4Socket__object* s0 = self;
  struct sidl_BaseClass__object*       s1 = &s0->d_sidl_baseclass;

  if (!strcmp(name, "sidlx.rmi.IPv4Socket")) {
    cast = (void*) s0;
  } else if (!strcmp(name, "sidlx.rmi.Socket")) {
    cast = (void*) &s0->d_sidlx_rmi_socket;
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

static void ior_sidlx_rmi_IPv4Socket__delete(
  struct sidlx_rmi_IPv4Socket__object* self)
{
  sidlx_rmi_IPv4Socket__fini(self);
  memset((void*)self, 0, sizeof(struct sidlx_rmi_IPv4Socket__object));
  free((void*) self);
}

static char*
ior_sidlx_rmi_IPv4Socket__getURL(
    struct sidlx_rmi_IPv4Socket__object* self) {
  sidl_rmi_InstanceHandle conn = (sidl_rmi_InstanceHandle)self->d_data;
  sidl_BaseInterface _ex = NULL;
  /* TODO: Make this work for local object! */
  return NULL;
}
struct sidlx_rmi_IPv4Socket__method {
  const char *d_name;
  void (*d_func)(struct sidlx_rmi_IPv4Socket__object*,
    struct sidl_io_Deserializer__object *,
    struct sidl_io_Serializer__object *);
};

static void
ior_sidlx_rmi_IPv4Socket__exec(
    struct sidlx_rmi_IPv4Socket__object* self,
    const char* methodName,
    struct sidl_io_Deserializer__object* inArgs,
    struct sidl_io_Serializer__object* outArgs ) { 
  static const struct sidlx_rmi_IPv4Socket__method  s_methods[] = {
    { "addRef", sidlx_rmi_IPv4Socket_addRef__exec },
    { "close", sidlx_rmi_IPv4Socket_close__exec },
    { "deleteRef", sidlx_rmi_IPv4Socket_deleteRef__exec },
    { "getClassInfo", sidlx_rmi_IPv4Socket_getClassInfo__exec },
    { "getFileDescriptor", sidlx_rmi_IPv4Socket_getFileDescriptor__exec },
    { "getpeername", sidlx_rmi_IPv4Socket_getpeername__exec },
    { "getsockname", sidlx_rmi_IPv4Socket_getsockname__exec },
    { "isSame", sidlx_rmi_IPv4Socket_isSame__exec },
    { "isType", sidlx_rmi_IPv4Socket_isType__exec },
    { "queryInt", sidlx_rmi_IPv4Socket_queryInt__exec },
    { "readint", sidlx_rmi_IPv4Socket_readint__exec },
    { "readline", sidlx_rmi_IPv4Socket_readline__exec },
    { "readn", sidlx_rmi_IPv4Socket_readn__exec },
    { "readstring", sidlx_rmi_IPv4Socket_readstring__exec },
    { "readstring_alloc", sidlx_rmi_IPv4Socket_readstring_alloc__exec },
    { "setFileDescriptor", sidlx_rmi_IPv4Socket_setFileDescriptor__exec },
    { "writeint", sidlx_rmi_IPv4Socket_writeint__exec },
    { "writen", sidlx_rmi_IPv4Socket_writen__exec },
    { "writestring", sidlx_rmi_IPv4Socket_writestring__exec }
  };
  int i, cmp, l = 0;
  int u = sizeof(s_methods)/sizeof(struct sidlx_rmi_IPv4Socket__method);
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

static void sidlx_rmi_IPv4Socket__init_epv(
  struct sidlx_rmi_IPv4Socket__object* self)
{
/*
 * assert( HAVE_LOCKED_STATIC_GLOBALS );
 */

  struct sidlx_rmi_IPv4Socket__object* s0 = self;
  struct sidl_BaseClass__object*       s1 = &s0->d_sidl_baseclass;

  struct sidlx_rmi_IPv4Socket__epv*  epv  = &s_new_epv__sidlx_rmi_ipv4socket;
  struct sidl_BaseClass__epv*        e0   = &s_new_epv__sidl_baseclass;
  struct sidl_BaseInterface__epv*    e1   = &s_new_epv__sidl_baseinterface;
  struct sidlx_rmi_Socket__epv*      e2   = &s_new_epv__sidlx_rmi_socket;

  s_old_epv__sidl_baseinterface = s1->d_sidl_baseinterface.d_epv;
  s_old_epv__sidl_baseclass     = s1->d_epv;

  epv->f__cast                    = ior_sidlx_rmi_IPv4Socket__cast;
  epv->f__delete                  = ior_sidlx_rmi_IPv4Socket__delete;
  epv->f__exec                    = ior_sidlx_rmi_IPv4Socket__exec;
  epv->f__getURL                  = ior_sidlx_rmi_IPv4Socket__getURL;
  epv->f__ctor                    = NULL;
  epv->f__dtor                    = NULL;
  epv->f_addRef                   = (void (*)(struct 
    sidlx_rmi_IPv4Socket__object*)) s1->d_epv->f_addRef;
  epv->f_deleteRef                = (void (*)(struct 
    sidlx_rmi_IPv4Socket__object*)) s1->d_epv->f_deleteRef;
  epv->f_isSame                   = (sidl_bool (*)(struct 
    sidlx_rmi_IPv4Socket__object*,
    struct sidl_BaseInterface__object*)) s1->d_epv->f_isSame;
  epv->f_queryInt                 = (struct sidl_BaseInterface__object* 
    (*)(struct sidlx_rmi_IPv4Socket__object*,
    const char*)) s1->d_epv->f_queryInt;
  epv->f_isType                   = (sidl_bool (*)(struct 
    sidlx_rmi_IPv4Socket__object*,const char*)) s1->d_epv->f_isType;
  epv->f_getClassInfo             = (struct sidl_ClassInfo__object* (*)(struct 
    sidlx_rmi_IPv4Socket__object*)) s1->d_epv->f_getClassInfo;
  epv->f_getsockname              = NULL;
  epv->f_getpeername              = NULL;
  epv->f_close                    = NULL;
  epv->f_readn                    = NULL;
  epv->f_readline                 = NULL;
  epv->f_readstring               = NULL;
  epv->f_readstring_alloc         = NULL;
  epv->f_readint                  = NULL;
  epv->f_writen                   = NULL;
  epv->f_writestring              = NULL;
  epv->f_writeint                 = NULL;
  epv->f_setFileDescriptor        = NULL;
  epv->f_getFileDescriptor        = NULL;

  sidlx_rmi_IPv4Socket__set_epv(epv);

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
  e2->f_close               = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_close;
  e2->f_readn               = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readn;
  e2->f_readline            = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readline;
  e2->f_readstring          = (int32_t (*)(void*,int32_t,
    struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring;
  e2->f_readstring_alloc    = (int32_t (*)(void*,struct sidl_char__array**,
    struct sidl_BaseInterface__object **)) epv->f_readstring_alloc;
  e2->f_readint             = (int32_t (*)(void*,int32_t*,
    struct sidl_BaseInterface__object **)) epv->f_readint;
  e2->f_writen              = (int32_t (*)(void*,int32_t,
    struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_writen;
  e2->f_writestring         = (int32_t (*)(void*,int32_t,
    struct sidl_char__array*,
    struct sidl_BaseInterface__object **)) epv->f_writestring;
  e2->f_writeint            = (int32_t (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_writeint;
  e2->f_setFileDescriptor   = (void (*)(void*,int32_t,
    struct sidl_BaseInterface__object **)) epv->f_setFileDescriptor;
  e2->f_getFileDescriptor   = (int32_t (*)(void*,
    struct sidl_BaseInterface__object **)) epv->f_getFileDescriptor;

  s_method_initialized = 1;
  ior_sidlx_rmi_IPv4Socket__ensure_load_called();
}

/*
 * SUPER: return's parent's non-overrided EPV
 */

static struct sidl_BaseClass__epv* sidlx_rmi_IPv4Socket__super(void) {
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
      sidl_ClassInfoI_setName(impl, "sidlx.rmi.IPv4Socket");
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
initMetadata(struct sidlx_rmi_IPv4Socket__object* self)
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

struct sidlx_rmi_IPv4Socket__object*
sidlx_rmi_IPv4Socket__new(void)
{
  struct sidlx_rmi_IPv4Socket__object* self =
    (struct sidlx_rmi_IPv4Socket__object*) malloc(
      sizeof(struct sidlx_rmi_IPv4Socket__object));
  sidlx_rmi_IPv4Socket__init(self);
  initMetadata(self);
  return self;
}

/*
 * INIT: initialize a new instance of the class object.
 */

void sidlx_rmi_IPv4Socket__init(
  struct sidlx_rmi_IPv4Socket__object* self)
{
  struct sidlx_rmi_IPv4Socket__object* s0 = self;
  struct sidl_BaseClass__object*       s1 = &s0->d_sidl_baseclass;

  sidl_BaseClass__init(s1);

  LOCK_STATIC_GLOBALS;
  if (!s_method_initialized) {
    sidlx_rmi_IPv4Socket__init_epv(s0);
  }
  UNLOCK_STATIC_GLOBALS;

  s1->d_sidl_baseinterface.d_epv = &s_new_epv__sidl_baseinterface;
  s1->d_epv                      = &s_new_epv__sidl_baseclass;

  s0->d_sidlx_rmi_socket.d_epv = &s_new_epv__sidlx_rmi_socket;
  s0->d_epv                    = &s_new_epv__sidlx_rmi_ipv4socket;

  s0->d_sidlx_rmi_socket.d_object = self;

  s0->d_data = NULL;


  (*(self->d_epv->f__ctor))(self);
}

/*
 * FINI: deallocate a class instance (destructor).
 */

void sidlx_rmi_IPv4Socket__fini(
  struct sidlx_rmi_IPv4Socket__object* self)
{
  struct sidlx_rmi_IPv4Socket__object* s0 = self;
  struct sidl_BaseClass__object*       s1 = &s0->d_sidl_baseclass;

  (*(s0->d_epv->f__dtor))(s0);

  s1->d_sidl_baseinterface.d_epv = s_old_epv__sidl_baseinterface;
  s1->d_epv                      = s_old_epv__sidl_baseclass;

  sidl_BaseClass__fini(s1);
}

/*
 * VERSION: Return the version of the IOR used to generate this IOR.
 */

void
sidlx_rmi_IPv4Socket__IOR_version(int32_t *major, int32_t *minor)
{
  *major = s_IOR_MAJOR_VERSION;
  *minor = s_IOR_MINOR_VERSION;
}

static const struct sidlx_rmi_IPv4Socket__external
s_externalEntryPoints = {
  sidlx_rmi_IPv4Socket__new,
  sidlx_rmi_IPv4Socket__super
};

/*
 * This function returns a pointer to a static structure of
 * pointers to function entry points.  Its purpose is to provide
 * one-stop shopping for loading DLLs.
 */

const struct sidlx_rmi_IPv4Socket__external*
sidlx_rmi_IPv4Socket__externals(void)
{
  return &s_externalEntryPoints;
}

