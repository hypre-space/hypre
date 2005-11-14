/*
 * File:          sidlx_io_Serializable_IOR.h
 * Symbol:        sidlx.io.Serializable-v0.1
 * Symbol Type:   interface
 * Babel Version: 0.10.12
 * Description:   Intermediate Object Representation for sidlx.io.Serializable
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.12
 */

#ifndef included_sidlx_io_Serializable_IOR_h
#define included_sidlx_io_Serializable_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "sidlx.io.Serializable" (version 0.1)
 */

struct sidlx_io_Serializable__array;
struct sidlx_io_Serializable__object;

/*
 * Forward references for external classes and interfaces.
 */

struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;
struct sidl_io_Deserializer__array;
struct sidl_io_Deserializer__object;
struct sidl_io_Serializer__array;
struct sidl_io_Serializer__object;
struct sidlx_io_IStream__array;
struct sidlx_io_IStream__object;
struct sidlx_io_OStream__array;
struct sidlx_io_OStream__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_io_Serializable__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ void* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ void* self);
  void (*f__exec)(
    /* in */ void* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ void* self);
  void (*f_deleteRef)(
    /* in */ void* self);
  sidl_bool (*f_isSame)(
    /* in */ void* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ void* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ void* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ void* self);
  /* Methods introduced in sidlx.io.Serializable-v0.1 */
  void (*f_pack)(
    /* in */ void* self,
    /* in */ struct sidlx_io_OStream__object* ostr);
  void (*f_unpack)(
    /* in */ void* self,
    /* in */ struct sidlx_io_IStream__object* istr);
};

/*
 * Define the interface object structure.
 */

struct sidlx_io_Serializable__object {
  struct sidlx_io_Serializable__epv* d_epv;
  void*                              d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif
#ifndef included_sidlx_io_Serializable_IOR_h
#include "sidlx_io_Serializable_IOR.h"
#endif

/*
 * Symbol "sidlx.io._Serializable" (version 1.0)
 */

struct sidlx_io__Serializable__array;
struct sidlx_io__Serializable__object;

/*
 * Declare the method entry point vector.
 */

struct sidlx_io__Serializable__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct sidlx_io__Serializable__object* self);
  void (*f__exec)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct sidlx_io__Serializable__object* self);
  void (*f__ctor)(
    /* in */ struct sidlx_io__Serializable__object* self);
  void (*f__dtor)(
    /* in */ struct sidlx_io__Serializable__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct sidlx_io__Serializable__object* self);
  void (*f_deleteRef)(
    /* in */ struct sidlx_io__Serializable__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct sidlx_io__Serializable__object* self);
  /* Methods introduced in sidlx.io.Serializable-v0.1 */
  void (*f_pack)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ struct sidlx_io_OStream__object* ostr);
  void (*f_unpack)(
    /* in */ struct sidlx_io__Serializable__object* self,
    /* in */ struct sidlx_io_IStream__object* istr);
  /* Methods introduced in sidlx.io._Serializable-v1.0 */
};

/*
 * Define the class object structure.
 */

struct sidlx_io__Serializable__object {
  struct sidl_BaseInterface__object    d_sidl_baseinterface;
  struct sidlx_io_Serializable__object d_sidlx_io_serializable;
  struct sidlx_io__Serializable__epv*  d_epv;
  void*                                d_data;
};


#ifdef __cplusplus
}
#endif
#endif
