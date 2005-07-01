/*
 * File:          bHYPRE_Vector_IOR.h
 * Symbol:        bHYPRE.Vector-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.Vector
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_Vector_IOR_h
#define included_bHYPRE_Vector_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.Vector" (version 1.0.0)
 */

struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;

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

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_Vector__epv {
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
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    /* in */ void* self);
  int32_t (*f_Copy)(
    /* in */ void* self,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    /* in */ void* self,
    /* out */ struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    /* in */ void* self,
    /* in */ double a);
  int32_t (*f_Dot)(
    /* in */ void* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ double* d);
  int32_t (*f_Axpy)(
    /* in */ void* self,
    /* in */ double a,
    /* in */ struct bHYPRE_Vector__object* x);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_Vector__object {
  struct bHYPRE_Vector__epv* d_epv;
  void*                      d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_bHYPRE_Vector_IOR_h
#include "bHYPRE_Vector_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

/*
 * Symbol "bHYPRE._Vector" (version 1.0)
 */

struct bHYPRE__Vector__array;
struct bHYPRE__Vector__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE__Vector__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE__Vector__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE__Vector__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE__Vector__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE__Vector__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE__Vector__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE__Vector__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE__Vector__object* self);
  /* Methods introduced in bHYPRE.Vector-v1.0.0 */
  int32_t (*f_Clear)(
    /* in */ struct bHYPRE__Vector__object* self);
  int32_t (*f_Copy)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Clone)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* out */ struct bHYPRE_Vector__object** x);
  int32_t (*f_Scale)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ double a);
  int32_t (*f_Dot)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ struct bHYPRE_Vector__object* x,
    /* out */ double* d);
  int32_t (*f_Axpy)(
    /* in */ struct bHYPRE__Vector__object* self,
    /* in */ double a,
    /* in */ struct bHYPRE_Vector__object* x);
  /* Methods introduced in bHYPRE._Vector-v1.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE__Vector__object {
  struct bHYPRE_Vector__object      d_bhypre_vector;
  struct sidl_BaseInterface__object d_sidl_baseinterface;
  struct bHYPRE__Vector__epv*       d_epv;
  void*                             d_data;
};


#ifdef __cplusplus
}
#endif
#endif
