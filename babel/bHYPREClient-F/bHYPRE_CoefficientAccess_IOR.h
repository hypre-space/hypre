/*
 * File:          bHYPRE_CoefficientAccess_IOR.h
 * Symbol:        bHYPRE.CoefficientAccess-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.8
 * Description:   Intermediate Object Representation for bHYPRE.CoefficientAccess
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.8
 */

#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#define included_bHYPRE_CoefficientAccess_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.CoefficientAccess" (version 1.0.0)
 */

struct bHYPRE_CoefficientAccess__array;
struct bHYPRE_CoefficientAccess__object;

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

struct bHYPRE_CoefficientAccess__epv {
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
  /* Methods introduced in bHYPRE.CoefficientAccess-v1.0.0 */
  int32_t (*f_GetRow)(
    /* in */ void* self,
    /* in */ int32_t row,
    /* out */ int32_t* size,
    /* out array<int,column-major> */ struct sidl_int__array** col_ind,
    /* out array<double,column-major> */ struct sidl_double__array** values);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_CoefficientAccess__object {
  struct bHYPRE_CoefficientAccess__epv* d_epv;
  void*                                 d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_bHYPRE_CoefficientAccess_IOR_h
#include "bHYPRE_CoefficientAccess_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

/*
 * Symbol "bHYPRE._CoefficientAccess" (version 1.0)
 */

struct bHYPRE__CoefficientAccess__array;
struct bHYPRE__CoefficientAccess__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE__CoefficientAccess__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self);
  /* Methods introduced in bHYPRE.CoefficientAccess-v1.0.0 */
  int32_t (*f_GetRow)(
    /* in */ struct bHYPRE__CoefficientAccess__object* self,
    /* in */ int32_t row,
    /* out */ int32_t* size,
    /* out array<int,column-major> */ struct sidl_int__array** col_ind,
    /* out array<double,column-major> */ struct sidl_double__array** values);
  /* Methods introduced in bHYPRE._CoefficientAccess-v1.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE__CoefficientAccess__object {
  struct bHYPRE_CoefficientAccess__object d_bhypre_coefficientaccess;
  struct sidl_BaseInterface__object       d_sidl_baseinterface;
  struct bHYPRE__CoefficientAccess__epv*  d_epv;
  void*                                   d_data;
};


#ifdef __cplusplus
}
#endif
#endif
