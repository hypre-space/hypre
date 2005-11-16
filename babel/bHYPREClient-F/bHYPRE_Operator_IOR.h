/*
 * File:          bHYPRE_Operator_IOR.h
 * Symbol:        bHYPRE.Operator-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.10.4
 * Description:   Intermediate Object Representation for bHYPRE.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.10.4
 */

#ifndef included_bHYPRE_Operator_IOR_h
#define included_bHYPRE_Operator_IOR_h

#ifndef included_sidl_header_h
#include "sidl_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "bHYPRE.Operator" (version 1.0.0)
 * 
 * An Operator is anything that maps one Vector to another.  The
 * terms {\tt Setup} and {\tt Apply} are reserved for Operators.
 * The implementation is allowed to assume that supplied parameter
 * arrays will not be destroyed.
 * 
 */

struct bHYPRE_Operator__array;
struct bHYPRE_Operator__object;

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_MPICommunicator__array;
struct bHYPRE_MPICommunicator__object;
struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
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

struct bHYPRE_Operator__epv {
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
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ void* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_SetIntParameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ int32_t value);
  int32_t (*f_SetDoubleParameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ double value);
  int32_t (*f_SetStringParameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ void* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ int32_t* value);
  int32_t (*f_GetDoubleValue)(
    /* in */ void* self,
    /* in */ const char* name,
    /* out */ double* value);
  int32_t (*f_Setup)(
    /* in */ void* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    /* in */ void* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  int32_t (*f_ApplyAdjoint)(
    /* in */ void* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_Operator__object {
  struct bHYPRE_Operator__epv* d_epv;
  void*                        d_object;
};

/**
 * 
 * 
 * Anonymous class definition
 * 
 * 
 */
#ifndef included_bHYPRE_Operator_IOR_h
#include "bHYPRE_Operator_IOR.h"
#endif
#ifndef included_sidl_BaseInterface_IOR_h
#include "sidl_BaseInterface_IOR.h"
#endif

/*
 * Symbol "bHYPRE._Operator" (version 1.0)
 */

struct bHYPRE__Operator__array;
struct bHYPRE__Operator__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE__Operator__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name);
  void (*f__delete)(
    /* in */ struct bHYPRE__Operator__object* self);
  void (*f__exec)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* methodName,
    /* in */ struct sidl_io_Deserializer__object* inArgs,
    /* in */ struct sidl_io_Serializer__object* outArgs);
  char* (*f__getURL)(
    /* in */ struct bHYPRE__Operator__object* self);
  void (*f__ctor)(
    /* in */ struct bHYPRE__Operator__object* self);
  void (*f__dtor)(
    /* in */ struct bHYPRE__Operator__object* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.3 */
  void (*f_addRef)(
    /* in */ struct bHYPRE__Operator__object* self);
  void (*f_deleteRef)(
    /* in */ struct bHYPRE__Operator__object* self);
  sidl_bool (*f_isSame)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name);
  sidl_bool (*f_isType)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    /* in */ struct bHYPRE__Operator__object* self);
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ struct bHYPRE_MPICommunicator__object* mpi_comm);
  int32_t (*f_SetIntParameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ int32_t value);
  int32_t (*f_SetDoubleParameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ double value);
  int32_t (*f_SetStringParameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* in */ struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* out */ int32_t* value);
  int32_t (*f_GetDoubleValue)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ const char* name,
    /* out */ double* value);
  int32_t (*f_Setup)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* in */ struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  int32_t (*f_ApplyAdjoint)(
    /* in */ struct bHYPRE__Operator__object* self,
    /* in */ struct bHYPRE_Vector__object* b,
    /* inout */ struct bHYPRE_Vector__object** x);
  /* Methods introduced in bHYPRE._Operator-v1.0 */
};

/*
 * Define the class object structure.
 */

struct bHYPRE__Operator__object {
  struct bHYPRE_Operator__object    d_bhypre_operator;
  struct sidl_BaseInterface__object d_sidl_baseinterface;
  struct bHYPRE__Operator__epv*     d_epv;
  void*                             d_data;
};


#ifdef __cplusplus
}
#endif
#endif
