/*
 * File:          bHYPRE_Operator_IOR.h
 * Symbol:        bHYPRE.Operator-v1.0.0
 * Symbol Type:   interface
 * Babel Version: 0.9.8
 * sidl Created:  20050225 15:45:37 PST
 * Generated:     20050225 15:45:37 PST
 * Description:   Intermediate Object Representation for bHYPRE.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.9.8
 * source-line   = 590
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
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

extern struct bHYPRE_Operator__object*
bHYPRE_Operator__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct bHYPRE_Vector__array;
struct bHYPRE_Vector__object;
struct sidl_BaseInterface__array;
struct sidl_BaseInterface__object;
struct sidl_ClassInfo__array;
struct sidl_ClassInfo__object;

/*
 * Declare the method entry point vector.
 */

struct bHYPRE_Operator__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in sidl.BaseInterface-v0.9.0 */
  void (*f_addRef)(
    void* self);
  void (*f_deleteRef)(
    void* self);
  sidl_bool (*f_isSame)(
    void* self,
    struct sidl_BaseInterface__object* iobj);
  struct sidl_BaseInterface__object* (*f_queryInt)(
    void* self,
    const char* name);
  sidl_bool (*f_isType)(
    void* self,
    const char* name);
  struct sidl_ClassInfo__object* (*f_getClassInfo)(
    void* self);
  /* Methods introduced in bHYPRE.Operator-v1.0.0 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* mpi_comm);
  int32_t (*f_SetIntParameter)(
    void* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetDoubleParameter)(
    void* self,
    const char* name,
    double value);
  int32_t (*f_SetStringParameter)(
    void* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArray1Parameter)(
    void* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetIntArray2Parameter)(
    void* self,
    const char* name,
    struct sidl_int__array* value);
  int32_t (*f_SetDoubleArray1Parameter)(
    void* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_SetDoubleArray2Parameter)(
    void* self,
    const char* name,
    struct sidl_double__array* value);
  int32_t (*f_GetIntValue)(
    void* self,
    const char* name,
    int32_t* value);
  int32_t (*f_GetDoubleValue)(
    void* self,
    const char* name,
    double* value);
  int32_t (*f_Setup)(
    void* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object* x);
  int32_t (*f_Apply)(
    void* self,
    struct bHYPRE_Vector__object* b,
    struct bHYPRE_Vector__object** x);
};

/*
 * Define the interface object structure.
 */

struct bHYPRE_Operator__object {
  struct bHYPRE_Operator__epv* d_epv;
  void*                        d_object;
};

#ifdef __cplusplus
}
#endif
#endif
