/*
 * File:          Hypre_Operator_IOR.h
 * Symbol:        Hypre.Operator-v0.1.5
 * Symbol Type:   interface
 * Babel Version: 0.7.4
 * SIDL Created:  20021101 15:14:28 PST
 * Generated:     20021101 15:14:31 PST
 * Description:   Intermediate Object Representation for Hypre.Operator
 * 
 * WARNING: Automatically generated; changes will be lost
 * 
 * babel-version = 0.7.4
 * source-line   = 327
 * source-url    = file:/home/painter/linear_solvers/babel/Interfaces.idl
 */

#ifndef included_Hypre_Operator_IOR_h
#define included_Hypre_Operator_IOR_h

#ifndef included_SIDL_header_h
#include "SIDL_header.h"
#endif
#ifdef __cplusplus
extern "C" {
#endif

/*
 * Symbol "Hypre.Operator" (version 0.1.5)
 * 
 * An Operator is anything that maps one Vector to another.
 * The terms "Setup" and "Apply" are reserved for Operators.
 * The implementation is allowed to assume that supplied parameter
 * arrays will not be destroyed.
 */

struct Hypre_Operator__array;
struct Hypre_Operator__object;

extern struct Hypre_Operator__object*
Hypre_Operator__remote(const char *url);

/*
 * Forward references for external classes and interfaces.
 */

struct Hypre_Vector__array;
struct Hypre_Vector__object;
struct SIDL_BaseInterface__array;
struct SIDL_BaseInterface__object;

/*
 * Declare the method entry point vector.
 */

struct Hypre_Operator__epv {
  /* Implicit builtin methods */
  void* (*f__cast)(
    void* self,
    const char* name);
  void (*f__delete)(
    void* self);
  /* Methods introduced in SIDL.BaseInterface-v0.7.4 */
  void (*f_addReference)(
    void* self);
  void (*f_deleteReference)(
    void* self);
  SIDL_bool (*f_isSame)(
    void* self,
    struct SIDL_BaseInterface__object* iobj);
  struct SIDL_BaseInterface__object* (*f_queryInterface)(
    void* self,
    const char* name);
  SIDL_bool (*f_isInstanceOf)(
    void* self,
    const char* name);
  /* Methods introduced in Hypre.Operator-v0.1.5 */
  int32_t (*f_SetCommunicator)(
    void* self,
    void* comm);
  int32_t (*f_GetDoubleValue)(
    void* self,
    const char* name,
    double* value);
  int32_t (*f_GetIntValue)(
    void* self,
    const char* name,
    int32_t* value);
  int32_t (*f_SetDoubleParameter)(
    void* self,
    const char* name,
    double value);
  int32_t (*f_SetIntParameter)(
    void* self,
    const char* name,
    int32_t value);
  int32_t (*f_SetStringParameter)(
    void* self,
    const char* name,
    const char* value);
  int32_t (*f_SetIntArrayParameter)(
    void* self,
    const char* name,
    struct SIDL_int__array* value);
  int32_t (*f_SetDoubleArrayParameter)(
    void* self,
    const char* name,
    struct SIDL_double__array* value);
  int32_t (*f_Setup)(
    void* self,
    struct Hypre_Vector__object* x,
    struct Hypre_Vector__object* y);
  int32_t (*f_Apply)(
    void* self,
    struct Hypre_Vector__object* x,
    struct Hypre_Vector__object** y);
};

/*
 * Define the interface object structure.
 */

struct Hypre_Operator__object {
  struct Hypre_Operator__epv* d_epv;
  void*                       d_object;
};

#ifdef __cplusplus
}
#endif
#endif
